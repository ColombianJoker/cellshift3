#!/usr/bin/env python3 

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from typing import Generator, Union, Optional, List, Tuple
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def set_column_type(self, 
                    column_name: Union[str, List[str], Tuple[str]], 
                    new_type: Union[str, List[str], Tuple[str]], 
                    verbose: bool = False) -> CS:
    """
    Changes the data type of one or more columns in the CS object's data.
    This operation preserves the original order of the columns in the table.

    Args:
        column_name: The name(s) of the column(s) whose type is to be changed.
                     Can be a single string, or a list/tuple of strings.
        new_type: The new data type(s) for the column(s) (e.g., 'INTEGER', 'VARCHAR', 'DOUBLE', 'DATE').
                  Must be a single string, or a list/tuple of strings of the same size as `column_name`.
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the column types changed.
    """

    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Normalize inputs to lists for consistent processing
    if isinstance(column_name, str):
        column_names_list = [column_name]
    elif isinstance(column_name, (list, tuple)):
        column_names_list = list(column_name)
    else:
        raise TypeError("'column_name' must be a string, list of strings, or tuple of strings.")

    if isinstance(new_type, str):
        new_types_list = [new_type]
    elif isinstance(new_type, (list, tuple)):
        new_types_list = list(new_type)
    else:
        raise TypeError("'new_type' must be a string, list of strings, or tuple of strings.")

    # Validate that the lists are of the same size
    if len(column_names_list) != len(new_types_list):
        raise ValueError("The number of column names must match the number of new types.")
    
    if not column_names_list: # Check if lists are empty after normalization
        if verbose:
            print("change_column_type: No columns specified for type change. Exiting.", file=sys.stderr)
        return self

    # Get current column names (case-insensitive for validation)
    current_columns_lower = {col.lower(): col for col in self.data.columns} # Map lower to original case

    # Validate column names exist and new types are strings
    for i in range(len(column_names_list)):
        col_name_input = column_names_list[i]
        type_input = new_types_list[i]

        if not isinstance(col_name_input, str):
            raise TypeError(f"Column name at index {i} is not a string: '{col_name_input}'.")
        if col_name_input.lower() not in current_columns_lower:
            raise ValueError(f"Column '{col_name_input}' not found in the data.")
        
        if not isinstance(type_input, str):
            raise TypeError(f"New type at index {i} is not a string: '{type_input}'.")

    try:
        # Perform each type change operation individually
        for i in range(len(column_names_list)):
            col_name_input = column_names_list[i]
            type_input = new_types_list[i]

            # Get the exact case of the column name from the current data.columns
            actual_col_name_in_table = current_columns_lower[col_name_input.lower()]

            sql_query = f"""
                ALTER TABLE \"{self._tablename}\"
                ALTER COLUMN \"{actual_col_name_in_table}\" SET DATA TYPE {type_input};
            """
            # if verbose:
            #     print(f"change_column_type: Executing SQL: {sql_query}", file=sys.stderr)
            self.cx.execute(sql_query)
            if verbose:
                print(f"change_column_type: Column '{actual_col_name_in_table}' type changed to '{type_input}'.", file=sys.stderr)

        # After all type changes, refresh self.data to reflect the new schema
        self.data = self.cx.table(self._tablename)
        if verbose:
            print(f"change_column_type: self.data refreshed. New columns/types: {self.data.columns}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in change_column_type: {e}", file=sys.stderr)
        raise e # Re-raise the exception to propagate it.

    finally:
       pass
    return self

def add_column(self, 
               column_object: Union[pd.DataFrame, pl.DataFrame, pd.Series, np.ndarray, list, tuple, duckdb.DuckDBPyRelation, pa.Array], 
               column_name: str) -> CS:
    """
    Adds a column to the CS object's data using a positional join.

    Args:
        column_object: A Pandas DataFrame/Series, Polars DataFrame, NumPy ndarray, Python list/tuple,
                       DuckDB or PyArrow Array relation containing the data for the new column.
                       If DataFrame/Series, must have only one column (or be a 1D list/tuple).
        column_name: The name of the new column (string).

    Returns:
        self: The CS object with the added column.
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Input validation and conversion
    if isinstance(column_object, (pd.DataFrame, pl.DataFrame)):
        if len(column_object.columns) != 1:
            raise ValueError("Column object must have only one column when it is a DataFrame.")
        if isinstance(column_object, pl.DataFrame):
            column_object = column_object.to_pandas()  # Convert to Pandas
        new_column_relation = self.cx.from_df(column_object)  # Create relation
    elif isinstance(column_object, pd.Series):
        new_column_relation = self.cx.from_df(pd.DataFrame({column_name: column_object}))
    elif isinstance(column_object, np.ndarray):
        if column_object.ndim != 1:
            raise ValueError("NumPy array must be 1-dimensional.")
        new_column_relation = self.cx.from_df(pd.DataFrame({column_name: column_object}))
    elif isinstance(column_object, (list, tuple)):
        new_column_relation = self.cx.from_df(pd.DataFrame({column_name: column_object}))
    elif isinstance(column_object, duckdb.DuckDBPyRelation):
        new_column_relation = column_object
    elif isinstance(column_object, pa.Array):
      new_column_table = pa.table({column_name: column_object})
      new_column_relation = self.cx.from_arrow(new_column_table)
    else:
        raise TypeError(
            "Unsupported column object type. Must be Pandas DataFrame/Series, Polars DataFrame, NumPy ndarray, list, tuple, DuckDB relation or PyArrow array."
        )

    # Ensure the column name is valid
    if not isinstance(column_name, str):
        raise TypeError("column_name must be a string.")
    if not column_name.isidentifier():
        raise ValueError("column_name must be a valid identifier (no spaces or special characters).")

    # Register the  relation as a view in DuckDB.
    temp_view_name = f"_temp_df_{id(new_column_relation)}"  # Unique view name.
    self.cx.register(temp_view_name, new_column_relation)
    # print(f"{temp_view_name=} registered!", file=sys.stderr)

    # Get the number of rows in the original data
    original_num_rows = self.cx.execute(f"SELECT count(*) FROM \"{self._original_tablename}\"").fetchone()[0]
    new_column_length = self.cx.execute(f"SELECT count(*) FROM \"{temp_view_name}\"").fetchone()[0]
    # print(f"{original_num_rows=}, {new_column_length=}", file=sys.stderr)

    if original_num_rows != new_column_length:
        self.cx.unregister(temp_view_name)
        raise ValueError(
            f"Number of rows in new column ({new_column_length}) must match the original data ({original_num_rows})."
        )
    # Construct the SQL query for the positional join.
    original_columns_sql = ", ".join(
        [f'"{col}"' for col in self.data.columns]
    )  # Quote original column names
    sql_query = f""" SELECT {original_columns_sql}, t2."{column_name}"
                     FROM "{self._original_tablename}" AS t1
                     POSITIONAL JOIN "{temp_view_name}" AS t2
                 """
    # print(f"add_column: SQL Query: {sql_query=}", file=sys.stderr)

    try:
        # Execute the query to add the column.
        new_data = self.cx.execute(sql_query).fetch_arrow_table()

        # Drop the existing table and create a new one from the result.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data;")
        self.data = self.cx.table(self._tablename)
        # return self

    finally:
        # Unregister the temporary view.
        self.cx.unregister(temp_view_name)
        # print(f"add_column: Unregistered view {temp_view_name}", file=sys.stderr)
    return self

def drop_column(self,
                column_names: Union[str, List[str], Tuple[str]]) -> CS:
    """
    Drops one or more columns from the CS object's data.

    Args:
        column_names: A string representing a single column name, or a list/tuple
                      of strings representing multiple column names to drop.

    Returns:
        a new version of the CS object
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    if isinstance(column_names, str):
        columns_to_drop = [column_names]  # Convert to a list for consistency
    elif isinstance(column_names, (list, tuple)):
        columns_to_drop = list(column_names)  # Ensure it's a list
    else:
        raise TypeError("column_names must be a string or a list/tuple of strings.")

    if not columns_to_drop:  # Check for empty list
        print("drop_column: No columns to drop.", file=sys.stderr)
        return

    # Validate column names
    valid_columns = [col.lower() for col in self.data.columns]
    columns_to_drop_lower = [col.lower() for col in columns_to_drop]
    for col in columns_to_drop_lower:
        if col not in valid_columns:
            raise ValueError(f"Column '{col}' not found in the data.")

    # Construct the SQL query
    columns_to_keep = [col for col in self.data.columns if col.lower() not in columns_to_drop_lower]
    if not columns_to_keep:
        raise ValueError("Cannot drop all columns.")

    original_table_name = self._original_tablename
    quoted_columns_to_keep = [f'"{col}"' for col in columns_to_keep]
    select_statement = ", ".join(quoted_columns_to_keep)

    sql_query = f""" SELECT {select_statement} FROM "{original_table_name}" """
    # print(f"drop_column: SQL Query: {sql_query}", file=sys.stderr)

    try:
        # Execute the query to get the data with dropped columns.
        new_data = self.cx.execute(sql_query).fetch_arrow_table()

        # Drop the existing table and create a new one with the remaining columns.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        self.data = self.cx.table(self._tablename)

    finally:
        pass
    return self

def replace_column(self, 
                   column_to_replace: Union[str, List[str], Tuple[str]], 
                   replace_column: Union[str, List[str], Tuple[str]]) -> CS:
    """
    Replaces the contents of one or more columns with the data from another column or set of columns,
    preserving the original column order.

    Args:
        column_to_replace: A string or list/tuple of strings representing the column(s) to be replaced.
        replace_column: A string or list/tuple of strings representing the column(s) to replace with.

    Returns:
        a new version of the CS object
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    if isinstance(column_to_replace, str):
        columns_to_replace_list = [column_to_replace]
    elif isinstance(column_to_replace, (list, tuple)):
        columns_to_replace_list = list(column_to_replace)
    else:
        raise TypeError("column_to_replace must be a string or a list/tuple of strings.")

    if isinstance(replace_column, str):
        replace_column_list = [replace_column]
    elif isinstance(replace_column, (list, tuple)):
        replace_column_list = list(replace_column)
    else:
        raise TypeError("replace_column must be a string or a list/tuple of strings.")

    if not columns_to_replace_list or not replace_column_list:
        raise ValueError("Both column_to_replace and replace_column must contain at least one column.")

    if len(columns_to_replace_list) != len(replace_column_list):
        raise ValueError("The number of columns to replace must match the number of replacement columns.")

    # Validate column names
    valid_columns = [col.lower() for col in self.data.columns]
    columns_to_replace_lower = [col.lower() for col in columns_to_replace_list]
    replace_column_lower = [col.lower() for col in replace_column_list]

    for col in columns_to_replace_lower + replace_column_lower:
        if col not in valid_columns:
            raise ValueError(f"Column '{col}' not found in the data.")

    # Construct the SQL query, preserving column order
    original_table_name = self._original_tablename
    original_columns = list(self.data.columns)  # Get original column order
    select_parts = []

    for original_col in original_columns:
        original_col_lower = original_col.lower()
        if original_col_lower in columns_to_replace_lower:
            # Find the index of the column to replace
            replace_index = columns_to_replace_lower.index(original_col_lower)
            replacement_col_name = replace_column_list[replace_index]
            select_parts.append(f'"{replacement_col_name}" AS "{original_col}"')
        else:
            select_parts.append(f'"{original_col}"')

    select_statement = ", ".join(select_parts)
    sql_query = f""" SELECT {select_statement} FROM "{original_table_name}" """

    try:
        # Execute the query to get the transformed data.
        new_data = self.cx.execute(sql_query).fetch_arrow_table()

        # Drop the existing table and create a new one with the replaced columns.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        self.data = self.cx.table(self._tablename)

    finally:
        pass
    return self

def rename_column(self,
                  base_column: Union[str, List[str], Tuple[str]], 
                  new_column_name: Union[str, List[str], Tuple[str]],
                  verbose: bool = False) -> CS:
    """
    Renames one or more columns in the CS object's data.

    Args:
        base_column: The current name(s) of the column(s) to be renamed.
                     Can be a single string, or a list/tuple of strings.
        new_column_name: The new name(s) for the column(s).
                         Must be a single string, or a list/tuple of strings of the same size as `base_column`.
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Normalize inputs to lists for consistent processing
    if isinstance(base_column, str):
        base_columns_list = [base_column]
    elif isinstance(base_column, (list, tuple)):
        base_columns_list = list(base_column)
    else:
        raise TypeError("'base_column' must be a string, list of strings, or tuple of strings.")

    if isinstance(new_column_name, str):
        new_column_names_list = [new_column_name]
    elif isinstance(new_column_name, (list, tuple)):
        new_column_names_list = list(new_column_name)
    else:
        raise TypeError("'new_column_name' must be a string, list of strings, or tuple of strings.")

    # Validate that the lists are of the same size
    if len(base_columns_list) != len(new_column_names_list):
        raise ValueError("The number of base columns must match the number of new column names.")
    
    if not base_columns_list: # Check if lists are empty after normalization
        if verbose:
            print("rename_column: No columns specified for renaming. Exiting.", file=sys.stderr)
        return self

    # Get current column names (case-insensitive for validation)
    current_columns_lower = [col.lower() for col in self.data.columns]
    current_columns_original_case = list(self.data.columns) # To preserve original casing if needed

    # Validate old column names exist and new column names are valid identifiers
    for i, old_name in enumerate(base_columns_list):
        if not isinstance(old_name, str):
            raise TypeError(f"Base column name at index {i} is not a string: '{old_name}'.")
        if old_name.lower() not in current_columns_lower:
            raise ValueError(f"Column to rename '{old_name}' not found in the data.")
        
        new_name = new_column_names_list[i]
        if not isinstance(new_name, str):
            raise TypeError(f"New column name at index {i} is not a string: '{new_name}'.")
        if not new_name.isidentifier():
            raise ValueError(f"New column name '{new_name}' is not a valid identifier (e.g., no spaces or special characters).")
        
        # Check for conflicts: new name should not exist as another *current* column
        # unless it's the very column being renamed (which is handled by ALTER TABLE)
        # We need to be careful here: if 'A' is renamed to 'B', and 'B' already exists, it's an error.
        if new_name.lower() in current_columns_lower and old_name.lower() != new_name.lower():
            # Ensure the new name doesn't conflict with an *unaffected* existing column
            # DuckDB's ALTER TABLE RENAME handles this
            pass # Rely on DuckDB's error for now.

    try:
        # Perform each rename operation individually
        for i in range(len(base_columns_list)):
            old_name = base_columns_list[i]
            new_name = new_column_names_list[i]

            # Find the exact case of the old column name in the current data.columns
            actual_old_name_in_table = next((col for col in current_columns_original_case if col.lower() == old_name.lower()), old_name)

            sql_query = f"""
                ALTER TABLE \"{self._tablename}\"
                RENAME COLUMN \"{actual_old_name_in_table}\" TO \"{new_name}\";
            """
            if verbose:
                print(f"2: rename_column: Executing SQL: {sql_query}", file=sys.stderr)
            self.cx.execute(sql_query)
            if verbose:
                print(f"rename_column: Column '{actual_old_name_in_table}' renamed to '{new_name}'.", file=sys.stderr)

        # After all renames, refresh self.data to reflect the new schema
        self.data = self.cx.table(self._tablename)
        if verbose:
            print(f"rename_column: self.data refreshed.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in rename_column: {e}", file=sys.stderr)
        raise e # Re-raise the exception to propagate it.

    finally:
       pass
    return self