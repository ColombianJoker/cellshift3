#!/usr/bin/env python3 

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from typing import Generator, Union, Optional
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def set_type(self, column_name: str, new_type: str) -> CS:
  """Set the type of a relation (table) column by casting and using the same name
  
  Args:
      column_name: The name of the column to change (string)
      new_type: The new (SQL) type. As INT, VARCHAR, DATE, TIMESTAMP, and such
  """
  new_rel = self.data.project(
    f"* EXCLUDE {column_name}, CAST({column_name} AS {new_type}) AS {column_name};"
  )
  self.data = new_rel
  return self

def add_column(self, column_object: Union[pd.DataFrame, pl.DataFrame, pd.Series, np.ndarray, list, tuple, duckdb.DuckDBPyRelation, pa.Array], column_name: str) -> CS:
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
    # print("1: add_column: Start", file=sys.stderr)
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
    # print(f"2: {temp_view_name=} registered!", file=sys.stderr)

    # Get the number of rows in the original data
    original_num_rows = self.cx.execute(f"SELECT count(*) FROM \"{self._original_tablename}\"").fetchone()[0]
    new_column_length = self.cx.execute(f"SELECT count(*) FROM \"{temp_view_name}\"").fetchone()[0]
    # print(f"3: {original_num_rows=}, {new_column_length=}", file=sys.stderr)

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
    # print(f"4: add_column: SQL Query: {sql_query=}", file=sys.stderr)

    try:
        # Execute the query to add the column.
        # self.cx.execute(sql_query)
        new_data = self.cx.execute(sql_query).fetch_arrow_table()

        # Drop the existing table and create a new one from the result.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data;")
        self.data = self.cx.table(self._tablename)
        # return self

    finally:
        # Unregister the temporary view.
        self.cx.unregister(temp_view_name)
        # print(f"5: add_column: Unregistered view {temp_view_name}", file=sys.stderr)
    # print("6: add_column: End")
    return self

def drop_column(self, column_names: Union[str, list, tuple]) -> CS:
    """
    Drops one or more columns from the CS object's data.

    Args:
        column_names: A string representing a single column name, or a list/tuple
                      of strings representing multiple column names to drop.

    Returns:
        a new version of the CS object
    """
    # print("drop_column: Start", file=sys.stderr)
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
        # print("drop_column: End", file=sys.stderr)
    return self

def replace_column(self, column_to_replace: Union[str, list, tuple], replace_column: Union[str, list, tuple]) -> CS:
    """
    Replaces the contents of one or more columns with the data from another column or set of columns,
    preserving the original column order.

    Args:
        column_to_replace: A string or list/tuple of strings representing the column(s) to be replaced.
        replace_column: A string or list/tuple of strings representing the column(s) to replace with.

    Returns:
        a new version of the CS object
    """
    # print("replace_column: Start", file=sys.stderr)
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
    # print(f"replace_column: SQL Query: {sql_query}", file=sys.stderr)

    try:
        # Execute the query to get the transformed data.
        new_data = self.cx.execute(sql_query).fetch_arrow_table()

        # Drop the existing table and create a new one with the replaced columns.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        self.data = self.cx.table(self._tablename)

    finally:
        pass
        # print("replace_column: End", file=sys.stderr)
    return self