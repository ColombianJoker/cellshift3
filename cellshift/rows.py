import duckdb
from duckdb import DuckDBPyConnection
from os import urandom
import pandas as pd
import polars as pl
import pyarrow as pa
import sys
import tempfile
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING
from . import CS  # Import CS from the main module to be able to return self with typing

def add_data(self,
             data: Union[str, List[str], pd.DataFrame, pl.DataFrame, pa.Table],
             verbose: bool = False) -> CS:
    """
    Adds new data to the existing .data member of the class.
    The new data is added using INSERT into current data.
    Column names and types must be compatible with existing data.

    Args:
        data: the data to add. Receives:
                    + a string (filename of a file to load)
                    + a list of strings (each a filename of a file to load)
                    + a Pandas DataFrame
                    + a Polars DataFrame
                    + a PyArrow Table
        verbose: If True, show debug messages

    Returns:
        a new version of the CS object
    """
    if verbose:
        print(f"add: Start, input_data type: {type(data)}", file=sys.stderr)
    if isinstance(data, str):
        sql_insert = sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{data}';"
        if verbose:
            print(f"add('{data}') (str)", file=sys.stderr)
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
        # self.data = self.cx.table(self._tablename)
    elif isinstance(data, list):
        for item in data:
            sql_insert = sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{item}';"
            if verbose:
                print(f"add('{item}') (str)", file=sys.stderr)
                print(sql_insert, file=sys.stderr)
            self.cx.execute(sql_insert)
        # self.data = self.cx.table(self._tablename)
    elif isinstance(data, pd.DataFrame):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, pl.DataFrame):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, pa.Table):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, duckdb.DuckDBPyRelation):
        print(f"Unsupported input type: {type(data)}", file=sys.stderr)
        print(f"Please re-try using data.arrow()", file=sys.stderr)
        return self
    else:
        print(f"Unsupported input type: {type(data)}", file=sys.stderr)
        return self
    self.data = self.cx.table(self._tablename)
    return self

def remove_rows(self,
                 base_column: Union[str, List[str]],
                 condition: str = '? IS NULL',
                 meta: str = '?',
                 all_and: bool = True,
                 verbose: bool = False) -> CS:
    """
    Removes rows from the .data member based on a custom SQL condition applied to specified columns,
            defaulting to removing rows where the column is NULL.

    Args:
        base_column: A single column name (string) or a list of column names
                     (strings) to apply the condition to.
        condition: A string representing the SQL condition template for each column.
                   It must contain the 'meta' string which will be replaced by the column name.
                   Defaults to '? IS NULL' to remove rows where the column is NULL.
        meta: The meta-character string within the 'condition' that will be replaced
              by the actual column name. Defaults to '?'.
        all_and: if True, applies the condition using AND to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) AND CONDITION(COLUMN2) ...
                 if False, applies the condition using OR to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) OR CONDITION(COLUMN2) ...
        verbose: If True, print debug information.

    Returns:
        self: The CS object with rows filtered based on the specified condition.
    """
    if self.data is None:
        if verbose:
            print("remove_rows: No data loaded in .data. Nothing to filter.", file=sys.stderr)
        return self

    # Normalize base_column to a list of strings
    columns_to_check: List[str] = []
    if isinstance(base_column, str):
        columns_to_check = [base_column]
    elif isinstance(base_column, list):
        if not all(isinstance(col, str) for col in base_column):
            raise TypeError("remove_rows: All items in 'base_column' list must be strings.")
        columns_to_check = base_column
    else:
        raise TypeError("remove_rows: 'base_column' must be a string or a list of strings.")

    if not columns_to_check:
        if verbose:
            print("remove_rows: No columns specified to check. No filtering applied.", file=sys.stderr)
        return self

    if not isinstance(condition, str) or not condition:
        raise TypeError("remove_rows: 'condition' must be a non-empty string.")
    if not isinstance(meta, str) or not meta:
        raise TypeError("remove_rows: 'meta' must be a non-empty string.")
    if meta not in condition:
        raise ValueError(f"remove_rows: 'meta' string '{meta}' not found in 'condition' string '{condition}'.")

    # Get existing column names from the data
    # Note: self.data.columns is already a list of strings (column names)
    existing_columns = self.data.columns
    if verbose:
        print(f"{self.data.columns=}\n", file=sys.stderr)

    # Validate if all specified columns exist
    for col in columns_to_check:
        if col not in existing_columns:
            raise ValueError(f"remove_rows: Column '{col}' not found in the data. Existing columns: {', '.join(existing_columns)}")

    try:
        # Build the WHERE clause using the custom condition and meta-character replacement
        # Example: for condition='? < 0' and meta='?',
        #          it becomes '(NOT ("col1" < 0)) AND (NOT ("col2" < 0))'
        conditions_for_delete = []
        for col in columns_to_check:
            # Replace the meta-character with the quoted column name
            sql_condition_for_col = condition.replace(meta, f"\"{col}\"")
            conditions_for_delete.append(f"({sql_condition_for_col})") # Wrap each condition in parentheses for safety
        if all_and:
          where_delete = " AND ".join(conditions_for_delete)
        else: # if not and then OR
          where_delete = " OR ".join(conditions_for_delete)

        if verbose:
            print(f"remove_rows: Removing rows where {where_delete}", file=sys.stderr)
            initial_row_count = self.data.shape[0]

        # Construct the SQL query to select filtered rows
        delete_sql = f'DELETE FROM "{self._tablename}" WHERE {where_delete}'

        if verbose:
            print(f"{delete_sql=}", file=sys.stderr)

        # Execute the query and fetch the result as an Arrow Table
        # new_data = self.cx.execute(filter_query_sql).fetch_arrow_table()
        new_data = self.cx.sql(delete_sql)
        self.data = self.cx.table(self._tablename)

        if verbose:
            print(f"{type(self.data)=}", file=sys.stderr)
            final_row_count = self.data.shape[0]
            rows_removed = initial_row_count - final_row_count
            print(f"remove_rows: Filtered successfully. {rows_removed} rows removed. New row count: {final_row_count}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"remove_rows: An error occurred during filtering: {e}", file=sys.stderr)
        # Re-raise the exception after printing verbose info
        raise ValueError(f"Failed to remove rows based on condition: {e}")
    return self

def filter_rows(self,
                 base_column: Union[str, List[str]],
                 condition: str = '? IS NOT NULL',
                 meta: str = '?',
                 all_and: bool = True,
                 verbose: bool = False) -> CS:
    """
    Filters rows from the .data member based on a custom SQL condition applied to specified columns,
        leaving only rows where the condition is met, defaulting to removing rows where the column is NULL.

    Args:
        base_column: A single column name (string) or a list of column names
                     (strings) to apply the condition to.
        condition: A string representing the SQL condition template for each column.
                   It must contain the 'meta' string which will be replaced by the column name.
                   Defaults to '? IS NULL' to remove rows where the column is NULL.
        meta: The meta-character string within the 'condition' that will be replaced
              by the actual column name. Defaults to '?'.
        all_and: if True, applies the condition using AND to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) AND CONDITION(COLUMN2) ...
                 if False, applies the condition using OR to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) OR CONDITION(COLUMN2) ...
        verbose: If True, print debug information.

    Returns:
        self: The CS object with rows filtered based on the specified condition.
    """
    if self.data is None:
        if verbose:
            print("remove_rows: No data loaded in .data. Nothing to filter.", file=sys.stderr)
        return self

    # Normalize base_column to a list of strings
    columns_to_check: List[str] = []
    if isinstance(base_column, str):
        columns_to_check = [base_column]
    elif isinstance(base_column, list):
        if not all(isinstance(col, str) for col in base_column):
            raise TypeError("remove_rows: All items in 'base_column' list must be strings.")
        columns_to_check = base_column
    else:
        raise TypeError("remove_rows: 'base_column' must be a string or a list of strings.")

    if not columns_to_check:
        if verbose:
            print("remove_rows: No columns specified to check. No filtering applied.", file=sys.stderr)
        return self

    if not isinstance(condition, str) or not condition:
        raise TypeError("remove_rows: 'condition' must be a non-empty string.")
    if not isinstance(meta, str) or not meta:
        raise TypeError("remove_rows: 'meta' must be a non-empty string.")
    if meta not in condition:
        raise ValueError(f"remove_rows: 'meta' string '{meta}' not found in 'condition' string '{condition}'.")

    # Get existing column names from the data
    # Note: self.data.columns is already a list of strings (column names)
    existing_columns = self.data.columns
    if verbose:
        print(f"{self.data.columns=}\n", file=sys.stderr)

    # Validate if all specified columns exist
    for col in columns_to_check:
        if col not in existing_columns:
            raise ValueError(f"remove_rows: Column '{col}' not found in the data. Existing columns: {', '.join(existing_columns)}")

    try:
        # Build the WHERE clause using the custom condition and meta-character replacement
        # Example: for condition='? < 0' and meta='?',
        #          it becomes '(NOT ("col1" < 0)) AND (NOT ("col2" < 0))'
        conditions_for_sql = []
        for col in columns_to_check:
            # Replace the meta-character with the quoted column name
            sql_condition_for_col = condition.replace(meta, f"\"{col}\"")
            conditions_for_sql.append(f"({sql_condition_for_col})") # Wrap each condition in parentheses for safety

        if all_and:
          where_clause = " AND ".join(conditions_for_sql)
        else: # if not AND then OR
          where_clause = " OR ".join(conditions_for_sql)

        if verbose:
            print(f"remove_rows: Filtering rows where {where_clause}", file=sys.stderr)
            initial_row_count = self.data.shape[0]

        # Construct the SQL query to select filtered rows
        select_cols = ", ".join([f"\"{col}\"" for col in existing_columns])
        # filter_query_sql = f"SELECT {select_cols} FROM \"{self._original_tablename}\" WHERE {where_clause}"
        filter_query_sql = f'SELECT * FROM "{self._tablename}" WHERE {where_clause}'
        
        if verbose:
            print(f"{filter_query_sql=}", file=sys.stderr)

        # Execute the query and fetch the result as an Arrow Table
        # new_data = self.cx.execute(filter_query_sql).fetch_arrow_table()
        new_data = self.cx.sql(filter_query_sql)
        # Drop the existing table and create a new one with the replaced columns.
        # self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        # self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        if (new_data) and isinstance(new_data, duckdb.duckdb.DuckDBPyRelation):
          self.cx.execute(f"CREATE OR REPLACE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
          self.data = self.cx.table(self._tablename)

        if verbose:
            final_row_count = self.data.shape[0]
            rows_removed = initial_row_count - final_row_count
            print(f"remove_rows: Filtered successfully. {rows_removed} rows removed. New row count: {final_row_count}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"remove_rows: An error occurred during filtering: {e}", file=sys.stderr)
        # Re-raise the exception after printing verbose info
        raise ValueError(f"Failed to remove rows based on condition: {e}")
    return self
