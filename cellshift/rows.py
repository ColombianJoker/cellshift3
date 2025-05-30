import duckdb
from duckdb import DuckDBPyConnection
import pandas as pd
import polars as pl
import pyarrow as pa
import sys
import tempfile
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING
from . import CS  # Import CS from the main module to be able to return self with typing

def add_data(self,
             data: Union[str, List[str], pd.DataFrame, pl.DataFrame, pa.Table],
             verbose: bool = False) -> 'CS':
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

def remove_na_rows(self, 
                   base_column: Union[str, List[str]], 
                   verbose: bool = False) -> 'CS':
        """
        Removes rows from the .data member that have NULL values in the specified base_column(s).

        Args:
            base_column: A single column name (string) or a list of column names
                         (strings) to check for NULL values.
            verbose: If True, print debug information.

        Returns:
            a new version of the CS object
        """
        if self.data is None:
            if verbose:
                print("remove_na_rows: No data loaded in .data. Nothing to filter.", file=sys.stderr)
            return self

        # Normalize base_column to a list of strings
        columns_to_check: List[str] = []
        if isinstance(base_column, str):
            columns_to_check = [base_column]
        elif isinstance(base_column, list):
            if not all(isinstance(col, str) for col in base_column):
                raise TypeError("remove_na_rows: All items in 'base_column' list must be strings.")
            columns_to_check = base_column
        else:
            raise TypeError("remove_na_rows: 'base_column' must be a string or a list of strings.")

        if not columns_to_check:
            if verbose:
                print("remove_na_rows: No columns specified to check for NA/NULL values. No filtering applied.", file=sys.stderr)
            return self

        # Get existing column names from the data
        existing_columns = self.data.columns[:]
        if verbose:
          print(f"{self.data.columns=}\n", file=sys.stderr)
        
        # Validate if all specified columns exist
        for col in columns_to_check:
            if col not in self.data.columns:
                raise ValueError(f"remove_na_rows: Column '{col}' not found in the data. Existing columns: {', '.join(self.data.columns)}")

        try:
            # Build the WHERE clause for filtering NULLs
            # Example: "col1 IS NOT NULL AND col2 IS NOT NULL"
            conditions = [f"NOT (\"{col}\" IS NULL)" for col in columns_to_check]
            where_clause = " AND ".join(conditions)

            if verbose:
                print(f"remove_na_rows: Filtering rows where {where_clause}", file=sys.stderr)
                initial_row_count = self.data.shape[0]

            # Construct the SQL query to select non-NULL rows
            # We explicitly select all columns to maintain order and structure
            select_cols = ", ".join([f"\"{col}\"" for col in existing_columns])
            filter_query_sql = f"SELECT {select_cols} FROM \"{self._original_tablename}\" WHERE {where_clause}"
            if verbose:
                print(f"{filter_query_sql=}", file=sys.stderr)
            # Execute the query and get the resulting relation
            filtered_data = self.cx.execute(filter_query_sql).fetch_arrow_table()

            # Overwrite the original table with the filtered data
            # Drop the existing table and create a new one with reduced table.
            self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
            self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM filtered_data")
            self.data = self.cx.table(self._tablename)
            
            if verbose:
                final_row_count = self.data.shape[0]
                rows_removed = initial_row_count - final_row_count
                print(f"remove_na_rows: Filtered successfully. {rows_removed} rows removed. New row count: {final_row_count}", file=sys.stderr)

            return self

        except Exception as e:
            if verbose:
                print(f"remove_na_rows: An error occurred during filtering: {e}", file=sys.stderr)
            raise ValueError(f"Failed to remove NA/NULL rows: {e}")
        return self