#!/usr/bin/env python3 

import duckdb as d
import numpy as np
import pandas as pd
import polars as pl
from typing import Generator, Union, Optional
import sys

def set_type(self, column_name: str, new_type: str):
  """Set the type of a relation (table) column by casting and using the same name
  
  Args:
      column_name: The name of the column to change (string)
      new_type: The new (SQL) type. As INT, VARCHAR, DATE, TIMESTAMP, and such
  """
  new_rel = self.data.project(
    f"* EXCLUDE {column_name}, CAST({column_name} AS {new_type}) AS {column_name};"
  )
  self.data = new_rel

def add_column(self, column_object: Union[pd.DataFrame, pl.DataFrame, pd.Series, np.ndarray, list, tuple], column_name: str) -> None:
    """
    Adds a column to the CS object's data using a positional join.

    Args:
        column_object: A Pandas DataFrame/Series, Polars DataFrame, NumPy ndarray,
                       Python list, or tuple containing the data for the new column.
                       Must have only one column (or be a 1D list/tuple).
        column_name: The name of the new column (string).
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Input validation and conversion
    if isinstance(column_object, (pd.DataFrame, pl.DataFrame)):
        if len(column_object.columns) != 1:
            raise ValueError("Column object must have only one column when it is a DataFrame.")
        if isinstance(column_object, pl.DataFrame):
            column_object = column_object.to_pandas()  # Convert to Pandas
        new_column_pd = column_object.iloc[:, 0]  # Extract the Series
    elif isinstance(column_object, pd.Series):
        new_column_pd = column_object
    elif isinstance(column_object, np.ndarray):
        if column_object.ndim != 1:
            raise ValueError("NumPy array must be 1-dimensional.")
        new_column_pd = pd.Series(column_object)
    elif isinstance(column_object, (list, tuple)):
        new_column_pd = pd.Series(column_object)
    else:
        raise TypeError(
            "Unsupported column object type. Must be Pandas DataFrame/Series, Polars DataFrame, NumPy ndarray, list, or tuple."
        )

    # Ensure the column name is valid
    if not isinstance(column_name, str):
        raise TypeError("column_name must be a string.")
    if not column_name.isidentifier():
        raise ValueError("column_name must be a valid identifier (no spaces or special characters).")

    # Create a temporary DataFrame for the new column.
    temp_df = pd.DataFrame({column_name: new_column_pd})

    # Register the temporary DataFrame as a view in DuckDB.
    temp_view_name = f"_temp_df_{id(temp_df)}"  # Unique view name.
    self.cx.register(temp_view_name, temp_df)

    # Get the number of rows in the original data
    original_num_rows = self.cx.execute(f"SELECT count(*) FROM \"{self._original_tablename}\"").fetchone()[0]
    new_column_length = len(temp_df)

    if original_num_rows != new_column_length:
        self.cx.unregister(temp_view_name)
        raise ValueError(
            f"Number of rows in new column ({new_column_length}) must match the original data ({original_num_rows})."
        )
    # Construct the SQL query for the positional join.
    original_columns_sql = ", ".join(
        [f'"{col}"' for col in self.data.columns]
    )  # Quote original column names
    sql_query = f"""
        SELECT
            {original_columns_sql},
            t2."{column_name}"
        FROM
            "{self._original_tablename}" AS t1
        POSITIONAL JOIN
            "{temp_view_name}" AS t2
    """
    
    try:
        # Execute the query to add the column.
        new_data = self.cx.execute(sql_query).fetch_arrow_table() # Fetch as Arrow table

        # Drop the existing table and create a new one from the result.
        self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        # Create a table from the arrow table.
        self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        self.data = self.cx.table(self._tablename)

    finally:
        # Unregister the temporary view.
        self.cx.unregister(temp_view_name)

