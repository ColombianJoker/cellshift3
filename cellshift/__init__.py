#!/usr/bin/env python3
from typing import Union, Optional
import pandas as pd
import duckdb
import polars as pl

# Define global variables for the table name generator
_table_name_prefix = "table"
_table_name_separator = "_"

def table_name_generator():
    """Generator function to create unique table names."""
    seq = 0
    while True:
        yield f"{_table_name_prefix}{_table_name_separator}{seq}"
        seq += 1

_table_name_gen = table_name_generator()

class CS:
    """
    Class that loads data, generates a unique table name, and provides
    methods to get the data as Pandas/Polars DataFrames, and save to
    CSV/DuckDB.
    """
    def __init__(self, input_data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame]):
        """
        Initializes the CS instance.

        Args:
            input_data: The data to load.
        """
        self.data: Optional[duckdb.DuckDBPyRelation] = self._load_data(input_data)
        self._tablename: str = next(_table_name_gen)

    def _load_data(self, data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame]) -> Optional[duckdb.DuckDBPyRelation]:
        """
        Internal method to load data and return a DuckDB relation or None.
        """
        try:
            if isinstance(data, str):
                con = duckdb.connect(database=':memory:', read_only=False)
                relation = con.read_csv(data)
                return relation
            elif isinstance(data, pd.DataFrame):
                con = duckdb.connect(database=':memory:', read_only=False)
                relation = con.from_df(data)
                return relation
            elif isinstance(data, duckdb.DuckDBPyRelation):
                return data
            elif isinstance(data, pl.DataFrame):
                con = duckdb.connect(database=':memory:', read_only=False)
                relation = con.from_df(data.to_pandas())
                return relation
            else:
                print(f"Unsupported input type: {type(data)}")
                return None
        except Exception as e:
            print(f"An error occurred during loading: {e}")
            return None

    def get_tablename(self) -> str:
        """Returns the generated table name."""
        return self._tablename

    def to_pandas(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the data as a Pandas DataFrame.

        Returns:
            Pandas DataFrame or None if no data is loaded.
        """
        if self.data:
            return self.data.df()
        else:
            return None

    def to_polars(self) -> Optional[pl.DataFrame]:
        """
        Retrieves the data as a Polars DataFrame.

        Returns:
            Polars DataFrame or None if no data is loaded.
        """
        if self.data:
            return pl.from_pandas(self.data.df())
        else:
            return None

    def to_csv(self, filename: str, **kwargs) -> bool:
        """
        Saves the data to a CSV file.

        Args:
            filename: The name of the CSV file.
            **kwargs:  Additional arguments to pass to Pandas' `to_csv()`.

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                df_pd = self.data.df()
                df_pd.to_csv(filename, **kwargs)
                return True
            except Exception as e:
                print(f"Error saving to CSV: {e}")
                return False
        else:
            print("No data to save to CSV.")
            return False

    def to_duckdb(self, filename: str, table_name: Optional[str] = None) -> bool:
        """
        Saves the data to a DuckDB database file.

        Args:
            filename: The name of the DuckDB database file (.duckdb).
            table_name: Optional table name. If None, uses the generated name.

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                con = duckdb.connect(filename)
                con.register(table_name if table_name else self._tablename, self.data)
                con.close()
                return True
            except Exception as e:
                print(f"Error saving to DuckDB: {e}")
                return False
        else:
            print("No data to save to DuckDB.")
            return False

    @classmethod
    def set_table_name_prefix(cls, prefix: str) -> None:
        """Sets the global table name prefix."""
        global _table_name_prefix
        _table_name_prefix = prefix

    @classmethod
    def set_table_name_separator(cls, separator: str) -> None:
        """Sets the global table name separator."""
        global _table_name_separator
        _table_name_separator = separator
        global _table_name_gen
        _table_name_gen = table_name_generator()
