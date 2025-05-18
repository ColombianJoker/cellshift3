#!/usr/bin/env python3
from typing import Union, Optional
import pandas as pd
import duckdb
import polars as pl
from duckdb import DuckDBPyConnection

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
    CSV/DuckDB.  A single DuckDB connection is maintained for the
    lifetime of the object.
    """
    def __init__(self, input_data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame]):
        """
        Initializes the CS instance.

        Args:
            input_data: The data to load.
        """
        self.cx: DuckDBPyConnection = duckdb.connect(database=':memory:', read_only=False)  # Initialize connection once
        self.data: Optional[duckdb.DuckDBPyRelation] = self._load_data(input_data)
        self._tablename: str = next(_table_name_gen)

    def _load_data(self, data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame]) -> Optional[duckdb.DuckDBPyRelation]:
        """
        Internal method to load data and return a DuckDB relation or None.
        Uses the object's connection (self.cx).
        """
        try:
            if isinstance(data, str):
                relation = self.cx.read_csv(data)
                return relation
            elif isinstance(data, pd.DataFrame):
                relation = self.cx.from_df(data)
                return relation
            elif isinstance(data, duckdb.DuckDBPyRelation):
                return data
            elif isinstance(data, pl.DataFrame):
                relation = self.cx.from_df(data.to_pandas())
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
        """Retrieves the data as a Pandas DataFrame."""
        if self.data:
            return self.data.df()
        else:
            return None

    def to_polars(self) -> Optional[pl.DataFrame]:
        """Retrieves the data as a Polars DataFrame."""
        if self.data:
            return pl.from_pandas(self.data.df())
        else:
            return None

    def to_csv(self, filename: str, **kwargs) -> bool:
        """
        Saves the data to a CSV file using DuckDB's SQL interface.

        Args:
            filename: The name of the CSV file.
            **kwargs:  Currently not used, but kept for potential future extensions.

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                # Create a temporary view with the table name
                self.cx.register(self._tablename, self.data)
                # Use DuckDB's SQL to write to CSV
                self.cx.execute(f"COPY (SELECT * FROM \"{self._tablename}\") TO '{filename}' (HEADER, DELIMITER ',');")
                self.cx.unregister(self._tablename) # clean up
                return True
            except Exception as e:
                print(f"Error saving to CSV using DuckDB: {e}")
                return False
        else:
            print("No data to save to CSV.")
            return False

    def to_parquet(self, filename: str, **kwargs) -> bool:
        """
        Saves the data to a Parquet file using DuckDB's SQL interface.

        Args:
            filename: The name of the Parquet file.
            **kwargs:  Additional arguments to pass to DuckDB's COPY statement.
                       For example, 'COMPRESSION'='SNAPPY'

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                # Create a temporary view
                self.cx.register(self._tablename, self.data)
                
                # Construct the COPY statement.  Start with the basics.
                sql = f"COPY (SELECT * FROM \"{self._tablename}\") TO '{filename}' (FORMAT 'PARQUET'"
                
                # Add any additional keyword arguments to the SQL command
                for key, value in kwargs.items():
                    sql += f", {key.upper()}='{value}'" # convert key to uppercase
                sql += ");" # close the sql statement
                
                # Execute the SQL command
                self.cx.execute(sql)
                self.cx.unregister(self._tablename)
                return True
            except Exception as e:
                print(f"Error saving to Parquet using DuckDB: {e}")
                return False
        else:
            print("No data to save to Parquet.")
            return False
    def to_json(self, filename: str, **kwargs) -> bool:
        """
        Saves the data to a JSON file using DuckDB's SQL interface.

        Args:
            filename: The name of the JSON file.
            **kwargs:  Additional arguments to pass to DuckDB's COPY statement
                       (e.g., 'ARRAY' = TRUE, 'LINE DELIMITED' = TRUE).

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                # Register the relation as a temporary view
                self.cx.register(self._tablename, self.data)

                # Construct the COPY statement
                sql = f"COPY (SELECT * FROM \"{self._tablename}\") TO '{filename}' (FORMAT 'JSON'"

                # Add any additional keyword arguments to the SQL command
                for key, value in kwargs.items():
                    sql += f", {key.upper()} = {value}"  #  No quotes for boolean
                sql += ");"

                # Execute the SQL command
                self.cx.execute(sql)
                self.cx.unregister(self._tablename)
                return True
            except Exception as e:
                print(f"Error saving to JSON using DuckDB: {e}")
                return False
        else:
            print("No data to save to JSON.")
            return False

    def to_duckdb(self, filename: str, table_name: Optional[str] = None, debug: bool = False ) -> bool:
        """
        Saves the in-memory database to a DuckDB database file using ATTACH and COPY FROM DATABASE MEMORY.

        Args:
            filename: The name of the output DuckDB database file (.duckdb).
            table_name: Optional table name.  If None, uses the generated name.
                         Note: This table name is used *within* the attached database.

        Returns:
            True on success, False on failure.
        """
        if debug:
            import sys
        if self.data:
            try:
                output_table_name = table_name if table_name else self._tablename

                # 1. Attach the output database.
                self.cx.execute(f"ATTACH '{filename}' AS output_db;")
                if debug:
                    print(f"ATTACH '{filename}' AS output_db;", file=sys.stderr)

                # 2. Register the data (relation) as a view in the *main* database.
                self.cx.register(self._tablename, self.data)
                if debug:
                    print(f"self.cx.register('{self._tablename}', self.data)", file=sys.stderr)

                # 3.  Use COPY FROM to copy the *entire* in-memory database.
                self.cx.execute(f"COPY FROM DATABASe memory TO output_db;")
                if debug:
                    print(f"COPY FROM DATABASe memory TO output_db;", file=sys.stderr)
                    
                # 4. Unregister the view (cleanup).
                self.cx.unregister(self._tablename)
                if debug:
                    print(f"self.cx.unregister({self._tablename})", file=sys.stderr)

                # 5. Detach the output database.
                self.cx.execute(f"DETACH output_db;")
                if debug:
                    print(f"DETACH output_db;", file=sys.stderr)
                return True
            except Exception as e:
                print(f"Error saving to DuckDB: {e}")
                return False
        else:
            print("No data to save to DuckDB.")
            return False

    def close_connection(self) -> None:
        """
        Closes the DuckDB connection associated with this instance.
        It's good practice to close connections when you're done with them.
        """
        if hasattr(self, 'cx') and self.cx is not None:
            self.cx.close()
            self.cx = None  # Prevent further use of the closed connection

    def __del__(self):
        """
        Destructor to ensure the DuckDB connection is closed when
        the CS object is garbage collected.
        """
        self.close_connection()

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
