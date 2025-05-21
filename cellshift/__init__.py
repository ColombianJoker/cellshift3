
from typing import Union, Optional
import pandas as pd
import duckdb
import polars as pl
from duckdb import DuckDBPyConnection
import sys

# Define global variables for the table name generator
_table_name_prefix = "table"
_table_name_separator = "_"

def table_name_generator() -> str:
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
    def __init__(self, input_data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame], db_path: str = ':memory:'):
        """
        Initializes the CS instance.

        Args:
            input_data: The data to load.
            db_path: The path to the DuckDB database file to use.
                     Defaults to ':memory:' for an in-memory database.
        """
        self.db_path = db_path  # Store the database path
        self.cx: DuckDBPyConnection = duckdb.connect(database=self.db_path, read_only=False)  # Initialize connection once
        self._tablename: str = next(_table_name_gen)  # Generate table name *before* loading
        self._original_tablename: str = self._tablename # store original table name
        self.data: Optional[duckdb.DuckDBPyRelation] = self._load_data(input_data)
        if self.data is not None:  # Add this check
            self.data = self.cx.table(self._original_tablename) # force a named relation

    def _load_data(self, data: Union[str, pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame]) -> Optional[duckdb.DuckDBPyRelation]:
        """
        Internal method to load data and return a DuckDB relation or None.
        Uses the object's connection (self.cx).
        """
        try:
            if isinstance(data, str):
                # Load from CSV and create a table (materialize the data) with the generated name
                self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM read_csv_auto('{data}')")
                # Important:  Create a named relation *immediately*
                relation = self.cx.table(self._tablename)
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
        return self._original_tablename

    def to_pandas(self) -> Optional[pd.DataFrame]:
        """Retrieves the data as a Pandas DataFrame.
      
        Returns:
            the contents of .data member in Polars DataFrame format.
        """
        if self.data:
            return self.data.df()
        else:
            return None

    def to_polars(self) -> Optional[pl.DataFrame]:
        """Retrieves the data as a Polars DataFrame.
      
        Returns:
            the contents of .data member in Polars DataFrame format.
        """
        if self.data:
            return pl.from_pandas(self.data.df())
        else:
            return None

    def to_csv(self, filename: str, **kwargs) -> bool:
        """Saves the data to a CSV file using DuckDB's SQL interface.
        
        Args:
          filename: The name of the output CSV file.
      
        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                # Create a temporary view with the table name
                self.cx.register(self._tablename, self.data)
                # Use DuckDB's SQL to write to CSV
                self.cx.execute(f"COPY (SELECT * FROM \"{self._tablename}\") TO '{filename}' (HEADER, DELIMITER ',');")
                self.cx.unregister(self._tablename)  # clean up
                return True
            except Exception as e:
                print(f"Error saving to CSV using DuckDB: {e}")
                return False
        else:
            print("No data to save to CSV.")
            return False

    def to_duckdb(self, filename: str, table_name: Optional[str] = None) -> bool:
        """
        Saves the data to a single DuckDB database file (.duckdb).

        Args:
            filename: The name of the output DuckDB database file (.duckdb).
            table_name: Optional table name. If None, uses the generated name.

        Returns:
            True on success, False on failure.
        """
        if self.data:
            try:
                output_table_name = table_name if table_name else self._original_tablename # Use original table name

                # 1. Create a new connection to the *output* database file.
                with duckdb.connect(database=filename, read_only=False) as output_cx:
                    # 2. Register the data (relation) as a view in the *output* connection.
                    output_cx.register(self._tablename, self.data)

                    # 3. Create a table in the output database with the desired name.
                    output_cx.execute(
                        f"CREATE TABLE IF NOT EXISTS \"{output_table_name}\" AS SELECT * FROM \"{self._tablename}\""
                    )

                    # 4. Unregister the view.
                    output_cx.unregister(self._tablename)

                return True  # Indicate success
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

# Additional methods in accesory files
from .columns import set_type, add_column, drop_column, replace_column
from .auxiliary import letters_for, random_code, generate_kb_code, generate_mb_code, get_file_size
from .destroy import fast_overwrite, destroy
from .noise import add_gaussian_noise_column, add_impulse_noise_column, add_salt_pepper_noise_column
from .noise import gaussian_column, impulse_column, salt_pepper_column
from .ranges import add_integer_range_column, add_age_range_column

CS.set_type = set_type
CS.add_column = add_column
CS.drop_column = drop_column
CS.replace_column = replace_column
CS.add_gaussian_noise_column = add_gaussian_noise_column
CS.add_impulse_noise_column = add_impulse_noise_column
CS.add_salt_pepper_noise_column = add_salt_pepper_noise_column
CS.gaussian_column = gaussian_column
CS.impulse_column = impulse_column
CS.salt_pepper_column = salt_pepper_column
CS.add_integer_range_column = add_integer_range_column
CS.add_age_range_column = add_age_range_column
