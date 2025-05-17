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
        """Saves the data to a CSV file."""
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
        Saves the data to a DuckDB database file.  This method now
        creates a *new* connection to the specified file, registers
        the data, and closes the connection.  It does *not* use
        the object's connection (self.cx) for this operation.
        """
        if self.data:
            try:
                con = duckdb.connect(filename)  # Connect to the output file
                con.register(table_name if table_name else self._tablename, self.data)
                con.close()  # Close the connection after registering
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

# Example Usage
import pandas as pd
import polars as pl
import duckdb

# Create sample data
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

# Create CS instance
cs1 = CS(df_pl) # Initialize with Polars DataFrame

# Get data as Pandas and Polars DataFrames
df_pandas1 = cs1.to_pandas()
df_polars1 = cs1.to_polars()

if df_pandas1 is not None:
    print("Pandas DataFrame 1:")
    print(df_pandas1)
if df_polars1 is not None:
    print("Polars DataFrame 1:")
    print(df_polars1)
    
# Create another CS instance, loading from the first CS object's data
cs2 = CS(cs1.data)

df_pandas2 = cs2.to_pandas()
df_polars2 = cs2.to_polars()

if df_pandas2 is not None:
    print("Pandas DataFrame 2:")
    print(df_pandas2)
if df_polars2 is not None:
    print("Polars DataFrame 2:")
    print(df_polars2)

# Save data to CSV and DuckDB
csv_success = cs1.to_csv("output.csv", index=False)
duckdb_success = cs1.to_duckdb("output.duckdb")

if csv_success:
    print("Data saved to output.csv")
if duckdb_success:
    print("Data saved to output.duckdb")

# Demonstrate setting prefix/separator
CS.set_table_name_prefix("my_table")
CS.set_table_name_separator("_")

cs3 = CS(df_pd)
print(f"Table name 3: {cs3.get_tablename()}")

cs1.close_connection()  # Explicitly close the connection when done with cs1
del cs1 # delete the object
