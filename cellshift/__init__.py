
import duckdb
from duckdb import DuckDBPyConnection
import pandas as pd
import polars as pl
import pyarrow as pa
import sys
import tempfile
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING

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
    CSV/DuckDB.  Try to maintain only one DuckDB connection for everything.
    """
    def __init__(self, 
                 input_data: Union[str, List[str], pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame, pa.Table], 
                 db_path: str = ':memory:'):
        """
        Initializes the CS instance.

        Args:
            input_data: The data to load.
            db_path: The path to the DuckDB database file to use.
                     Defaults to ':memory:' for an in-memory database.
        """
        self.db_path = db_path                             # Store the database path
        self.temp_dir = tempfile.mkdtemp()
        self.cx: DuckDBPyConnection = duckdb.connect(database=self.db_path,
                                      read_only=False,
                                      config={"temp_directory": self.temp_dir, },
        ) # Initialize
        self._tablename: str = next(_table_name_gen)       # Generate table name *before* loading
        self._original_tablename: str = self._tablename    # store original table name
        
        # Call _load_data, which will now ensure the data is materialized into a named 
        # table and return the relation for that table.
        self.data: Optional[duckdb.DuckDBPyRelation] = self._load_data(input_data)
        self._faker_locale: str = "es_CO"          # Initialize with default 'Colombia'
        self._equiv: Dict[str, Any] = {}           # For equivalence tables 'unused'

    def _load_data(self, 
                   data: Union[str, List[str], pd.DataFrame, duckdb.DuckDBPyRelation, pl.DataFrame],
                   verbose: bool = False) -> Optional[duckdb.DuckDBPyRelation]:
        """
        Internal method to load data and return a DuckDB relation or None.
        Ensures the data is materialized into a named table within the DuckDB connection.
        Uses the object's connection (self.cx).
        """
        try:
            if isinstance(data, str):
                if verbose:
                    print(f"_load_data(self, '{data}' (str))", file=sys.stderr)
                # This path already creates a named table from CSV
                self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM '{data}'")
            elif isinstance(data, list):
                first_item = data.pop(0) # get first item and remove it to create
                sql_create = f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM '{first_item}';"
                if verbose:
                    print(f"_load_data(self, '{first_item}' (str))", file=sys.stderr)
                    print(f"{sql_create=}", file=sys.stderr)
                self.cx.execute(sql_create)
                for item in data:
                    sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{item}';"
                    if verbose:
                        print(f"_load_data(self, '{item}' (str))", file=sys.stderr)
                        print(f"{sql_insert=}", file=sys.stderr)
                    self.cx.execute(sql_insert)
            elif isinstance(data, pd.DataFrame):
                sql_create = f"CREATE TABLE {self._tablename} AS SELECT * FROM data;"
                if verbose:
                    print(f"{type(data)=}\n", file=sys.stderr)
                    print(f"{type(self.cx)=}\n", file=sys.stderr)
                    print(f"{sql_create=}\n", file=sys.stderr)
                self.cx.sql(sql_create)
            elif isinstance(data, pl.DataFrame):
                sql_create = f"CREATE TABLE {self._tablename} AS SELECT * FROM data;"
                if verbose:
                    print(f"{type(data)=}\n", file=sys.stderr)
                    print(f"{type(self.cx)=}\n", file=sys.stderr)
                    print(f"{sql_create=}\n", file=sys.stderr)
                self.cx.sql(sql_create)
            elif isinstance(data, pa.Table):
                sql_create = f"CREATE TABLE {self._tablename} AS SELECT * FROM data;"
                if verbose:
                    print(f"{type(data)=}\n", file=sys.stderr)
                    print(f"{type(self.cx)=}\n", file=sys.stderr)
                    print(f"{sql_create=}\n", file=sys.stderr)
                self.cx.sql(sql_create)
            elif isinstance(data, duckdb.DuckDBPyRelation):
                if verbose:
                    print(f"_load_data(self, data (DuckDBPyRelation))", file=sys.stderr)
                # If it's already a relation, materialize it into a named table
                data.create_table(self._tablename)
            else:
                print(f"Unsupported input type: {type(data)}", file=sys.stderr)
                return None

            # After any of the above, the table should exist, so we can reliably get its relation by name
            relation = self.cx.table(self._tablename)
            return relation
        except Exception as e:
            print(f"An error occurred during loading: {e}", file=sys.stderr)
            return None

    def get_tablename(self) -> str:
        """Returns the generated table name."""
        return self._original_tablename

    def to_pandas(self) -> Optional[pd.DataFrame]:
        """Retrieves the data as a Pandas DataFrame.
        
        Returns:
            the contents of .data member in Pandas DataFrame format.
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
                print(f"Error saving to CSV using DuckDB: {e}", file=sys.stderr)
                return False
        else:
            print("No data to save to CSV.", file=sys.stderr)
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

                # Create a new connection to the *output* database file.
                with duckdb.connect(database=filename, read_only=False) as output_cx:
                    # Register the data (relation) as a view in the *output* connection.
                    output_cx.register(self._tablename, self.data)

                    # Create a table in the output database with the desired name.
                    output_cx.execute(
                        f"CREATE TABLE IF NOT EXISTS \"{output_table_name}\" AS SELECT * FROM \"{self._tablename}\""
                    )

                    # Unregister the view.
                    output_cx.unregister(self._tablename)

                return True  # Indicate success
            except Exception as e:
                print(f"Error saving to DuckDB: {e}", file=sys.stderr)
                return False
        else:
            print("No data to save to DuckDB.", file=sys.stderr)
            return False

    def add(self,
            input_data: Union[str, List[str], pd.DataFrame, pl.DataFrame, pa.Table],
            verbose: bool = False) -> 'CS':
      """
      Adds new data to the existing .data member of the class.
      The new data is added using INSERT into current data.
      Column names and types must be compatible with existing data.
    
      Args:
          input_data: the data to add. Receives:
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
          print(f"add: Start, input_data type: {type(input_data)}", file=sys.stderr)
      if isinstance(input_data, str):
          sql_insert = sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{input_data}';"
          if verbose:
              print(f"add('{input_data}') (str)", file=sys.stderr)
              print(sql_insert)
          self

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

    @property
    def faker_locale(self) -> str:
        """
        Getter for the faker_locale attribute.
        Returns:
            The current Faker locale string.
        """
        return self._faker_locale

    @faker_locale.setter
    def faker_locale(self, locale: str) -> None:
        """
        Setter for the faker_locale attribute.
        Args:
            locale: The new Faker locale string (e.g., "en_US", "es_MX").
        Raises:
            ValueError: If the provided locale is not a string or is empty.
        """
        if not isinstance(locale, str) or not locale:
            raise ValueError("Faker locale must be a non-empty string.")
        self._faker_locale = locale

    @property
    def equiv(self) -> Dict[str, Any]:
        """
        Getter for the equiv attribute.
        Returns:
            The current equivalence dictionary.
        """
        return self._equiv

    @equiv.setter
    def equiv(self, new_equiv: Dict[str, Any]) -> None:
        """
        Setter for the equiv attribute.
        Args:
            new_equiv: The new equivalence dictionary.
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(new_equiv, dict):
            raise TypeError("Equivalence mapping must be a dictionary.")
        self._equiv = new_equiv        
# Additional methods in accesory files
from .columns import set_column_type, add_column, drop_column, replace_column, rename_column
from .auxiliary import letters_for, random_code, generate_kb_code, generate_mb_code, get_file_size
from .destroy import fast_overwrite, destroy
from .noise import add_gaussian_noise_column, add_impulse_noise_column, add_salt_pepper_noise_column
from .noise import gaussian_column, impulse_column, salt_pepper_column
from .ranges import add_integer_range_column, add_age_range_column, add_float_range_column
from .ranges import integer_range_column, age_range_column, float_range_column
from .synthetic import add_syn_date_column

CS.set_column_type = set_column_type
CS.add_column = add_column
CS.drop_column = drop_column
CS.replace_column = replace_column
CS.rename_column = rename_column
CS.add_gaussian_noise_column = add_gaussian_noise_column
CS.add_impulse_noise_column = add_impulse_noise_column
CS.add_salt_pepper_noise_column = add_salt_pepper_noise_column
CS.gaussian_column = gaussian_column
CS.impulse_column = impulse_column
CS.salt_pepper_column = salt_pepper_column
CS.add_integer_range_column = add_integer_range_column
CS.add_age_range_column = add_age_range_column
CS.add_float_range_column = add_float_range_column
CS.integer_range_column = integer_range_column
CS.age_range_column = age_range_column
CS.float_range_column = float_range_column
CS.add_syn_date_column = add_syn_date_column