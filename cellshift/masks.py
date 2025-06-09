import duckdb
from duckdb.typing import BIGINT, INTEGER, VARCHAR
import polars as pl
import pyarrow as pa
from typing import Union, List, Optional, Iterator
import uuid
# import tempfile
import secrets
import sys
from . import CS
from . import table_name_generator

_table_name_gen = table_name_generator()

def _mask_value_for_duckdb(value: Union[int, str],
                           mask_left_count: int,
                           mask_right_count: int,
                           mask_char_str: str) -> str:
    """
    Internal helper for DuckDB UDF. Masks an integer or string value to a string.
    Handles numeric signs if applicable.
    """
    if not mask_char_str:
        mask_char_single = " "
    else:
        mask_char_single = mask_char_str[0]

    original_str_value = str(value)
    processed_value_str = original_str_value
    sign_prefix = ""

    # Adjust sign handling to be robust for both int and string inputs
    if isinstance(value, int):
        if value < 0:
            sign_prefix = "-"
            processed_value_str = original_str_value[1:]
    elif isinstance(value, str):
        # If it's a string that starts with '-' and the rest is digits, treat it as a negative number string
        if original_str_value.startswith("-") and original_str_value[1:].isdigit():
            sign_prefix = "-"
            processed_value_str = original_str_value[1:]
        # Otherwise, for general strings, no special sign handling; mask as is.

    current_length = len(processed_value_str)

    actual_mask_left = max(0, min(mask_left_count, current_length))
    actual_mask_right = max(0, min(mask_right_count, current_length - actual_mask_left))

    if actual_mask_left + actual_mask_right >= current_length:
        masked_string_parts = [mask_char_single] * current_length
    else:
        masked_string_parts = list(processed_value_str)
        # Mask left part
        for i in range(actual_mask_left):
            masked_string_parts[i] = mask_char_single

        # Mask right part (from the original string's right end)
        for i in range(current_length - actual_mask_right, current_length):
            masked_string_parts[i] = mask_char_single

    final_masked_str = "".join(masked_string_parts)

    return sign_prefix + final_masked_str
    
def add_masked_column(self,
                      base_column: str,
                      new_column_name: Optional[str] = None,
                      mask_left: int = 0,
                      mask_right: int = 0,
                      mask_char: Union[str, int] = "×",
                      verbose: bool = False) -> CS:
    """
    Adds a string column of masked values to the .data member (DuckDBPyRelation)
    by applying a mask to an INTEGER or VARCHAR base_column.

    The `mask_left` leftmost characters and `mask_right` rightmost characters of
    the `base_column` are replaced by `mask_char`.
    The function uses only the first character of `mask_char`.

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        new_column_name (str, optional): The name for the new masked VARCHAR column.
                                         Defaults to "masked_{base_column}".
        mask_left (int): The number of left digits/characters to mask. Must be non-negative.
        mask_right (int): The number of right digits/characters to mask. Must be non-negative.
        mask_char (Union[str, int]): The character/digit to use for masking.
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """

    # Determine new_column_name
    if new_column_name is None:
        new_column_name = f"masked_{base_column}"
    elif not isinstance(new_column_name, str) or not new_column_name:
        raise ValueError(f"add_masked_column: new_column_name must be a non-empty string or None.")

    # Validate base_column
    if not isinstance(base_column, str) or not base_column:
        raise ValueError(f"add_masked_column: base_column must be a non-empty string.")

    # Process mask_char for consistency and validation
    processed_mask_char: str
    if isinstance(mask_char, int):
        processed_mask_char = str(mask_char)
    elif isinstance(mask_char, str) and len(mask_char) > 0:
        processed_mask_char = mask_char[0]
    else:
        raise ValueError(f"add_masked_column: mask_char must be a non-empty string or an integer.")

    # Validate mask_left and mask_right (Python-side validation)
    if not isinstance(mask_left, int) or mask_left < 0:
        raise ValueError(f"add_masked_column: mask_left must be a non-negative integer.")
    if not isinstance(mask_right, int) or mask_right < 0:
        raise ValueError(f"add_masked_column: mask_right must be a non-negative integer.")

    # Ensure self.data is not None before accessing .columns
    if self.data is None:
        raise ValueError(f"add_masked_column: Data has not been loaded into self.data yet. Cannot process columns.")

    # Get actual column names from the relation, converted to lowercase for case-insensitive comparison
    data_column_names_lower = [col.lower() for col in self.data.columns]

    if base_column.lower() not in data_column_names_lower:
        raise ValueError(f"Column '{base_column}' not found in the table.")

    # Get the data type of the base_column
    column_type_query = f"""
        SELECT data_type
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}' AND column_name = '{base_column}'
    """
    if verbose:
        print(f"add_masked_column: {column_type_query=}", file=sys.stderr)
    result = self.cx.execute(column_type_query).fetchall()

    if not result:
         raise ValueError(f"add_masked_column: Could not retrieve type for column '{base_column}'. This indicates an unexpected state.")

    column_data_type = result[0][0].upper() # Convert to uppercase for consistent comparison

    # --- Determine UDF name and type flag based on column_data_type ---
    int_version_fn = False
    fn_name_in_duckdb: str

    if "INT" in column_data_type: # matches BIGINT, INTEGER, SMALLINT etc.
        fn_name_in_duckdb = 'mask_int_val'
        int_version_fn = True
    elif column_data_type in ("CHAR", "VARCHAR", "STRING"): # Also includes CHAR
        fn_name_in_duckdb = 'mask_char_val'
        int_version_fn = False # Explicitly set to False for char/varchar/string
    else:
        raise ValueError(f"Column '{base_column}' is not of an allowed type (INTEGER or VARCHAR/CHAR/STRING). It is '{column_data_type}'.")

    # --- Register the appropriate UDF if not already registered ---
    get_fn_query = f"SELECT COUNT(*) FROM duckdb_functions() WHERE function_name='{fn_name_in_duckdb}'"
    if verbose:
        print(f"add_masked_column: {get_fn_query=}", file=sys.stderr)
    fn_count = self.cx.execute(get_fn_query).fetchall()[0][0]

    if fn_count == 0:
        if int_version_fn:
            self.cx.create_function(
                fn_name_in_duckdb,
                _mask_value_for_duckdb, # Use the generic Python function
                [BIGINT, INTEGER, INTEGER, VARCHAR], # DuckDB expects BIGINT if the column is an integer type
                VARCHAR
            )
            if verbose:
                print(f"add_masked_column: UDF {fn_name_in_duckdb=} (INT version) created.", file=sys.stderr)
        else: # not int_version_fn, meaning it's a CHAR/VARCHAR/STRING type
            self.cx.create_function(
                fn_name_in_duckdb,
                _mask_value_for_duckdb, # Use the generic Python function
                [VARCHAR, INTEGER, INTEGER, VARCHAR], # DuckDB expects VARCHAR for string types
                VARCHAR
            )
            if verbose:
                print(f"add_masked_column: UDF {fn_name_in_duckdb=} (CHAR version) created.", file=sys.stderr)
    else:
        if verbose:
            print(f"add_masked_column: UDF {fn_name_in_duckdb=} already registered.", file=sys.stderr)

    # Update self.data to the new relation which includes the masked column
    alter_add_column = f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" VARCHAR;"
    if verbose:
        print(f"{alter_add_column=}", file=sys.stderr)
    self.cx.execute(alter_add_column)
    update_new_column = f"""UPDATE \"{self._tablename}\"
                            SET \"{new_column_name}\"={fn_name_in_duckdb}(\"{base_column}\", {mask_left}, {mask_right}, '{processed_mask_char}')
                           """
    if verbose:
        print(f"{update_new_column=}", file=sys.stderr)
    self.cx.execute(update_new_column)
    self.data = self.cx.table(self._tablename)
    if verbose:
        print(f"{self.data.columns=}", file=sys.stderr)
        self.data.show()

    return self

def add_masked_mail_column(self,
                      base_column: str,
                      new_column_name: Optional[str] = None,
                      mask_user: Optional[Union[bool, str]] = False,
                      mask_domain: Optional[Union[bool, str]] = False,
                      domain_choices: Optional[Union[bool, str, List[str]]] = False,
                      verbose: bool = False) -> CS:
    """
    Adds a string column of masked values to the .data member by replacing the user part of email
        if chosen, replacing the domain part if chosen, and replacing from a list of domains if given

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        new_column_name (str, optional): The name for the new masked VARCHAR column.
                                         Defaults to "masked_{base_column}".
        mask_user (Optional[Union[bool, str]]): Or True (replaces with a default) or a string with
                                                the replacement (for all values in base_column).
                                                Defaults to None (don't replace).
        mask_domain (Optional[Union[bool, str]]): Or True (replaces with something) or a string with
                                                  the replacement (for all values in base_column).
        domain_choices (Optional[Union[bool, str]]): True (replaces with something) or a list
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """

    # Determine new_column_name
    if new_column_name is None:
        new_column_name = f"masked_{base_column}"
    elif not isinstance(new_column_name, str) or not new_column_name:
        raise ValueError(f"add_masked_column: new_column_name must be a non-empty string or None.")

    # Validate base_column
    if not isinstance(base_column, str) or not base_column:
        raise ValueError(f"add_masked_column: base_column must be a non-empty string.")

    # Validate mask_user
    if not isinstance(mask_user, bool) and not isinstance(mask_user, str):
        raise ValueError(f"add_masked_column: mask_user must be boolean or a string.")
    # Validate mask_domain
    if not isinstance(mask_domain, bool) and not isinstance(mask_domain, str):
        raise ValueError(f"add_masked_column: mask_domain must be boolean or a string.")
    # Validate domain_choices
    if not isinstance(domain_choices, bool) and \
      (not isinstance(domain_choices, str) and not isinstance(domain_choices, List)):
        raise ValueError(f"add_masked_column: domain_choices must be boolean or a string or a list of strings.")
    
    # Ensure self.data is not None before accessing .columns
    if self.data is None:
        raise ValueError(f"add_masked_column: Data has not been loaded into self.data yet. Cannot process columns.")

    # Get actual column names from the relation, converted to lowercase for case-insensitive comparison
    data_column_names_lower = [col.lower() for col in self.data.columns]

    if base_column.lower() not in data_column_names_lower:
        raise ValueError(f"Column '{base_column}' not found in the table.")

    # Get the data type of the base_column
    column_type_query = f"""
        SELECT data_type
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}' AND column_name = '{base_column}'
    """
    if verbose:
        print(f"add_masked_column: {column_type_query=}", file=sys.stderr)
    result = self.cx.execute(column_type_query).fetchall()

    if not result:
         raise ValueError(f"add_masked_column: Could not retrieve type for column '{base_column}'. This indicates an unexpected state.")

    column_data_type = result[0][0].upper() # Convert to uppercase for consistent comparison

    fn_name_in_duckdb: str

    if column_data_type in ("CHAR", "VARCHAR", "STRING"): # Also includes CHAR
        fn_name_in_duckdb = 'mask_char_val'
    else:
        raise ValueError(f"Column '{base_column}' is not of an allowed type (VARCHAR/CHAR/STRING). It is '{column_data_type}'.")

    user_mask: str
    domain_mask: str
    resrc: str
    remsk: str
    
    if mask_user: # Handle "mask_user was given"
        if mask_user is True:
          user_mask='????????'
        else:
          user_mask = mask_user
    if mask_domain:
        if mask_domain is True:
            if not domain_choices:
                domain_mask = secrets.choice([
                  "example.com",
                  "example.org",
                  "example.net",
                  "example.edu",
                  "example.co",
                ])
            else:
              domain_mask = secrets.choice(domain_choices)
        else:
            domain_mask = mask_domain
    if mask_user and mask_domain:
        resrc = "^([\w._-]+)@([\w._-]+)"
        remsk = f"{user_mask}@{domain_mask}"
    elif mask_user:
        resrc="^([\w._-]+)@"
        remsk=f"{user_mask}@"
    elif mask_domain:
        resrc="@([\w._-]+)"
        remsk=f"@{domain_mask}"
    if mask_user or mask_domain:
      # Update self.data to the new relation which includes the masked column
      alter_add_column: str = f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" VARCHAR;"
      if verbose:
          print(f"{alter_add_column=}", file=sys.stderr)
      self.cx.execute(alter_add_column)
      update_new_column: str = f"""UPDATE \"{self._tablename}\"
                              SET \"{new_column_name}\"=regexp_replace({base_column}, '{resrc}', '{remsk}');
                             """
      if verbose:
          print(f"{update_new_column=}", file=sys.stderr)
      self.cx.execute(update_new_column)
      self.data = self.cx.table(self._tablename)
      # new_data = self.cx.sql(select_updated)
      # self.data = self.cx.from_query(select_updated)
      if verbose:
          print(f"{self.data.columns=}", file=sys.stderr)
          self.data.show()

    return self