import duckdb
from duckdb.typing import BIGINT, INTEGER, VARCHAR
import polars as pl
import pyarrow as pa
from typing import Union, List, Optional, Iterator
import uuid
# import tempfile
import sys
from . import CS

# Assuming _table_name_gen is defined globally in your module,
# perhaps as a generator to create unique temporary table names.
def _generate_unique_table_name() -> Iterator[str]:
    """Generator for unique table names."""
    i = 0
    while True:
        yield f"temp_cs_table_{uuid.uuid4().hex}_{i}"
        i += 1

_table_name_gen = _generate_unique_table_name()


# --- Re-adapted _mask_bigint_val_for_duckdb for DuckDB UDF registration ---
def _mask_bigint_val_for_duckdb(
    value: int,
    mask_left_count: int,
    mask_right_count: int,
    mask_char_str: str
) -> str:
    """
    Internal helper for DuckDB UDF. Masks an integer value to a string.
    Assumes inputs are valid based on DuckDB's type system and pre-checks.
    """
    if not mask_char_str:
        mask_char_single = " "
    else:
        mask_char_single = mask_char_str[0]

    original_str_value = str(value)
    sign = 1
    processed_value_str = original_str_value

    if original_str_value.startswith("-"):
        sign = -1
        processed_value_str = original_str_value[1:]

    current_length = len(processed_value_str)

    actual_mask_left = max(0, min(mask_left_count, current_length))
    actual_mask_right = max(0, min(mask_right_count, current_length - actual_mask_left))

    if actual_mask_left + actual_mask_right >= current_length:
        masked_string_parts = [mask_char_single] * current_length
    else:
        masked_string_parts = list(processed_value_str)
        for i in range(actual_mask_left):
            masked_string_parts[i] = mask_char_single

        for i in range(current_length - actual_mask_right, current_length):
            masked_string_parts[i] = mask_char_single

    final_masked_str = "".join(masked_string_parts)

    if sign < 0:
        final_masked_str = "-" + final_masked_str

    return final_masked_str

def add_masked_column_bigint(self,
    base_column: str,
    new_column_name: Optional[str] = None,
    mask_left: int = 0,
    mask_right: int = 0,
    mask_char: Union[str, int] = "×",
    verbose: bool = False) -> CS:
    """
    Adds a string column of masked values to the .data member (DuckDBPyRelation)
    by applying a mask to an INTEGER base_column.

    The `mask_left` leftmost characters and `mask_right` rightmost characters of
    the INTEGER `base_column` are replaced by `mask_char`.
    The function uses only the first character of `mask_char`.

    Args:
        base_column (str): The name of the existing INTEGER column to mask.
        new_column_name (str, optional): The name for the new masked VARCHAR column.
                                         Defaults to "masked_{base_column}".
        mask_left (int): The number of left digits to mask. Must be non-negative.
        mask_right (int): The number of right digits to mask. Must be non-negative.
        mask_char (Union[str, int]): The character/digit to use for masking.
        verbose (bool): If True, will show debug messages

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER.
    """

    # Determine new_column_name
    if new_column_name is None:
        new_column_name = f"masked_{base_column}"
    elif not isinstance(new_column_name, str) or not new_column_name:
         raise ValueError(f"add_masked_column_bigint: new_column_name must be a non-empty string or None.")

    # Validate base_column
    if not isinstance(base_column, str) or not base_column:
        raise ValueError(f"add_masked_column_bigint: base_column must be a non-empty string.")
   
    # Process mask_char for consistency and validation
    processed_mask_char: str
    if isinstance(mask_char, int):
        processed_mask_char = str(mask_char)
    elif isinstance(mask_char, str) and len(mask_char) > 0:
        processed_mask_char = mask_char[0]
    else:
        raise ValueError(f"add_masked_column_bigint: mask_char must be a non-empty string or an integer.")

    # Validate mask_left and mask_right (Python-side validation)
    if not isinstance(mask_left, int) or mask_left < 0:
        raise ValueError(f"add_masked_column_bigint: mask_left must be a non-negative integer.")
    if not isinstance(mask_right, int) or mask_right < 0:
        raise ValueError(f"add_masked_column_bigint: mask_right must be a non-negative integer.")

    # Ensure self.data is not None before accessing .columns
    if self.data is None:
        raise ValueError(f"add_masked_column_bigint: Data has not been loaded into self.data yet. Cannot process columns.")

    # Register the UDF if not already registered
    fn_name_in_duckdb = 'mask_BIGINT_val'
    get_fn_query = f"SELECT COUNT(*) FROM duckdb_functions() WHERE function_name='{fn_name_in_duckdb}'"
    if verbose:
        print(f"add_masked_column_bigint: {get_fn_query=}", file=sys.stderr)
    int_fn_count = self.cx.execute(get_fn_query).fetchall()[0][0]

    if int_fn_count == 0:
        self.cx.create_function(
            fn_name_in_duckdb,
            _mask_bigint_val_for_duckdb,
            [BIGINT, INTEGER, INTEGER, VARCHAR],
            VARCHAR
        )
    if verbose:
        print(f"add_masked_column_bigint: UDF {fn_name_in_duckdb=} created.", file=sys.stderr)

    # Get actual column names from the relation, converted to lowercase for case-insensitive comparison
    data_column_names_lower = [col.lower() for col in self.data.columns]

    if base_column.lower() not in data_column_names_lower:
        raise ValueError(f"Column '{base_column}' not found in the table.")

    # Now, get the data type. Since we know the column exists, we can query its type.
    # We use base_column directly in the SQL query as DuckDB handles case-insensitivity for unquoted identifiers.
    column_type_query = f"""
        SELECT data_type
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}' AND column_name = '{base_column}'
    """
    if verbose:
        print(f"add_masked_column_bigint: {column_type_query=}", file=sys.stderr)
    result = self.cx.execute(column_type_query).fetchall()

    # # This `if not result:` should theoretically not be hit if the `base_column.lower()` check passed,
    # # but it serves as a robustness check.
    # if not result:
    #      raise ValueError(f"add_masked_column_bigint: Could not retrieve type for column '{base_column}'. This indicates an unexpected state.")

    column_data_type = result[0][0]
    if "INT" not in column_data_type.upper(): # Check for any INT type (BIGINT, INTEGER, SMALLINT etc.)
        raise ValueError(f"Column '{base_column}' is not of INTEGER type. It is '{column_data_type}'.")

    # Add the new masked column and update the self.data member
    existing_columns_query = f"""
        SELECT column_name
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}'
    """
    if verbose:
        print(f"add_masked_column_bigint: {existing_columns_query}", file=sys.stderr)
    existing_columns = [row[0] for row in self.cx.execute(existing_columns_query).fetchall()]
    
    select_columns_str = ", ".join(existing_columns)
    select_columns_str += f", {fn_name_in_duckdb}({base_column}, {mask_left}, {mask_right}, '{processed_mask_char}') AS {new_column_name}"

    # Update self.data to the new relation which includes the masked column
    select_updated = f"SELECT {select_columns_str} FROM {self._tablename}"
    if verbose:
        print(f"add_masked_column_bigint: {select_updated=}", file=sys.stderr)
    self.data = self.cx.sql(select_updated)

    # Re-register the view to reflect the updated self.data for subsequent operations
    # self.data = self.cx.table(self._tablename)

    return self