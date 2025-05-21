import duckdb
import numpy as np
import pyarrow as pa
import pandas as pd
# import polars as pl
from tqdm import tqdm
from typing import Generator, Union, Optional
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def add_integer_range_column(self, base_column: str, new_column_name: Optional[str] = None,
                             num_ranges: Optional[int] = None, range_size: Optional[int] = None,
                             only_start: bool = False, verbose: bool = False) -> CS:
    """
    Adds a new column containing integer range values based on a base column's min/max.

    Args:
        base_column: The name of the numeric column to derive min/max values from.
        new_column_name: The name of the new column. If None, defaults to "integer_range_{base_column}".
        num_ranges: The desired number of ranges to create.
        range_size: The desired size of each range.
                    Exactly one of 'num_ranges' or 'range_size' must be provided.
        only_start: If True, the new column will contain only the integer start of each range.
                    If False, it will contain a list [range_start, range_end].
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added integer range column.

    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Validate base column name
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    if new_column_name is None:
        new_column_name = f"integer_range_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate range parameters
    if (num_ranges is None and range_size is None) or \
       (num_ranges is not None and range_size is not None):
        raise ValueError("Exactly one of 'num_ranges' or 'range_size' must be provided.")
    if num_ranges is not None and num_ranges <= 0:
        raise ValueError("'num_ranges' must be a positive integer.")
    if range_size is not None and range_size <= 0:
        raise ValueError("'range_size' must be a positive integer.")

    try:
        # Get min and max values from the base column
        min_max_sql = f"SELECT MIN(\"{base_column}\"), MAX(\"{base_column}\") FROM \"{self._tablename}\""
        min_val_raw, max_val_raw = self.cx.execute(min_max_sql).fetchone()

        if min_val_raw is None or max_val_raw is None:
            raise ValueError(f"Could not calculate MIN/MAX for column '{base_column}'. Is it all NULLs or non-numeric?")
        
        # Ensure min/max are integers for range calculation
        min_val = int(min_val_raw)
        max_val = int(max_val_raw)

        # Determine effective range_size and num_ranges
        calculated_range_size = 0
        calculated_num_ranges = 0
        total_span = max_val - min_val + 1

        if total_span <= 0: # Handle cases where min_val >= max_val
            calculated_range_size = 1
            calculated_num_ranges = 1
        elif num_ranges is not None:
            calculated_num_ranges = num_ranges
            calculated_range_size = (total_span + num_ranges - 1) // num_ranges # Ceiling division
        elif range_size is not None:
            calculated_range_size = range_size
            calculated_num_ranges = (total_span + range_size - 1) // range_size # Ceiling division
        
        if calculated_range_size == 0: # Fallback for very small spans if num_ranges is huge
            calculated_range_size = 1
            calculated_num_ranges = total_span # Each value gets its own range

        if verbose:
            print(f"add_integer_range_column: Calculated range_size: {calculated_range_size}, num_ranges: {calculated_num_ranges}", file=sys.stderr)

        # Fetch the base_column data to process in Python. rowid just because
        base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
        base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
        original_values_np = base_column_data_arrow[base_column].to_numpy()

        new_column_values = []
        for val in original_values_np:
            if np.isnan(val):                   # Check for N/As o NULLs
                new_column_values.append(None)
                continue

            # Calculate range for current value
            # Ensure integer conversion for index calculation
            range_idx = (int(val) - min_val) // calculated_range_size
            
            # Ensure range_idx stays within bounds of calculated_num_ranges
            range_idx = max(0, min(range_idx, calculated_num_ranges - 1))

            current_range_start = min_val + range_idx * calculated_range_size
            current_range_end = current_range_start + calculated_range_size - 1
            
            # Adjust the very last range's end to match max_val if needed
            if range_idx == calculated_num_ranges - 1:
                current_range_end = max_val

            if only_start:
                new_column_values.append(current_range_start)
            else:
                new_column_values.append([current_range_start, current_range_end])

        # Create PyArrow Array for the new column
        new_col_arrow_array = None
        if only_start:
            new_col_arrow_array = pa.array(new_column_values, type=pa.int64())
        else:
            # For list of lists, PyArrow needs a ListArray type.
            # Ensure all sublists are of length 2 and type is consistent.
            # Use pa.list_() to define the list type.
            new_col_arrow_array = pa.array(new_column_values, type=pa.list_(pa.int64())) # pa.list_() handles variable length, but here it's fixed at 2

        # Create a temporary PyArrow Table for the new column
        new_col_arrow_table = pa.table({new_column_name: new_col_arrow_array})

        if verbose:
            print(f"4: add_integer_range_column: Generated new column data as Arrow Table.", file=sys.stderr)

        # Add the new column using the existing add_column method
        # add_column expects a DuckDBPyRelation or other types it can convert.
        # self.cx.from_arrow() converts the PyArrow Table to a DuckDBPyRelation.
        self.add_column(self.cx.from_arrow(new_col_arrow_table), new_column_name)
        if verbose:
            print(f"5: add_integer_range_column: New column '{new_column_name}' added.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_integer_range_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        if verbose:
            print("6: add_integer_range_column: End", file=sys.stderr)
    return self