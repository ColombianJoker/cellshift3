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
                             only_start: bool = False, min_range_start: Optional[Union[int, float]] = None,
                             verbose: bool = False) -> CS:
    """
    Adds a new column containing integer range values based on a base column's min/max.

    Args:
        base_column: The name of the numeric column to derive min/max values from.
        new_column_name: The name of the new column. If None, defaults to "integer_range_{base_column}".
        num_ranges: The desired number of ranges to create.
        range_size: The desired size of each range. Exactly one of 'num_ranges' or 'range_size' must be given.
        only_start: If True, the new column will contain only the integer start of each range.
                    If False, it will contain a list [range_start, range_end].
        min_range_start: Optional. If provided, this value will be used as the absolute
                         minimum for range calculation, overriding the minimum value found in 'base_column'.
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
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

    # Validate min_range_start
    if min_range_start is not None and not isinstance(min_range_start, (int, float)):
        raise TypeError("'min_range_start' must be an int or float if provided.")

    try:
        # Get min and max values from the base column
        min_max_sql = f"SELECT MIN(\"{base_column}\"), MAX(\"{base_column}\") FROM \"{self._tablename}\""
        min_val_raw, max_val_raw = self.cx.execute(min_max_sql).fetchone()

        if min_val_raw is None or max_val_raw is None:
            raise ValueError(f"Could not calculate MIN/MAX for column '{base_column}'. Is it all NULLs or non-numeric?")
        
        # Determine the effective min_val for range calculation
        # Use min_range_start if provided, otherwise use the calculated min_val
        effective_min_val = int(min_val_raw)
        if min_range_start is not None:
            effective_min_val = int(min_range_start)

        # Ensure max_val is an integer
        max_val = int(max_val_raw)

        if verbose:
            print(f"add_integer_range_column: Base column '{base_column}' calculated min: {int(min_val_raw)}, max: {max_val}", file=sys.stderr)
            if min_range_start is not None:
                print(f"add_integer_range_column: Using overridden min_range_start: {effective_min_val}", file=sys.stderr)

        # Determine effective range_size and num_ranges
        calculated_range_size = 0
        calculated_num_ranges = 0
        total_span = max_val - effective_min_val + 1

        if total_span <= 0: # Handle cases where effective_min_val >= max_val
            calculated_range_size = 1
            calculated_num_ranges = 1
        elif num_ranges is not None:
            calculated_num_ranges = num_ranges
            calculated_range_size = (total_span + num_ranges - 1) // num_ranges # Integer division
        elif range_size is not None:
            calculated_range_size = range_size
            calculated_num_ranges = (total_span + range_size - 1) // range_size # Integer division
        if calculated_range_size == 0: # To handle some errors
            calculated_range_size = 1
            calculated_num_ranges = total_span
        if verbose:
            print(f"add_integer_range_column: Calculated range_size: {calculated_range_size}, num_ranges: {calculated_num_ranges}", file=sys.stderr)

        # Fetch the base_column data to process in Python
        base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
        base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
        original_values_np = base_column_data_arrow[base_column].to_numpy()

        new_column_values = []
        for val in original_values_np:
            if np.isnan(val): # Check for NaN values (which represent NULLs from DuckDB)
                new_column_values.append(None)
                continue
            # Calculate range for current value. Use effective_min_val here
            range_idx = (int(val) - effective_min_val) // calculated_range_size # Integer division
            # Ensure range_idx stays within limits of calculated_num_ranges
            range_idx = max(0, min(range_idx, calculated_num_ranges - 1))
            current_range_start = effective_min_val + range_idx * calculated_range_size
            current_range_end = current_range_start + calculated_range_size - 1
            # Adjust the very last range's end to match max_val if needed
            if range_idx == calculated_num_ranges - 1 and current_range_end > max_val:
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
            new_col_arrow_array = pa.array(new_column_values, type=pa.list_(pa.int64()))

        # Create a temporary PyArrow Table for the new column
        new_col_arrow_table = pa.table({new_column_name: new_col_arrow_array})
        # Add the new column using the existing add_column method
        self.add_column(self.cx.from_arrow(new_col_arrow_table), new_column_name)
        if verbose:
            print(f"add_integer_range_column: New column '{new_column_name}' added.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_integer_range_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
    return self
    
def add_age_range_column(self, base_column: str, new_column_name: Optional[str] = None,
                         min_age: Optional[Union[int, float]] = None, only_adult: bool = False,
                         num_ranges: Optional[int] = None, range_size: Optional[int] = None,
                         only_start: bool = False, verbose: bool = False) -> CS:
    """
    Adds a new column containing age range values based on a base column.
    Optionally filters data to include only 'adult' rows (>= min_age).

    Args:
        base_column: The name of the numeric column containing age values.
        new_column_name: The name of the new column. If None, defaults to
                         "age_range_{base_column}".
        min_age: Optional. If provided, this value will be used as:
                 1. The filtering threshold if `only_adult` is True.
                 2. The absolute minimum for range calculation (min_range_start) in add_integer_range_column.
        only_adult: If True, rows with `base_column` values less than `min_age`
                    will be removed from the .data member *after* the range column is added.
                    This requires `min_age` to be set.
        num_ranges: The desired number of ranges to create (passed to add_integer_range_column).
        range_size: The desired size of each range (passed to add_integer_range_column).
                    Exactly one of 'num_ranges' or 'range_size' must be provided.
        only_start: If True, the new column will contain only the integer start of each range.
                    If False, it will contain a list [range_start, range_end].
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added age range column (and potentially filtered data).
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Validate base column name
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    if new_column_name is None:
        new_column_name = f"age_range_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate min_age if provided
    if min_age is not None and not isinstance(min_age, (int, float)):
        raise TypeError("'min_age' must be an int or float if provided.")
    if only_adult and min_age is None:
        raise ValueError("If 'only_adult' is True, 'min_age' must be provided.")

    # Determine min_range_start to pass to add_integer_range_column
    # This will be used for the range calculation itself, regardless of filtering
    min_range_start_for_int_range = min_age # Pass min_age directly as min_range_start

    try:
        # Step 1: Call add_integer_range_column to add the age range column to the *current* data
        if verbose:
            print(f"add_age_range_column: Calling add_integer_range_column to add '{new_column_name}'.", file=sys.stderr)
        
        self.add_integer_range_column(
            base_column=base_column,
            new_column_name=new_column_name,
            num_ranges=num_ranges,
            range_size=range_size,
            only_start=only_start,
            min_range_start=min_range_start_for_int_range, # Pass the determined min_range_start
            verbose=verbose
        )
        if verbose:
            print(f"add_age_range_column: Age range column '{new_column_name}' added.", file=sys.stderr)

        # Step 2: Handle 'only_adult' filtering if requested (after column addition)
        if only_adult:
            if verbose:
                print(f"add_age_range_column: Filtering data for only_adult (>= {min_age}).", file=sys.stderr)
            
            # Create a new relation with filtered data
            # Use self._tablename as it now contains the newly added range column
            sql_filter = f"SELECT * FROM \"{self._tablename}\" WHERE \"{base_column}\" >= {min_age};"
            filtered_data = self.cx.execute(sql_filter).fetch_arrow_table()
            # Drop the current table and materialize the filtered relation into it
            sql_drop = f"DROP TABLE IF EXISTS \"{self._tablename}\";"
            self.cx.execute(sql_drop)
            # Create new table from filtered data
            sql_create = f"CREATE TABLE IF NOT EXISTS \"{self._tablename}\" AS SELECT * FROM filtered_data;"
            self.cx.execute(sql_create)
            # Update self.data to point to the newly filtered table
            self.data = self.cx.table(self._tablename)
            
            if verbose:
                new_row_count = self.cx.execute(f'SELECT COUNT(*) FROM \"{self._tablename}\"').fetchone()[0]
                print(f"add_age_range_column: Data filtered. New row count: {new_row_count}", file=sys.stderr)
            
        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_age_range_column for column '{base_column}': {e}", file=sys.stderr)
        raise e # Re-raise the exception to propagate it.

    finally:
        pass
    return self

def add_float_range_column(self, base_column: str, new_column_name: Optional[str] = None,
                           num_ranges: Optional[int] = None, range_size: Optional[Union[int, float]] = None,
                           only_start: bool = False, min_range_start: Optional[Union[int, float]] = None,
                           decimals: int = 1, verbose: bool = False) -> CS:
    """
    Adds a new column containing floating-point range values based on a base column's min/max.

    Args:
        base_column: The name of the numeric column to derive min/max values from.
        new_column_name: The name of the new column. If None, defaults to "float_range_{base_column}".
        num_ranges: The desired number of ranges to create.
        range_size: The desired size of each range.
                    Exactly one of 'num_ranges' or 'range_size' must be provided.
        only_start: If True, the new column will contain only the floating-point start of each range.
                    If False, it will contain a list [range_start, range_end].
        min_range_start: Optional. If provided, this value will be used as the absolute minimum for
                    range calculation, overriding the minimum value found in 'base_column'.
        decimals: The number of decimal places to round the range start and end values to.
                  Defaults to 1.
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Validate base column name
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    if new_column_name is None:
        new_column_name = f"float_range_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")
    if (num_ranges is None and range_size is None) or \
       (num_ranges is not None and range_size is not None):
        raise ValueError("Exactly one of 'num_ranges' or 'range_size' must be provided.")
    if num_ranges is not None and num_ranges <= 0:
        raise ValueError("'num_ranges' must be a positive integer.")
    if range_size is not None and range_size <= 0:
        raise ValueError("'range_size' must be a positive number.")
    # Validate min_range_start
    if min_range_start is not None and not isinstance(min_range_start, (int, float)):
        raise TypeError("'min_range_start' must be an int or float if provided.")
    # Validate decimals
    if not isinstance(decimals, int) or decimals < 0:
        raise TypeError("'decimals' must be a non-negative integer.")

    try:
        # Get min and max values from the base column
        min_max_sql = f"SELECT MIN(\"{base_column}\"), MAX(\"{base_column}\") FROM \"{self._tablename}\""
        min_val_raw, max_val_raw = self.cx.execute(min_max_sql).fetchone()
        if min_val_raw is None or max_val_raw is None:
            raise ValueError(f"Could not calculate MIN/MAX for column '{base_column}'. Is it all NULLs or non-numeric?")
        # Determine the effective min_val for range calculation
        effective_min_val = float(min_val_raw)
        if min_range_start is not None:
            effective_min_val = float(min_range_start)
        max_val = float(max_val_raw)

        # Determine effective range_size and num_ranges
        calculated_range_size = 0.0
        calculated_num_ranges = 0
        total_span = max_val - effective_min_val

        # Handle edge case where total_span is zero or negative
        if total_span <= 0:
            calculated_range_size = 1.0
            calculated_num_ranges = 1
        elif num_ranges is not None:
            calculated_num_ranges = num_ranges
            calculated_range_size = total_span / calculated_num_ranges
        elif range_size is not None:
            calculated_range_size = float(range_size)
            # Calculate num_ranges using ceil for floats
            calculated_num_ranges = int(np.ceil(total_span / calculated_range_size))
            if calculated_num_ranges == 0: # Ensure at least one range if span is tiny
                calculated_num_ranges = 1

        # Calculate epsilon for rounding adjustment. Used to reduce the range_end "a little"
        epsilon = 1 / (10 ** decimals)
        if verbose:
            print(f"add_float_range_column: Epsilon for rounding adjustment: {epsilon}", file=sys.stderr)

        if verbose:
            print(f"add_float_range_column: Calculated range_size: {calculated_range_size}, num_ranges: {calculated_num_ranges}", file=sys.stderr)

        # Get the base_column data to process in Python
        base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
        base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
        original_values_np = base_column_data_arrow[base_column].to_numpy()

        new_column_values = []
        for val in original_values_np:
            if np.isnan(val): # Check for NaN values (which represent NULLs from DuckDB)
                new_column_values.append(None)
                continue

            # Calculate the raw bin index
            if calculated_range_size == 0: # Handle division by zero
                range_idx = 0
            else:
                range_idx = (val - effective_min_val) / calculated_range_size
                range_idx = int(np.floor(range_idx))

            # Ensure range_idx stays within bounds of calculated_num_ranges
            range_idx = max(0, min(range_idx, calculated_num_ranges - 1))
            current_range_start = effective_min_val + range_idx * calculated_range_size
            current_range_start_rounded = round(current_range_start, decimals)
            current_range_end_rounded = 0.0

            if range_idx == calculated_num_ranges - 1:
                current_range_end_rounded = round(max_val, decimals)
                current_range_start_rounded = min(current_range_start_rounded, round(max_val, decimals))
            else:
                next_range_start_raw = effective_min_val + (range_idx + 1) * calculated_range_size
                next_range_start_rounded = round(next_range_start_raw, decimals)
                current_range_end_rounded = next_range_start_rounded - epsilon

            if only_start:
                new_column_values.append(current_range_start_rounded)
            else:
                new_column_values.append([current_range_start_rounded, current_range_end_rounded])

        # Create PyArrow Array for the new column
        new_col_arrow_array = None
        if only_start:
            new_col_arrow_array = pa.array(new_column_values, type=pa.float64())
        else:
            # For list of lists, PyArrow needs a ListArray type.
            # Ensure all sublists are of length 2 and type is consistent.
            new_col_arrow_array = pa.array(new_column_values, type=pa.list_(pa.float64()))

        # Create a temporary PyArrow Table for the new column
        new_col_arrow_table = pa.table({new_column_name: new_col_arrow_array})

        # Add the new column using the existing add_column method
        self.add_column(self.cx.from_arrow(new_col_arrow_table), new_column_name)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_float_range_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
    return self
    
def integer_range_column(self, base_column: str,
                         num_ranges: Optional[int] = None, range_size: Optional[int] = None,
                         only_start: bool = False, min_range_start: Optional[Union[int, float]] = None,
                         verbose: bool = False) -> CS:
    """
    Replaces an existing column with a new column containing integer range values.

    Args:
        base_column: The name of the column to replace with integer range values.
        num_ranges: The desired number of ranges to create.
        range_size: The desired size of each range.
                    Exactly one of 'num_ranges' or 'range_size' must be provided.
        only_start: If True, the new column will contain only the integer start of each range.
                    If False, it will contain a list [range_start, range_end].
        min_range_start: Optional. If provided, this value will be used as the
                         absolute minimum for range calculation, overriding the
                         minimum value found in 'base_column'.
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    # Validate base_column exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    # Define the name for the temporary range column
    temp_range_column_name = f"range_{base_column}"

    try:
        self.add_integer_range_column(
            base_column=base_column,
            new_column_name=temp_range_column_name,
            num_ranges=num_ranges,
            range_size=range_size,
            only_start=only_start,
            min_range_start=min_range_start,
            verbose=verbose
        )
        # Replace the original column with the new range column
        # DuckDB's replace_column (via CREATE TABLE AS SELECT) handles type changes.
        self.replace_column(
            column_to_replace=base_column,
            replace_column=temp_range_column_name
        )
        if verbose:
            print(f"integer_range_column: Column '{base_column}' replaced.", file=sys.stderr)

        # Remove the temporary range column
        self.drop_column(temp_range_column_name)
        if verbose:
            print(f"integer_range_column: Temporary range column '{temp_range_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in integer_range_column for column '{base_column}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # drop_column handles non-existent columns gracefully, so a direct call is fine.
            self.drop_column(temp_range_column_name)
            if verbose:
                print(f"Cleaned up temporary column '{temp_range_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            if verbose:
                print(f"Error during cleanup of temporary column '{temp_range_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        pass
    return self

def age_range_column(self, base_column: str,
                     min_age: Optional[Union[int, float]] = None, only_adult: bool = False,
                     num_ranges: Optional[int] = None, range_size: Optional[int] = None,
                     only_start: bool = False, verbose: bool = False) -> CS:
    """
    Replaces an existing column with a new column containing age range values.
    Optionally filters data to include only 'adult' rows (>= min_age).

    Args:
        base_column: The name of the column to replace with age range values.
        min_age: Optional. If provided, this value will be used as:
                 1. The filtering threshold if `only_adult` is True.
                 2. The absolute minimum for range calculation (min_range_start).
        only_adult: If True, rows with `base_column` values less than `min_age`
                    will be removed from the .data member. This requires `min_age` to be set.
        num_ranges: The desired number of ranges to create.
        range_size: The desired size of each range.
                    Exactly one of 'num_ranges' or 'range_size' must be provided.
        only_start: If True, the new column will contain only the integer start of each range.
                    If False, it will contain a list [range_start, range_end].
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    # Validate base_column exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    # Define the name for the temporary range column
    temp_range_column_name = f"range_age_{base_column}" # More specific temporary name

    try:
        # Create a new temporary column with age range values
        self.add_age_range_column(
            base_column=base_column,
            new_column_name=temp_range_column_name,
            min_age=min_age,
            only_adult=only_adult,
            num_ranges=num_ranges,
            range_size=range_size,
            only_start=only_start,
            verbose=verbose
        )
        if verbose:
            print(f"age_range_column: Temporary age range column '{temp_range_column_name}' added.", file=sys.stderr)

        # Replace the original column with the new range column
        # DuckDB's replace_column (via CREATE TABLE AS SELECT) handles type changes.
        self.replace_column(
            column_to_replace=base_column,
            replace_column=temp_range_column_name
        )
        if verbose:
            print(f"age_range_column: Column '{base_column}' replaced.", file=sys.stderr)

        # Remove the temporary range column
        self.drop_column(temp_range_column_name)
        if verbose:
            print(f"age_range_column: Temporary range column '{temp_range_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in age_range_column for column '{base_column}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # drop_column handles non-existent columns gracefully, so a direct call is fine.
            self.drop_column(temp_range_column_name)
            if verbose:
                print(f"Cleaned up temporary column '{temp_range_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            if verbose:
                print(f"Error during cleanup of temporary column '{temp_range_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        pass
    return self