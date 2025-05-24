from datetime import datetime, timedelta
import duckdb
import numpy as np
import pyarrow as pa
import pandas as pd
import polars as pl
import secrets
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def add_syn_date_column(self, base_column: Optional[str] = None, new_column_name: Optional[str] = None,
                        start_date: Optional[str] = None, end_date: Optional[str] = None,
                        date_format: str = '%Y-%m-%d', verbose: bool = False) -> CS:
    """
    Adds a new column with synthetic random dates based on specified date ranges.

    Args:
        base_column: The name of an existing date/numeric column to use as a boundary for random date generation.
                     If used, its values must be parsable into dates.
        new_column_name: The name of the new synthetic date column. If None, defaults to
                         "syn_{base_column}" if `base_column` is provided, otherwise an error.
        start_date: A string representing the start date for random generation. Must match `date_format`.
        end_date: A string representing the end date for random generation. Must match `date_format`.
        date_format: The format string for parsing/formatting dates (e.g., '%Y-%m-%d').
                     Defaults to '%Y-%m-%d'.
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added synthetic date column.

    Raises:
        ValueError: If invalid arguments are provided, columns not found, or date parsing fails.
        TypeError: If arguments are of incorrect types or base_column values are unparsable.
    """
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Determine new_column_name
    if new_column_name is None:
        if base_column is not None:
            new_column_name = f"syn_{base_column}"
        else:
            raise ValueError("If 'new_column_name' is not provided, 'base_column' must be provided to derive a name.")
    
    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string.")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate base_column if provided
    if base_column is not None:
        valid_columns = [col.lower() for col in self.data.columns]
        if base_column.lower() not in valid_columns:
            raise ValueError(f"Base column '{base_column}' not found in the data.")

    # Parse fixed start_date and end_date if provided
    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    try:
        if start_date:
            parsed_start_date = datetime.strptime(start_date, date_format)
        if end_date:
            parsed_end_date = datetime.strptime(end_date, date_format)
    except ValueError as e:
        raise ValueError(f"Error parsing fixed date '{start_date or end_date}' with format '{date_format}': {e}. Please ensure dates match the format.")

    # Determine the scenario for date generation
    scenario = 0 # 0: fixed range, 1: base_col to end_date, 2: start_date to base_col

    if base_column is None:
        if parsed_start_date is None or parsed_end_date is None:
            raise ValueError("If 'base_column' is not provided, both 'start_date' and 'end_date' must be provided.")
        if parsed_start_date >= parsed_end_date:
            raise ValueError("'start_date' must be before 'end_date' when 'base_column' is not used for range definition.")
        scenario = 0 # Fixed range for all rows
        if verbose:
            print(f"add_syn_date_column: case when fixed range between {start_date} and {end_date}.", file=sys.stderr)
    else:
        if parsed_start_date is not None and parsed_end_date is not None:
            # If all three are given, fixed range takes precedence for simplicity
            scenario = 0
            if verbose:
                print(f"add_syn_date_column: case when fixed range between {start_date} and {end_date} (base_column ignored as boundary).", file=sys.stderr)
        elif parsed_end_date is not None:
            scenario = 1 # base_column to end_date
            if verbose:
                print(f"add_syn_date_column: case when range from '{base_column}' to '{end_date}'.", file=sys.stderr)
        elif parsed_start_date is not None:
            scenario = 2 # start_date to base_column
            if verbose:
                print(f"add_syn_date_column: case when range from '{start_date}' to '{base_column}'.", file=sys.stderr)
        else:
            raise ValueError("When 'base_column' is provided, either 'start_date' or 'end_date' must also be provided to define a range.")

    # Fetch base_column data if needed for per-row range calculation
    base_column_values: Optional[np.ndarray] = None
    is_datetime_dtype = False
    is_numeric_dtype = False

    if scenario in [1, 2]: # Only fetch if base_column is used as a boundary
        base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
        base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
        base_column_values = base_column_data_arrow[base_column].to_numpy()
        
        is_datetime_dtype = np.issubdtype(base_column_values.dtype, np.datetime64)
        is_numeric_dtype = np.issubdtype(base_column_values.dtype, np.number)

        if verbose:
            print(f"add_syn_date_column: got base_column '{base_column}' data (dtype: {base_column_values.dtype}, is_datetime: {is_datetime_dtype}, is_numeric: {is_numeric_dtype}).", file=sys.stderr)

    total_rows_in_data = self.cx.execute(f"SELECT COUNT(*) FROM \"{self._tablename}\"").fetchone()[0]
    if total_rows_in_data == 0:
        print("add_syn_date_column: No rows in data, skipping date generation.", file=sys.stderr)
        return self

    synthetic_dates: List[Optional[str]] = [] # Allow None in the list

    if verbose:
        print(f"add_syn_date_column: synthetic dates for {total_rows_in_data} rows.", file=sys.stderr)

    for i in tqdm(range(total_rows_in_data), disable=not verbose, desc="Generating synthetic dates"):
        current_start_dt: Optional[datetime] = None
        current_end_dt: Optional[datetime] = None
        
        if scenario == 0: # Fixed range for all rows
            current_start_dt = parsed_start_date
            current_end_dt = parsed_end_date
        else: # Per-row range involving base_column
            base_val = base_column_values[i]
            
            # Robust check for missing values (None, NaN, NaT)
            is_missing_value = False
            if base_val is None: # Python None
                is_missing_value = True
            elif is_datetime_dtype and np.isnat(base_val): # NumPy NaT
                is_missing_value = True
            elif is_numeric_dtype and np.isnan(base_val): # NumPy NaN
                is_missing_value = True
            
            if is_missing_value:
                synthetic_dates.append(None)
                continue
            
            try:
                base_dt: datetime
                if is_datetime_dtype:
                    # Convert numpy.datetime64 to Python datetime object
                    # astype(object) converts to Python datetime, then check type
                    base_dt = base_val.astype(object)
                    if not isinstance(base_dt, datetime):
                        raise TypeError(f"Expected datetime object after conversion, got {type(base_dt)}")
                elif isinstance(base_val, str):
                    base_dt = datetime.strptime(base_val, date_format)
                elif is_numeric_dtype: # Assume numeric values are Unix timestamps (seconds since epoch)
                    base_dt = datetime.fromtimestamp(base_val)
                else:
                    raise TypeError(f"Unsupported base_column value type for date conversion: {type(base_val)}")
                # if verbose:
                #     print(f"Parsed base_dt for row {i}: {base_dt}", file=sys.stderr)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse base_column value '{base_val}' at row {i} into a date. Appending None. Error: {e}", file=sys.stderr)
                synthetic_dates.append(None)
                continue

            if scenario == 1: # base_column to end_date
                current_start_dt = base_dt
                current_end_dt = parsed_end_date
            elif scenario == 2: # start_date to base_column
                current_start_dt = parsed_start_date
                current_end_dt = base_dt

        # Ensure valid range for current row before generating random date
        if current_start_dt is None or current_end_dt is None or current_start_dt >= current_end_dt:
            synthetic_dates.append(None) # Append None if range is invalid for this row
            continue

        time_delta = current_end_dt - current_start_dt
        
        # Ensure time_delta is non-negative and has days to pick from
        if time_delta.days < 0: 
            synthetic_dates.append(None)
            continue
        
        # Use secrets.randbelow for random days
        # secrets.randbelow(n) returns a random int in range [0, n-1]
        # time_delta.days is the number of days, so we need to include time_delta.days itself.
        random_days = secrets.randbelow(time_delta.days + 1)
        syn_dt = current_start_dt + timedelta(days=random_days)
        synthetic_dates.append(syn_dt.strftime(date_format))

    try:
        # Create PyArrow Array for the new column
        syn_dates_arrow_array = pa.array(synthetic_dates, type=pa.string()) # Use pa.string() for formatted dates
        
        # Create a temporary PyArrow Table for the new column
        syn_dates_arrow_table = pa.table({new_column_name: syn_dates_arrow_array})

        if verbose:
            print(f"add_syn_date_column: Generated new column data as Arrow Table.", file=sys.stderr)

        # Add the new column using the existing add_column method
        self.add_column(self.cx.from_arrow(syn_dates_arrow_table), new_column_name)
        if verbose:
            print(f"add_syn_date_column: New column '{new_column_name}' added.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_syn_date_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
    return self