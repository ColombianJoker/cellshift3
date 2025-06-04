from datetime import datetime, timedelta
import duckdb
from faker import Faker
import numpy as np
import pyarrow as pa
import pandas as pd
import polars as pl
import secrets
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def add_syn_date_column(self, 
                        base_column: str, 
                        new_column_name: Optional[str] = None,
                        start_date: str = None,
                        date_format: str = '%Y-%m-%d', 
                        verbose: bool = False) -> 'CS':
    """
    Adds a new column with synthetic random dates generated between a fixed start_date
    and the corresponding date value in the base_column.

    Args:
        base_column: The name of an existing date column to use as the upper boundary
                     for random date generation. Its values must be parsable by `date_format`.
        new_column_name: The name of the new synthetic date column. If None, defaults to
                         "syn_{base_column}".
        start_date: A string representing the fixed start date for random generation.
                    Must be parsable by `date_format`. This is the lower boundary.
        date_format: The format string for parsing/formatting dates (e.g., '%Y-%m-%d').
                     Defaults to '%Y-%m-%d'.
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added synthetic date column.

    """
    if verbose:
        print(f"add_syn_date_column: Start, base_column='{base_column}', new_column_name='{new_column_name}', start_date='{start_date}', date_format='{date_format}'", file=sys.stderr)

    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Determine new_column_name
    if new_column_name is None:
        new_column_name = f"syn_{base_column}"
    
    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string.")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate base_column exists
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Base column '{base_column}' not found in the data.")

    # Validate start_date
    if start_date is None:
        raise ValueError("'start_date' must be provided for synthetic date generation.")

    parsed_start_date: datetime = None
    try:
        parsed_start_date = datetime.strptime(start_date, date_format)
    except ValueError as e:
        raise ValueError(f"Error parsing 'start_date' ('{start_date}') with format '{date_format}': {e}. Please ensure date matches the format.")

    # Initialize Faker with the class's locale
    faker_instance = Faker(self.faker_locale)
    if verbose:
        print(f"add_syn_date_column: Initialized Faker with locale: '{self.faker_locale}'.", file=sys.stderr)

    # Fetch base_column data
    base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
    base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
    original_base_values_np = base_column_data_arrow[base_column].to_numpy()
    
    is_datetime_dtype = np.issubdtype(original_base_values_np.dtype, np.datetime64)
    is_numeric_dtype = np.issubdtype(original_base_values_np.dtype, np.number) # For unix timestamps
    if verbose:
        print(f"3: add_syn_date_column: Fetched base_column '{base_column}' data (dtype: {original_base_values_np.dtype}).", file=sys.stderr)

    total_rows_in_data = self.cx.execute(f"SELECT COUNT(*) FROM \"{self._tablename}\"").fetchone()[0]
    if total_rows_in_data == 0:
        print("add_syn_date_column: No rows in data, skipping date generation.", file=sys.stderr)
        return self

    synthetic_dates: List[Optional[str]] = []

    if verbose:
        print(f"add_syn_date_column: Generating synthetic dates for {total_rows_in_data} rows.", file=sys.stderr)

    for i in tqdm(range(total_rows_in_data), 
                  disable=not (verbose and (total_rows_in_data>1000)), 
                  desc="Generating synthetic dates"):
        base_val = original_base_values_np[i]
        
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
                base_dt = base_val.astype(object)
                if not isinstance(base_dt, datetime):
                    raise TypeError(f"Expected datetime object after conversion, got {type(base_dt)}")
            elif isinstance(base_val, str):
                base_dt = datetime.strptime(base_val, date_format)
            elif is_numeric_dtype: # Assume numeric values are Unix timestamps (seconds since epoch)
                base_dt = datetime.fromtimestamp(base_val)
            else:
                raise TypeError(f"Unsupported base_column value type for date conversion: {type(base_val)}")

            # Ensure the range is valid: start_date <= base_dt
            if parsed_start_date > base_dt:
                if verbose:
                    print(f"Warning: start_date ({parsed_start_date.strftime(date_format)}) is after base_column value ({base_dt.strftime(date_format)}) at row {i}. Appending None.", file=sys.stderr)
                synthetic_dates.append(None)
                continue
            
            # Generate random date using Faker
            syn_dt = faker_instance.date_between_dates(date_start=parsed_start_date, date_end=base_dt)
            synthetic_dates.append(syn_dt.strftime(date_format))

        except Exception as e:
            if verbose:
                print(f"Warning: Could not parse base_column value '{base_val}' at row {i} or generate date. Appending None. Error: {e}", file=sys.stderr)
            synthetic_dates.append(None)
            continue

    try:
        # Create PyArrow Array for the new column
        syn_dates_arrow_array = pa.array(synthetic_dates, type=pa.string()) # Use pa.string() for formatted dates
        
        if verbose:
            print(f"add_syn_date_column: Generated new column data as PyArrow Array.", file=sys.stderr)
            if total_rows_in_data<=100:
              print(f"{syn_dates_arrow_array=}", file=sys.stderr)
            print(f"{self.data.shape=}", file=sys.stderr)

        # Add the new column using the existing add_column method
        # Pass the PyArrow Array directly and the verbose argument
        self.add_column(syn_dates_arrow_array, new_column_name, verbose=verbose)
        if verbose:
            print(f"add_syn_date_column: New column '{new_column_name}' added.", file=sys.stderr)
            print(f"{self.data.shape=}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in add_syn_date_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        if verbose:
            print("7: add_syn_date_column: End", file=sys.stderr)