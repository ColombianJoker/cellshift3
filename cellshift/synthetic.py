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
                        base_column: Optional[str] = None,
                        new_column_name: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        date_format: str = '%Y-%m-%d', 
                        verbose: bool = False) -> 'CS':
    """
    Adds a new column with synthetic random dates based on specified date ranges.
    The date range can be fixed (start_date to end_date) or dynamic (involving base_column).

    Args:
        base_column: The name of an existing date/numeric column to use as a boundary for random date generation.
                     If used, its values must be parsable by `date_format`.
        new_column_name: The name of the new synthetic date column. If None, defaults to
                         "syn_{base_column}" if `base_column` is provided, otherwise an error.
        start_date: A string representing the start date for random generation. Must match `date_format`.
        end_date: A string representing the end date for random generation. Must match `date_format`.
                  Exactly one of (start_date, end_date) or (base_column + one of start_date/end_date)
                  must define a valid range.
        date_format: The format string for parsing/formatting dates. Defaults to '%Y-%m-%d'.
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added synthetic date column.
    """
    if verbose:
        print(f"add_syn_date_column: Start, base_column='{base_column}', new_column_name='{new_column_name}', start_date='{start_date}', end_date='{end_date}', date_format='{date_format}'", file=sys.stderr)

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

    # Validate start_date and end_date presence
    if start_date is None and end_date is None:
        raise ValueError("Either 'start_date' or 'end_date' (or both) must be provided to define a date range.")

    parsed_start_date: Optional[datetime] = None
    parsed_end_date: Optional[datetime] = None

    try:
        if start_date:
            parsed_start_date = datetime.strptime(start_date, date_format)
        if end_date:
            parsed_end_date = datetime.strptime(end_date, date_format)
    except ValueError as e:
        raise ValueError(f"Error parsing provided date(s) with format '{date_format}': {e}. Please ensure dates match the format.")

    # Initialize Faker with the class's locale
    faker_instance = Faker(self.faker_locale)
    if verbose:
        print(f"add_syn_date_column: Initialized Faker with locale: '{self.faker_locale}'.", file=sys.stderr)

    # Fetch base_column data if it's needed as a dynamic boundary
    base_column_values: Optional[np.ndarray] = None
    is_datetime_dtype = False
    is_numeric_dtype = False # For unix timestamps

    # Determine if base_column is used as a dynamic boundary
    # It's dynamic if only one of start_date/end_date is given AND base_column is provided
    is_base_column_dynamic_boundary = (base_column is not None) and \
                                      ((start_date is None and end_date is not None) or \
                                       (start_date is not None and end_date is None))

    if is_base_column_dynamic_boundary:
        base_column_values_sql = f"SELECT \"{base_column}\" FROM \"{self._tablename}\" ORDER BY rowid"
        base_column_data_arrow = self.cx.execute(base_column_values_sql).fetch_arrow_table()
        original_base_values_np = base_column_data_arrow[base_column].to_numpy()
        
        is_datetime_dtype = np.issubdtype(original_base_values_np.dtype, np.datetime64)
        is_numeric_dtype = np.issubdtype(original_base_values_np.dtype, np.number)
        if verbose:
            print(f"add_syn_date_column: Fetched base_column '{base_column}' data (dtype: {original_base_values_np.dtype}) for dynamic boundary.", file=sys.stderr)

    total_rows_in_data = self.cx.execute(f"SELECT COUNT(*) FROM \"{self._tablename}\"").fetchone()[0]
    if total_rows_in_data == 0:
        print("add_syn_date_column: No rows in data, skipping date generation.", file=sys.stderr)
        return self

    synthetic_dates: List[Optional[str]] = []

    if verbose:
        print(f"add_syn_date_column: Generating synthetic dates for {total_rows_in_data} rows.", file=sys.stderr)

    for i in tqdm(range(total_rows_in_data), 
                  disable=not (verbose and (total_rows_in_data > 1000)), 
                  desc="Generating synthetic dates"):
        
        current_date_start_for_faker: Optional[datetime] = None
        current_date_end_for_faker: Optional[datetime] = None

        if start_date is not None and end_date is not None:
            # Scenario 1: Both start_date and end_date are fixed. Ignore base_column.
            if parsed_start_date >= parsed_end_date:
                if verbose:
                    print(f"Warning: Fixed start_date ({start_date}) is not before fixed end_date ({end_date}). Appending None for row {i}.", file=sys.stderr)
                synthetic_dates.append(None)
                continue
            current_date_start_for_faker = parsed_start_date
            current_date_end_for_faker = parsed_end_date
            if verbose and i == 0: # Only print once for fixed scenario
                 print(f"Scenario: Fixed range from {current_date_start_for_faker.strftime(date_format)} to {current_date_end_for_faker.strftime(date_format)}.", file=sys.stderr)

        elif is_base_column_dynamic_boundary:
            # Scenarios 2 & 3: base_column is a dynamic boundary
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
                    base_dt = base_val.astype(object)
                    if not isinstance(base_dt, datetime):
                        raise TypeError(f"Expected datetime object after conversion, got {type(base_dt)}")
                elif isinstance(base_val, str):
                    base_dt = datetime.strptime(base_val, date_format)
                elif is_numeric_dtype: # Assume numeric values are Unix timestamps (seconds since epoch)
                    base_dt = datetime.fromtimestamp(base_val)
                else:
                    raise TypeError(f"Unsupported base_column value type for date conversion: {type(base_val)}")

                if start_date is not None and end_date is None:
                    # Scenario 2: start_date fixed, base_column is end_date
                    current_date_start_for_faker = parsed_start_date
                    current_date_end_for_faker = base_dt
                    if verbose and i == 0:
                        print(f"Scenario: Dynamic range from fixed start_date to base_column value.", file=sys.stderr)
                elif start_date is None and end_date is not None:
                    # Scenario 3: base_column is start_date, end_date fixed
                    current_date_start_for_faker = base_dt
                    current_date_end_for_faker = parsed_end_date
                    if verbose and i == 0:
                        print(f"Scenario: Dynamic range from base_column value to fixed end_date.", file=sys.stderr)
                
                # Validate dynamic range for current row
                if current_date_start_for_faker >= current_date_end_for_faker:
                    if verbose:
                        print(f"Warning: Dynamic start date ({current_date_start_for_faker.strftime(date_format)}) is not before end date ({current_date_end_for_faker.strftime(date_format)}) at row {i}. Appending None.", file=sys.stderr)
                    synthetic_dates.append(None)
                    continue

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse base_column value '{base_val}' at row {i} into a date. Appending None. Error: {e}", file=sys.stderr)
                synthetic_dates.append(None)
                continue
        else:
            # This case should ideally be caught by initial validation, but as a fallback
            if verbose:
                print(f"Warning: No valid date generation scenario for row {i}. Appending None.", file=sys.stderr)
            synthetic_dates.append(None)
            continue

        try:
            # Generate random date using Faker for the determined range
            syn_dt = faker_instance.date_between_dates(date_start=current_date_start_for_faker, date_end=current_date_end_for_faker)
            synthetic_dates.append(syn_dt.strftime(date_format))
        except Exception as e:
            if verbose:
                print(f"Warning: Faker failed to generate date for row {i} within range [{current_date_start_for_faker}, {current_date_end_for_faker}]. Appending None. Error: {e}", file=sys.stderr)
            synthetic_dates.append(None)
            continue

    try:
        # Create PyArrow Array for the new column
        syn_dates_arrow_array = pa.array(synthetic_dates, type=pa.string()) # Use pa.string() for formatted dates
        
        if verbose:
            print(f"add_syn_date_column: Generated new column data as PyArrow Array.", file=sys.stderr)
            if total_rows_in_data <= 100 and verbose:
                print(f"{syn_dates_arrow_array=}", file=sys.stderr)
            print(f"{self.data.shape=}", file=sys.stderr)

        # Add the new column using the existing add_column method
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
        pass
    return self

def syn_date_column(self, 
                    base_column: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    date_format: str = '%Y-%m-%d',
                    verbose: bool = False) -> CS:
    """
    Replaces an existing column with a new column containing synthetic random dates.

    Args:
        base_column: The name of the column to replace with synthetic date values.
                     Its values might be used as a boundary for date generation if start_date/end_date
                     are not both fixed.
        start_date: A string representing the start date for random generation. Must match `date_format`.
        end_date: A string representing the end date for random generation. Must match `date_format`.
        date_format: The format string for parsing/formatting dates. Defaults to '%Y-%m-%d'.
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the 'base_column' replaced by synthetic date values.

    Raises:
        ValueError: If base_column is not found, or if date generation parameters are invalid.
        TypeError: If arguments are of incorrect types.
    """
    if verbose:
        print(f"syn_date_column: Start for column '{base_column}'", file=sys.stderr)
    
    # Validate base_column exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    # Define the name for the temporary synthetic date column
    # Using a more specific name to avoid potential conflicts
    temp_syn_date_column_name = f"_temp_syn_date_{base_column}"

    try:
        if verbose:
            print(f"syn_date_column: Adding temporary synthetic date column '{temp_syn_date_column_name}'.", file=sys.stderr)
        self.add_syn_date_column(
            base_column=base_column,
            new_column_name=temp_syn_date_column_name,
            start_date=start_date,
            end_date=end_date,
            date_format=date_format,
            verbose=verbose
        )
        if verbose:
            print(f"syn_date_column: Temporary synthetic date column '{temp_syn_date_column_name}' added.", file=sys.stderr)

        # replace_column handles potential type changes.
        if verbose:
            print(f"syn_date_column: Replacing '{base_column}' with '{temp_syn_date_column_name}'.", file=sys.stderr)
        self.replace_column(
            column_to_replace=base_column,
            replace_column=temp_syn_date_column_name,
            verbose=verbose # Pass verbose to replace_column
        )
        if verbose:
            print(f"syn_date_column: Column '{base_column}' replaced.", file=sys.stderr)

        # Step 3: Remove the temporary synthetic date column
        if verbose:
            print(f"syn_date_column: Dropping temporary synthetic date column '{temp_syn_date_column_name}'.", file=sys.stderr)
        self.drop_column(temp_syn_date_column_name, verbose=verbose) # Pass verbose to drop_column
        if verbose:
            print(f"syn_date_column: Temporary synthetic date column '{temp_syn_date_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in syn_date_column for column '{base_column}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # Check if the temporary column exists before trying to drop it
            # This check is robust as drop_column handles non-existent columns gracefully.
            self.drop_column(temp_syn_date_column_name, verbose=False) # Use verbose=False for cleanup
            if verbose:
                print(f"Cleaned up temporary column '{temp_syn_date_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            if verbose:
                print(f"Error during cleanup of temporary column '{temp_syn_date_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        pass
    return self

def add_syn_city_column(self, 
                        base_column: str, 
                        new_column_name: Optional[str] = None, 
                        max_uniques: Optional[int] = 1000,
                        verbose: bool = False) -> CS:
    """
    Add a new column with synthetic generated town/city names
    
    Args:
        base_column: the name a column to base the generation
        new_column_name: The name of the new synthetic city column. If None, defaults to
                         "syn_{base_column}" if `base_column` is provided, otherwise an error.
        max_uniques: max number of names save equivalence table
        verbose: if to show debug messages   

    Returns:
        self: The CS object with the 'base_column' replaced by synthetic date values.
    """

    if (new_column_name is None):
      if (isinstance(base_column, str)):
          new_column_name = f"syn_{base_column}"
      else:
          raise ValueError("add_syn_city_column: no new_column_name given and base_column is not valid.")
    fake=Faker(self.faker_locale)
    Faker.seed(42)
    if isinstance(base_column,str):
        # Get unique names in base_column
        n_uniques = self.cx.sql(f'SELECT COUNT(DISTINCT "{base_column}") FROM "{self._tablename}";').fetchall()[0][0]
        if verbose:
          print(f"add_syn_city_column: processing '{self._tablename}', with {n_uniques} values.", file=sys.stderr)
        if n_uniques>max_uniques:
            try:
                # Generate without equivalences
                add_column_sql = f'ALTER TABLE "{self._tablename}" ADD COLUMN "{new_column_name}" VARCHAR;'
                self.cx.sql(add_column_sql)
                self.data = self.cx.table(self._tablename)
                if verbose:
                    print(f"{self.data.columns=}", file=sys.stderr)
                    print(self.cx.sql(f'SELECT COUNT(*) FROM "{self._tablename}";').fetchone()[0], file=sys.stderr)
                all_count = self.cx.sql(f'SELECT COUNT(*) FROM "{self._tablename}";').fetchone()[0]
                for row_id in tqdm(range(all_count),
                                   disable=not (verbose and (n_uniques>1000))):
                    fake_city=fake.city()
                    # update_sql = f'UPDATE "{self._tablename}" SET "{new_column_name}"="{fake_city}" WHERE rowid=={row_id};'
                    update_sql = f"UPDATE \"{self._tablename}\" SET \"{new_column_name}\"='{fake_city}' WHERE rowid=={row_id};"
                    # if verbose:
                    #     print(f"{update_sql=}", file=sys.stderr)
                    self.cx.execute(update_sql)
            except Exception as e:
                printf(f"{e=}", file=sys.stderr)
                raise Exception
        else: # So little cities that can be saved the equivalences
            try:
                new_table_name = "city_equivalences" # f"temp_city_{self._tablename}"
                create_temp_sql = f"""
                    CREATE TABLE "{new_table_name}" AS
                    SELECT DISTINCT
                        "{base_column}",
                        CAST(NULL AS VARCHAR) AS "{new_column_name}"
                    FROM
                        "{self._tablename}";
                """
                if verbose:
                    print(f"{create_temp_sql=}", file=sys.stderr)
                self.cx.execute(create_temp_sql) # Create table for equivalences
                for row_id in tqdm(range(n_uniques),
                                   disable=not (verbose and (n_uniques>1000))):
                    fake_city=fake.city()
                    update_sql = f"UPDATE \"{new_table_name}\" SET \"{new_column_name}\"='{fake_city}' WHERE rowid=={row_id};"
                    if verbose:
                        print(f"{update_sql=}", file=sys.stderr)
                    self.cx.execute(update_sql)
                update_syn_cities = f"""
                    UPDATE "{self._tablename}"
                    SET "{new_column_name}"="{new_table_name}"."{new_column_name}"
                    FROM "{new_table_name}" 
                    WHERE "{self._tablename}"."{base_column}"=="{new_table_name}"."{base_column}"
                """
                add_column_sql = f'ALTER TABLE "{self._tablename}" ADD COLUMN "{new_column_name}" VARCHAR;'
                if verbose:
                    print(f"{add_column_sql=}", file=sys.stderr)
                    print(f"{update_syn_cities=}", file=sys.stderr)
                self.cx.sql(add_column_sql)
                self.data = self.cx.table(self._tablename)
                self.cx.execute(update_syn_cities)
                self.data = self.cx.table(self._tablename)
                self.city_equivalences = self.cx.table(new_table_name)
            except Exception as e:
                printf(f"{e=}", file=sys.stderr)
                raise Exception
    return self
    
def syn_city_column(self,
                    base_column: str,
                    max_uniques: Optional[int] = 1000,
                    verbose: bool = False) -> 'CS':
        """
        Replaces a base_column with a new column containing synthetic generated town/city names.

        Args:
            base_column: The name of the column to be replaced.
            max_uniques: Maximum number of unique base_column values to save in an equivalence table.
            verbose: If True, print debug messages.

        Returns:
            self: The CS object with the 'base_column' replaced by synthetic city values.
        """
        new_syn_column_name = f"syn_{base_column}"

        if verbose:
            print(f"syn_city_column: Replacing '{base_column}' with synthetic cities.", file=sys.stderr)

        # Add a new column 'syn_{base_column}' using add_syn_city_column
        try:
            self.add_syn_city_column(
                base_column=base_column,
                new_column_name=new_syn_column_name,
                max_uniques=max_uniques,
                verbose=verbose
            )
            if verbose:
                print(f"syn_city_column: Successfully added '{new_syn_column_name}'.", file=sys.stderr)
        except Exception as e:
            print(f"syn_city_column: Error adding synthetic column: {e}", file=sys.stderr)
            raise

        # Replace the original base_column with the new 'syn_{base_column}'
        try:
            self.replace_column(base_column, new_syn_column_name)
            if verbose:
                print(f"syn_city_column: Successfully replaced '{base_column}' with '{new_syn_column_name}'.", file=sys.stderr)
        except Exception as e:
            print(f"syn_city_column: Error replacing column: {e}", file=sys.stderr)
            raise

        # Remove the column that was added with .add_syn_city_column.
        try:
            self.drop_column(new_syn_column_name)
            if verbose:
                print(f"syn_city_column: Successfully dropped temporary column '{new_syn_column_name}'.", file=sys.stderr)
        except Exception as e:
            print(f"syn_city_column: Error dropping temporary column '{new_syn_column_name}': {e}", file=sys.stderr)
            raise

        return self