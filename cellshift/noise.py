import duckdb
import numpy as np
import pyarrow as pa
# import pandas as pd
# import polars as pl
from tqdm import tqdm
from typing import Generator, Union, Optional
import sys
from . import CS  # Import CS from the main module to be able to return self with typing

def add_gaussian_noise_column(self, base_column: str, new_column_name: str = None) -> CS:
    """
    Adds a new column containing Gaussian noise to the CS object's data,
    based on the statistics of an existing column.

    Args:
        base_column: The name of the column to use for calculating the
                     mean and standard deviation of the noise.
        new_column_name: The name of the new column.  If None, defaults to
                         "noise_{base_column}".

    Returns:
        self: The CS object with the added Gaussian noise column.
    """
    # print(f"1: add_gaussian_noise_column: Start, base_column={base_column}, new_column_name={new_column_name}", file=sys.stderr)
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")

    # Validate base column name
    valid_columns = [col.lower() for col in self.data.columns]
    if base_column.lower() not in valid_columns:
        raise ValueError(f"Column '{base_column}' not found in the data.")

    if new_column_name is None:
        new_column_name = f"noise_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")

    # Construct the SQL query to calculate mean and standard deviation.
    stats_sql = f""" SELECT COUNT("{base_column}") as count, AVG("{base_column}") as mean, STDDEV_POP("{base_column}") as stddev
                     FROM "{self._original_tablename}"
                 """
    # print(f"2: add_gaussian_noise_column: Stats SQL: {stats_sql}", file=sys.stderr)

    try:
        # Execute the query and fetch the results.
        stats_result = self.cx.execute(stats_sql).fetchone()
        if stats_result is None:
            raise ValueError(f"Could not calculate statistics for column '{base_column}'.")
        count, mean, stddev = stats_result

        # Generate the Gaussian noise using numpy
        noise_values = np.random.normal(mean, stddev, int(count))

        # Create an Arrow array
        noise_array = pa.array(noise_values)
        
        # create a table
        noise_table = pa.table({new_column_name: noise_array})

        # Add the noise column using the existing add_column method.
        self.add_column(self.cx.from_arrow(noise_table), new_column_name)
        # return self

    except Exception as e:
        print(f"Error in add_gaussian_noise_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
        # print("3: add_gaussian_noise_column: End", file=sys.stderr)
    return self

def add_impulse_noise_column(self, base_column: str, new_column_name: Optional[str] = None,
                             sample_pct: Optional[float] = None, n_samples: Optional[int] = None,
                             impulse_mag: Optional[Union[int, float]] = None, impulse_pct: Optional[float] = None,
                             verbose: bool = False) -> CS:
    """
    Adds a new column containing impulse noise to the CS object's data,
    basing its values from a preexisting one. The base column must be numeric,
    and the new column is always of float type.

    Args:
        base_column: The name of the column to use for calculating statistics.
        new_column_name: The name of the new column. If None, defaults to "noise_{base_column}".
        sample_pct: Percent of column values to be altered by noise (0 < sample_pct < 100).
        n_samples: Absolute number of column values to be altered by noise (positive integer).
        impulse_mag: Max absolute magnitude of impulse noise to be applied (non-negative).
        impulse_pct: Percent of the maximum absolute value in the base column to be applied as
                     impulse noise (0 < impulse_pct <= 100).
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
        new_column_name = f"impulse_noise_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate sample/impulse parameters
    if not (sample_pct is not None or n_samples is not None):
        raise ValueError("Either 'sample_pct' or 'n_samples' must be provided.")
    if sample_pct is not None and not (0 < sample_pct < 100):
        raise ValueError("'sample_pct' must be between 0 and 100 (exclusive).")
    if n_samples is not None and n_samples <= 0:
        raise ValueError("'n_samples' must be a positive integer.")

    # Validate impulse magnitude/percentage
    if impulse_mag is None and impulse_pct is None:
        raise ValueError("Either 'impulse_mag' or 'impulse_pct' must be provided.")
    if impulse_mag is not None and impulse_mag < 0:
        raise ValueError("'impulse_mag' must be non-negative.")
    if impulse_pct is not None and not (0 < impulse_pct <= 100):
        raise ValueError("'impulse_pct' must be between 0 and 100 (inclusive).")

    # Calculate max_impulse_val based on provided arguments
    max_impulse_val = 0.0
    if impulse_mag is not None:
        max_impulse_val = float(impulse_mag)
    elif impulse_pct is not None:
        # Calculate max absolute value of the base column
        max_abs_sql = f"SELECT MAX(ABS(\"{base_column}\")) FROM \"{self._tablename}\""
        max_abs_val = self.cx.execute(max_abs_sql).fetchone()[0]
        if max_abs_val is None:
            raise ValueError(f"Could not calculate MAX(ABS()) for column '{base_column}'. Is it all NULLs or non-numeric?")
        max_impulse_val = (max_abs_val * impulse_pct) / 100.0
    # If both are provided, impulse_mag takes precedence (already handled by the if/elif structure)

    # Get total rows
    count_sql = f"SELECT COUNT(*) FROM \"{self._tablename}\"" # Use _tablename as it's the current state
    total_rows = self.cx.execute(count_sql).fetchone()[0]
    if total_rows == 0:
        # print("add_impulse_noise_column: No rows in base table, skipping noise addition.", file=sys.stderr)
        return self

    num_samples_to_alter = 0
    if n_samples is not None:
        num_samples_to_alter = n_samples
    elif sample_pct is not None:
        num_samples_to_alter = max(1, int((sample_pct * total_rows) / 100))

    if num_samples_to_alter == 0:
        # print("add_impulse_noise_column: No samples to alter, skipping noise addition.", file=sys.stderr)
        return self

    try:
        # 1. Add the new column to the table with a DOUBLE type
        # print(f"add_impulse_noise_column: Adding column '{new_column_name}' as DOUBLE.", file=sys.stderr)
        self.cx.execute(f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" DOUBLE;")
        
        # 2. Initialize the new column with values from the base_column
        # print(f"add_impulse_noise_column: Initializing '{new_column_name}' with '{base_column}' values.", file=sys.stderr)
        self.cx.execute(f"UPDATE \"{self._tablename}\" SET \"{new_column_name}\" = \"{base_column}\";")

        # Update self.data to reflect the new schema
        self.data = self.cx.table(self._tablename)

        # 3. Generate random row IDs and noise values
        random_row_ids = np.random.choice(total_rows, size=num_samples_to_alter, replace=False)
        noise_amounts = np.random.uniform(-max_impulse_val, max_impulse_val, num_samples_to_alter)
        
        # print(f"add_impulse_noise_column: Applying noise to {num_samples_to_alter} samples.", file=sys.stderr)
        # Apply impulse noise via individual UPDATE statements
        # Use tqdm for progress if verbose is True
        for i in tqdm(range(num_samples_to_alter), disable=not verbose, desc="Applying impulse noise"):
            row_id = random_row_ids[i]
            noise_amount = noise_amounts[i]
            update_sql = f"""
                UPDATE \"{self._tablename}\"
                SET \"{new_column_name}\" = \"{new_column_name}\" + {noise_amount}
                WHERE rowid = {row_id};
            """
            self.cx.execute(update_sql)
        
        self.data = self.cx.table(self._tablename)
        # print(f"add_impulse_noise_column: Column '{new_column_name}' updated with impulse noise.", file=sys.stderr)

        return self

    except Exception as e:
        print(f"Error in add_impulse_noise_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
        # print("7: add_impulse_noise_column: End", file=sys.stderr)
    return self
    
def add_salt_pepper_noise_column(self, base_column: str, new_column_name: Optional[str] = None,
                                 sample_pct: Optional[float] = None, n_samples: Optional[int] = None,
                                 verbose: bool = False) -> CS:
    """
    Adds a new column containing salt-and-pepper noise to the CS object's data.
    Selected values in the new column will be set to the min or max of the base column.

    Args:
        base_column: The name of the column to use for calculating min/max values.
        new_column_name: The name of the new column. If None, defaults to "salt_pepper_noise_{base_column}".
        sample_pct: Percent of column values to be altered by noise (0 < sample_pct < 100).
        n_samples: Absolute number of column values to be altered by noise (positive integer).
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
        new_column_name = f"salt_pepper_noise_{base_column}"

    if not isinstance(new_column_name, str):
        raise TypeError("new_column_name must be a string")
    if not new_column_name.isidentifier():
        raise ValueError("new_column_name must be a valid identifier (e.g., no spaces or special characters).")

    # Validate sample parameters
    if not (sample_pct is not None or n_samples is not None):
        raise ValueError("Either 'sample_pct' or 'n_samples' must be provided.")
    if sample_pct is not None and not (0 < sample_pct < 100):
        raise ValueError("'sample_pct' must be between 0 and 100 (exclusive).")
    if n_samples is not None and n_samples <= 0:
        raise ValueError("'n_samples' must be a positive integer.")

    # Get total rows
    count_sql = f"SELECT COUNT(*) FROM \"{self._tablename}\"" # Use _tablename as it's the current state
    total_rows = self.cx.execute(count_sql).fetchone()[0]
    if total_rows == 0:
        # print("add_salt_pepper_noise_column: No rows in base table, skipping noise addition.", file=sys.stderr)
        return self

    # Calculate n_samples to alter
    num_samples_to_alter = 0
    if n_samples is not None:
        if n_samples > total_rows:
            raise ValueError(f"'n_samples' ({n_samples}) cannot be greater than total rows ({total_rows}).")
        num_samples_to_alter = n_samples
    elif sample_pct is not None:
        num_samples_to_alter = max(1, int((sample_pct * total_rows) / 100))

    if num_samples_to_alter == 0:
        # print("add_salt_pepper_noise_column: No samples to alter, skipping noise addition.", file=sys.stderr)
        return self

    # Get min and max values of the base column
    min_max_sql = f"SELECT MIN(\"{base_column}\"), MAX(\"{base_column}\") FROM \"{self._tablename}\""
    min_val, max_val = self.cx.execute(min_max_sql).fetchone()
    
    if min_val is None or max_val is None:
        raise ValueError(f"Could not calculate MIN/MAX for column '{base_column}'. Is it all NULLs or non-numeric?")
    
    # Ensure min_val and max_val are floats for consistency with noise
    min_val = float(min_val)
    max_val = float(max_val)

    try:
        # Add the new column to the table with a DOUBLE type
        # print(f"add_salt_pepper_noise_column: Adding column '{new_column_name}' as DOUBLE.", file=sys.stderr)
        self.cx.execute(f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" DOUBLE;")
        
        # Initialize the new column with values from the base_column
        # print(f"add_salt_pepper_noise_column: Initializing '{new_column_name}' with '{base_column}' values.", file=sys.stderr)
        self.cx.execute(f"UPDATE \"{self._tablename}\" SET \"{new_column_name}\" = \"{base_column}\";")

        # Update self.data to reflect the new schema
        self.data = self.cx.table(self._tablename)

        # Generate random row IDs for alteration
        random_row_ids = np.random.choice(total_rows, size=num_samples_to_alter, replace=False)
        # print(f"add_salt_pepper_noise_column: Applying salt-and-pepper noise to {num_samples_to_alter} samples.", file=sys.stderr)
        # Apply salt-and-pepper noise via individual UPDATE statements
        # Use tqdm for progress if verbose is True
        for i in tqdm(range(num_samples_to_alter), disable=not verbose, desc="Applying salt-and-pepper noise"):
            row_id = random_row_ids[i]
            # Randomly choose between min_val (pepper) and max_val (salt)
            noise_val = np.random.choice([min_val, max_val])
            
            update_sql = f"""
                UPDATE \"{self._tablename}\"
                SET \"{new_column_name}\" = {noise_val}
                WHERE rowid = {row_id};
            """
            self.cx.execute(update_sql)
        
        # 5. Update self.data one last time after all updates are done
        self.data = self.cx.table(self._tablename)
        # print(f"add_salt_pepper_noise_column: Column '{new_column_name}' updated with salt-and-pepper noise.", file=sys.stderr)

        return self

    except Exception as e:
        print(f"Error in add_salt_pepper_noise_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        pass
        # print("add_salt_pepper_noise_column: End", file=sys.stderr)
        
    return self

def gaussian_column(self, column_name: str, verbose: bool = False) -> CS:
    """
    Applies Gaussian noise to a specified column, replacing its original values.

    Args:
        column_name: The name of the column to apply noise to and replace.
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    # Validate column_name exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if column_name.lower() not in valid_columns:
        raise ValueError(f"Column '{column_name}' not found in the data.")

    # Define the name for the temporary noise column
    temp_noise_column_name = f"noised_{column_name}"

    try:
        # Create a new column with Gaussian noise
        # Call add_gaussian_noise_column with only the arguments it expects
        self.add_gaussian_noise_column(
            base_column=column_name,
            new_column_name=temp_noise_column_name,
        )
        if verbose:
            print(f"gaussian_column: Temporary noise column '{temp_noise_column_name}' added.", file=sys.stderr)

        # Replace the original column with the noise column
        self.replace_column(
            column_to_replace=column_name,
            replace_column=temp_noise_column_name
        )
        if verbose:
            print(f"gaussian_column: Column '{column_name}' replaced.", file=sys.stderr)

        # Remove the temporary noise column
        self.drop_column(temp_noise_column_name)
        if verbose:
          print(f"gaussian_column: Temporary noise column '{temp_noise_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        print(f"Error in gaussian_column for column '{column_name}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # Check if the temporary column exists before trying to drop it
            if temp_noise_column_name.lower() in valid_columns: # Use valid_columns from earlier for robustness
                self.drop_column(temp_noise_column_name)
                if verbose:
                  print(f"Cleaned up temporary column '{temp_noise_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            print(f"Error during cleanup of temporary column '{temp_noise_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        pass
        # print(f"gaussian_column: End for column '{column_name}'.", file=sys.stderr)
    return self
    
def impulse_column(self, column_name: str,
                   sample_pct: Optional[float] = None, n_samples: Optional[int] = None,
                   impulse_mag: Optional[Union[int, float]] = None, impulse_pct: Optional[float] = None,
                   verbose: bool = False) -> CS:
    """
    Applies impulse noise to a specified column, replacing its original values.

    Args:
        column_name: The name of the column to apply noise to and replace.
        sample_pct: Percent of column values to be altered by noise (0 < sample_pct < 100).
        n_samples: Absolute number of column values to be altered by noise (positive integer).
        impulse_mag: Max absolute magnitude of impulse noise to be applied (non-negative).
        impulse_pct: Percent of the maximum absolute value in the base column to be applied as impulse noise (0 < impulse_pct <= 100).
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    if verbose:
        print(f"1: impulse_column: Start for column '{column_name}'", file=sys.stderr)
    
    # Validate column_name exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if column_name.lower() not in valid_columns:
        raise ValueError(f"Column '{column_name}' not found in the data.")

    # Define the name for the temporary noise column
    temp_noise_column_name = f"noised_impulse_{column_name}"

    try:
        # Step 1: Create a new column with impulse noise
        if verbose:
            print(f"2: impulse_column: Adding temporary impulse noise column '{temp_noise_column_name}'.", file=sys.stderr)
        self.add_impulse_noise_column(
            base_column=column_name,
            new_column_name=temp_noise_column_name,
            sample_pct=sample_pct,
            n_samples=n_samples,
            impulse_mag=impulse_mag,
            impulse_pct=impulse_pct, # Pass impulse_pct
            verbose=verbose
        )
        if verbose:
            print(f"3: impulse_column: Temporary impulse noise column '{temp_noise_column_name}' added.", file=sys.stderr)

        # Step 2: Replace the original column with the noise column
        if verbose:
            print(f"4: impulse_column: Replacing '{column_name}' with '{temp_noise_column_name}'.", file=sys.stderr)
        self.replace_column(
            column_to_replace=column_name,
            replace_column=temp_noise_column_name
        )
        if verbose:
            print(f"5: impulse_column: Column '{column_name}' replaced.", file=sys.stderr)

        # Step 3: Remove the temporary noise column
        if verbose:
            print(f"6: impulse_column: Dropping temporary impulse noise column '{temp_noise_column_name}'.", file=sys.stderr)
        self.drop_column(temp_noise_column_name)
        if verbose:
            print(f"7: impulse_column: Temporary impulse noise column '{temp_noise_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in impulse_column for column '{column_name}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # Check if the temporary column exists before trying to drop it
            # Note: `drop_column` handles non-existent columns gracefully, so a direct call is fine.
            self.drop_column(temp_noise_column_name)
            if verbose:
                print(f"Cleaned up temporary column '{temp_noise_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            if verbose:
                print(f"Error during cleanup of temporary column '{temp_noise_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        if verbose:
            print(f"8: impulse_column: End for column '{column_name}'.", file=sys.stderr)
    return self

def salt_pepper_column(self, column_name: str,
                       sample_pct: Optional[float] = None, n_samples: Optional[int] = None,
                       verbose: bool = False) -> CS:
    """
    Applies salt-and-pepper noise to a specified column, replacing its original values.

    Args:
        column_name: The name of the column to apply noise to and replace.
        sample_pct: Percent of column values to be altered by noise (0 < sample_pct < 100).
        n_samples: Absolute number of column values to be altered by noise (positive integer).
        verbose: If True, print debug information.

    Returns:
        a new version of the CS object
    """
    # Validate column_name exists
    if self.data is None:
        raise ValueError("No data loaded in the CS object.")
    valid_columns = [col.lower() for col in self.data.columns]
    if column_name.lower() not in valid_columns:
        raise ValueError(f"Column '{column_name}' not found in the data.")

    # Define the name for the temporary noise column
    temp_noise_column_name = f"noised_salt_pepper_{column_name}"

    try:
        # Create a new column with salt-and-pepper noise
        self.add_salt_pepper_noise_column(
            base_column=column_name,
            new_column_name=temp_noise_column_name,
            sample_pct=sample_pct,
            n_samples=n_samples,
            verbose=verbose
        )
        if verbose:
            print(f"salt_pepper_column: Temporary salt-and-pepper noise column '{temp_noise_column_name}' added.", file=sys.stderr)

        # Replace the original column with the noise column
        self.replace_column(
            column_to_replace=column_name,
            replace_column=temp_noise_column_name
        )
        if verbose:
            print(f"salt_pepper_column: Column '{column_name}' replaced.", file=sys.stderr)

        # Remove the temporary noise column
        self.drop_column(temp_noise_column_name)
        if verbose:
            print(f"salt_pepper_column: Temporary noise column '{temp_noise_column_name}' dropped.", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"Error in salt_pepper_column for column '{column_name}': {e}", file=sys.stderr)
        # Attempt to clean up the temporary column if an error occurred before dropping it
        try:
            # drop_column handles non-existent columns gracefully, so a direct call is fine.
            self.drop_column(temp_noise_column_name)
            if verbose:
                print(f"Cleaned up temporary column '{temp_noise_column_name}' due to error.", file=sys.stderr)
        except Exception as cleanup_e:
            if verbose:
                print(f"Error during cleanup of temporary column '{temp_noise_column_name}': {cleanup_e}", file=sys.stderr)
        raise e # Re-raise the original exception

    finally:
        pass
    return self