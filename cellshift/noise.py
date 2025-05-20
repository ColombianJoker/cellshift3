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
                             impulse_mag: Optional[Union[int, float]] = None, verbose: bool = False) -> CS:
    """
    Adds a new column containing impulse noise to the CS object's data,
    basing its values from a preexisting one. The base column must be numeric,
    and the new column is always of float type.

    Args:
        base_column: The name of the column to use for calculating statistics.
        new_column_name: The name of the new column. If None, defaults to
                         "noise_{base_column}".
        sample_pct: Percent of column values to be altered by noise (0 < sample_pct < 100).
        n_samples: Absolute number of column values to be altered by noise (positive integer).
        impulse_mag: Max absolute magnitude of impulse noise to be applied (non-negative).
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the added impulse noise column.
    """
    print(f"1: add_impulse_noise_column: Start, base_column={base_column}, new_column_name={new_column_name}", file=sys.stderr)
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

    if impulse_mag is None:
        raise ValueError("'impulse_mag' must be provided for impulse noise.")
    if impulse_mag < 0:
        raise ValueError("'impulse_mag' must be non-negative.")
    max_impulse_val = float(impulse_mag)

    # Get total rows
    count_sql = f"SELECT COUNT(*) FROM \"{self._tablename}\"" # Use _tablename as it's the current state
    total_rows = self.cx.execute(count_sql).fetchone()[0]
    if total_rows == 0:
        print("add_impulse_noise_column: No rows in base table, skipping noise addition.", file=sys.stderr)
        return self

    num_samples_to_alter = 0
    if n_samples is not None:
        num_samples_to_alter = n_samples
    elif sample_pct is not None:
        num_samples_to_alter = max(1, int((sample_pct * total_rows) / 100))

    if num_samples_to_alter == 0:
        print("add_impulse_noise_column: No samples to alter, skipping noise addition.", file=sys.stderr)
        return self

    try:
        # 1. Add the new column to the table with a DOUBLE type
        print(f"2: add_impulse_noise_column: Adding column '{new_column_name}' as DOUBLE.", file=sys.stderr)
        self.cx.execute(f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" DOUBLE;")
        
        # 2. Initialize the new column with values from the base_column
        print(f"3: add_impulse_noise_column: Initializing '{new_column_name}' with '{base_column}' values.", file=sys.stderr)
        self.cx.execute(f"UPDATE \"{self._tablename}\" SET \"{new_column_name}\" = \"{base_column}\";")

        # Update self.data to reflect the new schema
        self.data = self.cx.table(self._tablename)

        # 3. Generate random row IDs and noise values
        random_row_ids = np.random.choice(total_rows, size=num_samples_to_alter, replace=False)
        noise_amounts = np.random.uniform(-max_impulse_val, max_impulse_val, num_samples_to_alter)
        
        print(f"4: add_impulse_noise_column: Applying noise to {num_samples_to_alter} samples.", file=sys.stderr)
        # 4. Apply impulse noise via individual UPDATE statements
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
        
        # 5. Update self.data one last time after all updates are done
        self.data = self.cx.table(self._tablename)
        print(f"6: add_impulse_noise_column: Column '{new_column_name}' updated with impulse noise.", file=sys.stderr)

        return self

    except Exception as e:
        print(f"Error in add_impulse_noise_column: {e}", file=sys.stderr)
        raise e  # Re-raise the exception to propagate it.

    finally:
        print("7: add_impulse_noise_column: End", file=sys.stderr)
    return self