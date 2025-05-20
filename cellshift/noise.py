import duckdb as d
import numpy as np
import pyarrow as pa
# import pandas as pd
# import polars as pl
from typing import Generator, Union, Optional
from . import CS  # Import CS from the main module to be able to return self with typing
import sys

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