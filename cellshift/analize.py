import duckdb
from duckdb import DuckDBPyConnection
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING
from . import CS  # Import CS from the main module to be able to return self with typing

def groups(self,
           base_column: Optional[Union[str, List[str]]] = None,
           name_prefix: str = 'Group_',
           group_column_name: str = 'Group_Name',
           count_column_name: str = 'Count',
           group_data_column_name: str = 'Group_Data',
           limit: Optional[int] = None,
           order_by: str = 'ASC',
           verbose: bool = False,
           count_filter: str = '? > 0',
           meta: str = '?') -> duckdb.DuckDBPyRelation:
    """
    Creates a DuckDB table relation by grouping data based on the specified columns.

    Args:
        base_column: The column(s) to group by and include in Group_Data.
                     If not given, all columns from the table are used.
        name_prefix : Prefix for the group names.
        group_column_name : Name of the column for group names.
        count_column_name : Name of the column for the count of each group.
        group_data_column_name : Name of the column for the list of group data (the grouping key).
        limit: Limits the number of groups returned.
        order_by : Order of the groups ('ASC' or 'DESC') based on count_column_name.
        verbose: If True, shows debug info.
        count_filter: A SQL condition string to filter groups by their count.
                      The 'meta' character (default '?') will be replaced by the count_column_name.
                      Defaults to '? > 0'.
        meta: The placeholder character in 'count_filter' that will be replaced by the actual count column name.
              Defaults to '?'.

    Returns:
        duckdb.DuckDBPyRelation: A new DuckDB table relation with three columns:
                                 - Group_Name: 'Group_{unique_group_id}'
                                 - Count: Count of rows in each group.
                                 - Group_Data: A list of contents from the given base_column(s) for the group.
    Raises:
        ValueError: If input arguments are invalid or data is not loaded.
        duckdb.Error: If a DuckDB query fails.
    """
    if self.data is None:
        raise ValueError("groups: No data loaded in the CS object. Cannot perform groupings.")

    # Determine columns to group by and select for LIST_VALUE
    if base_column is None:
        # Access column names from the DuckDBPyRelation object directly
        column_names = self.data.columns
        select_columns_for_array = ", ".join([f"CAST(\"{col}\" AS VARCHAR)" for col in column_names])
        group_by_columns = ", ".join([f"\"{col}\"" for col in column_names])
    elif isinstance(base_column, str):
        # Validate base_column exists
        if base_column not in self.data.columns:
            raise ValueError(f"groups: Column '{base_column}' not found in the data.")
        select_columns_for_array = f"CAST(\"{base_column}\" AS VARCHAR)"
        group_by_columns = f"\"{base_column}\""
    elif isinstance(base_column, list):
        # Validate all base_columns exist
        data_column_names = self.data.columns
        for col in base_column:
            if col not in data_column_names:
                raise ValueError(f"groups: Column '{col}' not found in the data.")
        select_columns_for_array = ", ".join([f"CAST(\"{col}\" AS VARCHAR)" for col in base_column])
        group_by_columns = ", ".join([f"\"{col}\"" for col in base_column])
    else:
        raise ValueError("groups: base_column must be a string, a list of strings, or None.")

    # Validate order_by argument
    order_by_upper = order_by.upper()
    if order_by_upper not in ['ASC', 'DESC']:
        raise ValueError("groups: order_by must be 'ASC' or 'DESC'")

    # Construct the HAVING clause if count_filter is provided and not empty
    having_clause = ""
    if count_filter is not None and count_filter.strip() != "":
        # Replace the 'meta' character with the actual count column name from the subquery's scope
        resolved_count_column_for_filter = f'"{count_column_name}"'
        having_condition = count_filter.replace(meta, resolved_count_column_for_filter)
        having_clause = f"HAVING {having_condition}"
        if verbose:
            print(f"groups: Applying HAVING clause: {having_clause}", file=sys.stderr)

    # Construct the main SQL query
    # The HAVING clause is placed inside the subquery to filter groups based on count
    # before the final selection and row numbering.
    sql_query = f"""
    SELECT
        '{name_prefix}' || (ROW_NUMBER() OVER (ORDER BY (SELECT NULL))) AS "{group_column_name}",
        temp_t."{count_column_name}",
        temp_t."{group_data_column_name}"
    FROM (
        SELECT
            LIST_VALUE({select_columns_for_array}) AS "{group_data_column_name}",
            COUNT(*) AS "{count_column_name}"
        FROM "{self._tablename}"
        GROUP BY {group_by_columns}
        {having_clause}
    ) AS temp_t
    """

    # Add ORDER BY clause for the final result set
    # The order_by now explicitly refers to the final 'Count' column name.
    sql_query += f' ORDER BY "{count_column_name}" {order_by_upper}'

    # Add LIMIT clause if provided
    if limit is not None:
        if not isinstance(limit, int) or (limit <= 0):
            raise ValueError("groups: limit must be a positive integer or None.")
        sql_query += f" LIMIT {limit}"

    # Add final semicolon
    sql_query += ";"

    if verbose:
        print(f"groups: Generated SQL query:\n{sql_query}", file=sys.stderr)

    # Execute the query and return the DuckDBPyRelation
    try:
        return self.cx.query(sql_query)
    except duckdb.Error as e:
        print(f"Error executing groups query: {e}", file=sys.stderr)
        raise
