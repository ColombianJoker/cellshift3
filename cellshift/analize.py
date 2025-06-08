import duckdb
from duckdb import DuckDBPyConnection
import sys
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING
from . import CS  # Import CS from the main module to be able to return self with typing

def sql(self,
        sql_sentence: str,
        table_name: str = 'TABLE',
        in_place: bool = True,
        verbose: bool = False) -> Union[CS,duckdb.duckdb.DuckDBPyRelation]:
    """
    Runs a given SQL query on the current .data member and updates .data
    with the result of the query. Uses the name in 'table_name' for .data member (defaults to 'TABLE')

    This method assumes the SQL query is a SELECT statement that produces
    a result table. If the query is a DDL/DML statement (e.g., INSERT, UPDATE, DELETE, CREATE),
    it will execute it but will not update .data from its result,
    unless the DDL/DML implicitly modifies the table behind .data
    (e.g., INSERT/UPDATE/DELETE on self._original_tablename).

    For SELECT statements, the result replaces the current .data.
    For other statements, the .data relation is refreshed from the underlying table.

    Args:
        sql_sentence: The SQL query string to execute.
        table_name: the name used for the .data member in SQL sentences
        in_place: if True, updates data member if the SQL sentence is a valid result set
        verbose: If True, print debug information.

    Returns:
        self: The CS object with the .data member updated according to the SQL query (if in_place=True) OR
              A new DuckDB relation with the SQL result set (if in_place=False)
    """
    if verbose:
        print(f"{sql_sentence=}", sys.stderr)
        print(f"{table_name=}", sys.stderr)
        
    if not isinstance(sql_sentence, str):
        raise TypeError("sql: The 'sql_sentence' argument must be a string.")

    if self.data is None:
        if verbose:
            print("sql: No data loaded in .data. The SQL query will run against an empty context.", file=sys.stderr)
        # Proceed anyway, as the SQL might be creating a table or loading data
        # self.data will remain None or be updated if the query is a CREATE TABLE etc.

    try:
        if verbose:
            print(f"sql: Executing SQL query:\n{sql_sentence}", file=sys.stderr)
            initial_row_count = self.data.shape[0] if self.data else 0

        sql_sentence = sql_sentence.replace(table_name, self._original_tablename)

        # Identify if it's a SELECT query to handle result differently
        is_select_query = sql_sentence.strip().upper().startswith('SELECT')

        if is_select_query:
            # For SELECT queries, we get a new relation.
            # The user's query implicitly operates on self._original_tablename
            # or expects to operate on some context.
            # If the user's SQL is `SELECT * FROM my_table WHERE ...` where `my_table`
            # is `self._original_tablename`, then we can run it.
            # If it's a completely different SELECT, the result will replace self.data.

            # Best approach for SELECT: Let the user provide the full SELECT query.
            # The result of this query will then be materialized into our internal table.

            # Check if the query refers to the current table.
            # This is a heuristic and might not cover all cases.
            # A more robust solution might involve parsing the SQL, but that's complex.
            # For simplicity, assume the user's SQL string is a valid query.

            # Execute the query and get the result as a relation
            #
            # self.cx.register(table_name, self.data)
            if verbose:
                print(f"sql: Executing changed SQL query:\n{sql_sentence}", file=sys.stderr)
            new_data = self.cx.sql(sql_sentence)
            # self.cx.unregister(table_name)

            if new_data is None:
                # This might happen if the SQL was not a SELECT, but .sql() sometimes returns None.
                # Or if the query was empty.
                raise ValueError("sql: The provided SQL query did not return a valid relation.")

            # Materialize the result back into the original table name
            # This ensures self.data always points to the named table.
            # new_data.create_table(self._original_tablename, overwrite=True)
            # self.data = self.cx.table(self._original_tablename)
            if verbose:
                final_row_count = self.data.shape[0]
                if in_place:
                    print(f"sql: SELECT query executed. Data updated. New row count: {final_row_count}", file=sys.stderr)
                else:
                    print(f"sql: SELECT query executed. Data returned. Returned row count: {final_row_count}", file=sys.stderr)

            if in_place:
                self.data = new_data
            else:
                return new_data

        else:
            # For DDL/DML statements (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, etc.)
            # We execute directly. The changes should reflect in the underlying table
            # that self.data points to.
            self.cx.execute(sql_sentence)

            # After DDL/DML, refresh the self.data relation from the table
            # to ensure it reflects any changes made by the DML/DDL.
            self.data = self.cx.table(self._original_tablename)

            if verbose:
                final_row_count = self.data.shape[0]
                print(f"run_sql: Non-SELECT SQL query executed. Data refreshed. Current row count: {final_row_count}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"run_sql: An error occurred during SQL execution: {e}", file=sys.stderr)
        raise ValueError(f"Failed to execute SQL query: {e}")
    return self

def groups(self,
           base_column: Optional[Union[str, List[str]]] = None,
           name_prefix: str = 'Group_',
           group_column_name: str = 'Group_Name',
           count_column_name: str = 'Count',
           group_data_column_name: str = 'Group_Data',
           order_by: str = 'ASC',
           count_filter: str = '? > 0',
           meta: str = '?',
           limit: Optional[int] = None,
           verbose: bool = False) -> duckdb.DuckDBPyRelation:
    """
    Creates a DuckDB table relation by grouping data based on the specified columns.

    Args:
        base_column: The column(s) to group by and include in Group_Data.
                     If not given, all columns from the table are used.
        name_prefix : Prefix for the group names.
        group_column_name : Name of the column for group names.
        count_column_name : Name of the column for the count of each group.
        group_data_column_name : Name of the column for the list of group data, copied from source
                                 This is (the grouping key).
        order_by : Order of the groups ('ASC' or 'DESC') based on count_column_name.
        count_filter: A SQL condition string to filter groups by their count.
                      The 'meta' character (default '?') will be replaced by the count_column_name.
                      Defaults to '? > 0'.
        meta: The placeholder character in 'count_filter' that will be replaced by the actual count column name.
              Defaults to '?'.
        limit: Limits the number of groups returned.
        verbose: If True, shows debug info.

    Returns:
        duckdb.DuckDBPyRelation: A new DuckDB table relation with three columns:
                                 - Group_Name: 'Group_{unique_group_id}'
                                 - Count: Count of rows in each group.
                                 - Group_Data: A list of contents from the given base_column(s) for the group.
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
    sql_query += f" ORDER BY \"{count_column_name}\", CAST(SUBSTR(\"{group_column_name}\",LENGTH('{name_prefix}')+1) AS INTEGER) {order_by_upper}"

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
