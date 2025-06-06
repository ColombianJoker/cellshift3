import duckdb
from duckdb import DuckDBPyConnection
from os import urandom
import pandas as pd
import polars as pl
import pyarrow as pa
import sys
import tempfile
from typing import Any, List, Dict, Union, Optional, TYPE_CHECKING
from . import CS  # Import CS from the main module to be able to return self with typing

def add_data(self,
             data: Union[str, List[str], pd.DataFrame, pl.DataFrame, pa.Table],
             verbose: bool = False) -> CS:
    """
    Adds new data to the existing .data member of the class.
    The new data is added using INSERT into current data.
    Column names and types must be compatible with existing data.

    Args:
        data: the data to add. Receives:
                    + a string (filename of a file to load)
                    + a list of strings (each a filename of a file to load)
                    + a Pandas DataFrame
                    + a Polars DataFrame
                    + a PyArrow Table
        verbose: If True, show debug messages

    Returns:
        a new version of the CS object
    """
    if verbose:
        print(f"add: Start, input_data type: {type(data)}", file=sys.stderr)
    if isinstance(data, str):
        sql_insert = sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{data}';"
        if verbose:
            print(f"add('{data}') (str)", file=sys.stderr)
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
        # self.data = self.cx.table(self._tablename)
    elif isinstance(data, list):
        for item in data:
            sql_insert = sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM '{item}';"
            if verbose:
                print(f"add('{item}') (str)", file=sys.stderr)
                print(sql_insert, file=sys.stderr)
            self.cx.execute(sql_insert)
        # self.data = self.cx.table(self._tablename)
    elif isinstance(data, pd.DataFrame):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, pl.DataFrame):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, pa.Table):
        sql_insert = f"INSERT INTO \"{self._tablename}\" SELECT * FROM data;"
        if verbose:
            print(sql_insert, file=sys.stderr)
        self.cx.execute(sql_insert)
    elif isinstance(data, duckdb.DuckDBPyRelation):
        print(f"Unsupported input type: {type(data)}", file=sys.stderr)
        print(f"Please re-try using data.arrow()", file=sys.stderr)
        return self
    else:
        print(f"Unsupported input type: {type(data)}", file=sys.stderr)
        return self
    self.data = self.cx.table(self._tablename)
    return self

def remove_rows(self,
                 base_column: Union[str, List[str]],
                 condition: str = '? IS NULL',
                 meta: str = '?',
                 all_and: bool = True,
                 verbose: bool = False) -> CS:
    """
    Removes rows from the .data member based on a custom SQL condition applied to specified columns,
            defaulting to removing rows where the column is NULL.

    Args:
        base_column: A single column name (string) or a list of column names
                     (strings) to apply the condition to.
        condition: A string representing the SQL condition template for each column.
                   It must contain the 'meta' string which will be replaced by the column name.
                   Defaults to '? IS NULL' to remove rows where the column is NULL.
        meta: The meta-character string within the 'condition' that will be replaced
              by the actual column name. Defaults to '?'.
        all_and: if True, applies the condition using AND to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) AND CONDITION(COLUMN2) ...
                 if False, applies the condition using OR to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) OR CONDITION(COLUMN2) ...
        verbose: If True, print debug information.

    Returns:
        self: The CS object with rows filtered based on the specified condition.
    """
    if self.data is None:
        if verbose:
            print("remove_rows: No data loaded in .data. Nothing to filter.", file=sys.stderr)
        return self

    # Normalize base_column to a list of strings
    columns_to_check: List[str] = []
    if isinstance(base_column, str):
        columns_to_check = [base_column]
    elif isinstance(base_column, list):
        if not all(isinstance(col, str) for col in base_column):
            raise TypeError("remove_rows: All items in 'base_column' list must be strings.")
        columns_to_check = base_column
    else:
        raise TypeError("remove_rows: 'base_column' must be a string or a list of strings.")

    if not columns_to_check:
        if verbose:
            print("remove_rows: No columns specified to check. No filtering applied.", file=sys.stderr)
        return self

    if not isinstance(condition, str) or not condition:
        raise TypeError("remove_rows: 'condition' must be a non-empty string.")
    if not isinstance(meta, str) or not meta:
        raise TypeError("remove_rows: 'meta' must be a non-empty string.")
    if meta not in condition:
        raise ValueError(f"remove_rows: 'meta' string '{meta}' not found in 'condition' string '{condition}'.")

    # Get existing column names from the data
    # Note: self.data.columns is already a list of strings (column names)
    existing_columns = self.data.columns
    if verbose:
        print(f"{self.data.columns=}\n", file=sys.stderr)

    # Validate if all specified columns exist
    for col in columns_to_check:
        if col not in existing_columns:
            raise ValueError(f"remove_rows: Column '{col}' not found in the data. Existing columns: {', '.join(existing_columns)}")

    try:
        # Build the WHERE clause using the custom condition and meta-character replacement
        # Example: for condition='? < 0' and meta='?',
        #          it becomes '(NOT ("col1" < 0)) AND (NOT ("col2" < 0))'
        conditions_for_delete = []
        for col in columns_to_check:
            # Replace the meta-character with the quoted column name
            sql_condition_for_col = condition.replace(meta, f"\"{col}\"")
            conditions_for_delete.append(f"({sql_condition_for_col})") # Wrap each condition in parentheses for safety
        if all_and:
          where_delete = " AND ".join(conditions_for_delete)
        else: # if not and then OR
          where_delete = " OR ".join(conditions_for_delete)

        if verbose:
            print(f"remove_rows: Removing rows where {where_delete}", file=sys.stderr)
            initial_row_count = self.data.shape[0]

        # Construct the SQL query to select filtered rows
        delete_sql = f'DELETE FROM "{self._tablename}" WHERE {where_delete}'

        if verbose:
            print(f"{delete_sql=}", file=sys.stderr)

        # Execute the query and fetch the result as an Arrow Table
        # new_data = self.cx.execute(filter_query_sql).fetch_arrow_table()
        new_data = self.cx.sql(delete_sql)
        self.data = self.cx.table(self._tablename)

        if verbose:
            print(f"{type(self.data)=}", file=sys.stderr)
            final_row_count = self.data.shape[0]
            rows_removed = initial_row_count - final_row_count
            print(f"remove_rows: Filtered successfully. {rows_removed} rows removed. New row count: {final_row_count}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"remove_rows: An error occurred during filtering: {e}", file=sys.stderr)
        # Re-raise the exception after printing verbose info
        raise ValueError(f"Failed to remove rows based on condition: {e}")
    return self

def filter_rows(self,
                 base_column: Union[str, List[str]],
                 condition: str = '? IS NOT NULL',
                 meta: str = '?',
                 all_and: bool = True,
                 verbose: bool = False) -> CS:
    """
    Filters rows from the .data member based on a custom SQL condition applied to specified columns,
        leaving only rows where the condition is met, defaulting to removing rows where the column is NULL.

    Args:
        base_column: A single column name (string) or a list of column names
                     (strings) to apply the condition to.
        condition: A string representing the SQL condition template for each column.
                   It must contain the 'meta' string which will be replaced by the column name.
                   Defaults to '? IS NULL' to remove rows where the column is NULL.
        meta: The meta-character string within the 'condition' that will be replaced
              by the actual column name. Defaults to '?'.
        all_and: if True, applies the condition using AND to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) AND CONDITION(COLUMN2) ...
                 if False, applies the condition using OR to all columns mentioned in base_column, like this:
                 SELECT * FROM TABLE WHERE CONDITION(COLUMN1) OR CONDITION(COLUMN2) ...
        verbose: If True, print debug information.

    Returns:
        self: The CS object with rows filtered based on the specified condition.
    """
    if self.data is None:
        if verbose:
            print("remove_rows: No data loaded in .data. Nothing to filter.", file=sys.stderr)
        return self

    # Normalize base_column to a list of strings
    columns_to_check: List[str] = []
    if isinstance(base_column, str):
        columns_to_check = [base_column]
    elif isinstance(base_column, list):
        if not all(isinstance(col, str) for col in base_column):
            raise TypeError("remove_rows: All items in 'base_column' list must be strings.")
        columns_to_check = base_column
    else:
        raise TypeError("remove_rows: 'base_column' must be a string or a list of strings.")

    if not columns_to_check:
        if verbose:
            print("remove_rows: No columns specified to check. No filtering applied.", file=sys.stderr)
        return self

    if not isinstance(condition, str) or not condition:
        raise TypeError("remove_rows: 'condition' must be a non-empty string.")
    if not isinstance(meta, str) or not meta:
        raise TypeError("remove_rows: 'meta' must be a non-empty string.")
    if meta not in condition:
        raise ValueError(f"remove_rows: 'meta' string '{meta}' not found in 'condition' string '{condition}'.")

    # Get existing column names from the data
    # Note: self.data.columns is already a list of strings (column names)
    existing_columns = self.data.columns
    if verbose:
        print(f"{self.data.columns=}\n", file=sys.stderr)

    # Validate if all specified columns exist
    for col in columns_to_check:
        if col not in existing_columns:
            raise ValueError(f"remove_rows: Column '{col}' not found in the data. Existing columns: {', '.join(existing_columns)}")

    try:
        # Build the WHERE clause using the custom condition and meta-character replacement
        # Example: for condition='? < 0' and meta='?',
        #          it becomes '(NOT ("col1" < 0)) AND (NOT ("col2" < 0))'
        conditions_for_sql = []
        for col in columns_to_check:
            # Replace the meta-character with the quoted column name
            sql_condition_for_col = condition.replace(meta, f"\"{col}\"")
            conditions_for_sql.append(f"({sql_condition_for_col})") # Wrap each condition in parentheses for safety

        if all_and:
          where_clause = " AND ".join(conditions_for_sql)
        else: # if not AND then OR
          where_clause = " OR ".join(conditions_for_sql)

        if verbose:
            print(f"remove_rows: Filtering rows where {where_clause}", file=sys.stderr)
            initial_row_count = self.data.shape[0]

        # Construct the SQL query to select filtered rows
        select_cols = ", ".join([f"\"{col}\"" for col in existing_columns])
        # filter_query_sql = f"SELECT {select_cols} FROM \"{self._original_tablename}\" WHERE {where_clause}"
        filter_query_sql = f'SELECT * FROM "{self._tablename}" WHERE {where_clause}'
        
        if verbose:
            print(f"{filter_query_sql=}", file=sys.stderr)

        # Execute the query and fetch the result as an Arrow Table
        # new_data = self.cx.execute(filter_query_sql).fetch_arrow_table()
        new_data = self.cx.sql(filter_query_sql)
        # Drop the existing table and create a new one with the replaced columns.
        # self.cx.execute(f"DROP TABLE IF EXISTS \"{self._tablename}\"")
        # self.cx.execute(f"CREATE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
        if (new_data) and isinstance(new_data, duckdb.duckdb.DuckDBPyRelation):
          self.cx.execute(f"CREATE OR REPLACE TABLE \"{self._tablename}\" AS SELECT * FROM new_data")
          self.data = self.cx.table(self._tablename)

        if verbose:
            final_row_count = self.data.shape[0]
            rows_removed = initial_row_count - final_row_count
            print(f"remove_rows: Filtered successfully. {rows_removed} rows removed. New row count: {final_row_count}", file=sys.stderr)

        return self

    except Exception as e:
        if verbose:
            print(f"remove_rows: An error occurred during filtering: {e}", file=sys.stderr)
        # Re-raise the exception after printing verbose info
        raise ValueError(f"Failed to remove rows based on condition: {e}")
    return self

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

        # It's safer to run the query directly, then determine how to update .data
        # DuckDB's .sql() method returns a relation for SELECT queries,
        # and None for DDL/DML queries.

        # data_table = self._original_tablename # Let the user refer to the main table by its actual name
                                                        # within the SQL string.
        sql_sentence = sql_sentence.replace(table_name, self._original_tablename)

        # Execute the SQL query
        # We use self.cx.execute for more control, especially for DDL/DML.
        # For SELECT, we can fetch the result.

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

def groupings(self,
              base_column: Optional[Union[str, List[str]]] = None,
              name_prefix: str = 'GROUP_',
              group_column_name: str = 'Group_Name',
              count_column_name: str = 'Count',
              limit: Optional[int] = None,
              order_by: str = 'ASC',
              verbose: bool = False) -> duckdb.DuckDBPyRelation:
        """
        Performs a SQL GROUP BY operation on the specified base_column(s)
        or all columns if no base_column is provided.
        Orders the result by the count of items in each group and assigns
        a unique 'group_name_prefix' to each group.

        Args:
            base_column: The column(s) to group by. Can be a single column name (str)
                         or a list of column names (List[str]). If None, defaults
                         to all columns in the .data member.
            name_prefix: The prefix for naming each group (e.g., 'GROUP_').
            group_column_name: The name of the groupings column (defaults to 'Group_Name')
            count_column_name: The name of the counts column (defaults to 'Count')
            limit: An optional integer to limit the number of rows in the result set.
                   If None, no limit is applied.
            order_by: How to sort the result set, defaults to 'ASC'
            verbose: if True, show debug messages

        Returns:
            A DuckDBPyRelation object with columns 'Group_Name' and 'Count'.
            Example:
            Group_Name | Count
            -----------|------
            GROUP_1    | 1
            GROUP_2    | 5
        """
        if self.data is None:
            raise ValueError("groupings: No data loaded in the CS object. Cannot perform groupings.")

        # Determine columns to group by
        if base_column is None:
            # Get all column names from the current relation
            group_by_columns = [f'"{col}"' for col in self.data.columns]
        elif isinstance(base_column, str):
            group_by_columns = [f'"{base_column}"']
        elif isinstance(base_column, list) and all(isinstance(col, str) for col in base_column):
            group_by_columns = [f'"{col}"' for col in base_column]
        else:
            raise ValueError("base_column must be a string, a list of strings, or None.")
        # Process ORDER BY
        if order_by.upper() != "DESC":
          order_by = "ASC"
        if (order_by.upper() != "ASC") and (order_by.upper() != "DESC"):
            raise ValueError(f"groupings: order_by must be one of 'ASC' or 'DESC' ({order_by=})")
        # Construct the GROUP BY clause
        group_by_clause = ", ".join(group_by_columns)

        group_query_sql = f"""
            WITH GroupedData AS (
                SELECT {group_by_clause}, COUNT(*) AS group_count
                FROM "{self._tablename}"
                GROUP BY {group_by_clause}
            ),
            OrderedGroups AS (
                SELECT *, ROW_NUMBER() OVER (ORDER BY group_count) AS rn
                FROM GroupedData
            )
            SELECT '{name_prefix}' || rn AS "{group_column_name}", 
                   group_count AS "{count_column_name}"
            FROM OrderedGroups
            ORDER BY "{count_column_name}" {order_by.upper()}
        """
        # Add LIMIT if needed
        if limit and (isinstance(limit,int) and (limit>0)):
            group_query_sql += f"    LIMIT {limit}"
        # Add final ';' even not needed really
        group_query_sql += ";"
        if verbose:
            print(f"Executing groupings SQL:\n{group_query_sql}", file=sys.stderr)

        try:
            result_relation = self.cx.sql(group_query_sql)
            return result_relation
        except duckdb.Error as e:
            print(f"Error performing groupings: {e}", file=sys.stderr)
            raise
