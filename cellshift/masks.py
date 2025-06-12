import duckdb
from duckdb.typing import BIGINT, INTEGER, VARCHAR
import polars as pl
import pyarrow as pa
from typing import Union, List, Optional, Iterator
import uuid
# import tempfile
# import secrets
import random
import sys
from . import CS
from . import table_name_generator

_table_name_gen = table_name_generator()

def _mask_value_for_duckdb(value: Union[int, str],
                           mask_left_count: int,
                           mask_right_count: int,
                           mask_char_str: str) -> str:
    """
    Internal helper for DuckDB UDF. Masks an integer or string value to a string.
    Handles numeric signs if applicable.
    """
    if not mask_char_str:
        mask_char_single = " "
    else:
        mask_char_single = mask_char_str[0]

    original_str_value = str(value)
    processed_value_str = original_str_value
    sign_prefix = ""

    # Adjust sign handling to be robust for both int and string inputs
    if isinstance(value, int):
        if value < 0:
            sign_prefix = "-"
            processed_value_str = original_str_value[1:]
    elif isinstance(value, str):
        # If it's a string that starts with '-' and the rest is digits, treat it as a negative number string
        if original_str_value.startswith("-") and original_str_value[1:].isdigit():
            sign_prefix = "-"
            processed_value_str = original_str_value[1:]
        # Otherwise, for general strings, no special sign handling; mask as is.

    current_length = len(processed_value_str)

    actual_mask_left = max(0, min(mask_left_count, current_length))
    actual_mask_right = max(0, min(mask_right_count, current_length - actual_mask_left))

    if actual_mask_left + actual_mask_right >= current_length:
        masked_string_parts = [mask_char_single] * current_length
    else:
        masked_string_parts = list(processed_value_str)
        # Mask left part
        for i in range(actual_mask_left):
            masked_string_parts[i] = mask_char_single

        # Mask right part (from the original string's right end)
        for i in range(current_length - actual_mask_right, current_length):
            masked_string_parts[i] = mask_char_single

    final_masked_str = "".join(masked_string_parts)

    return sign_prefix + final_masked_str
    
def add_masked_column(self,
                      base_column: str,
                      new_column_name: Optional[str] = None,
                      mask_left: int = 0,
                      mask_right: int = 0,
                      mask_char: Union[str, int] = "×",
                      verbose: bool = False) -> CS:
    """
    Adds a string column of masked values to the .data member (DuckDBPyRelation)
    by applying a mask to an INTEGER or VARCHAR base_column.

    The `mask_left` leftmost characters and `mask_right` rightmost characters of
    the `base_column` are replaced by `mask_char`.
    The function uses only the first character of `mask_char`.

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        new_column_name (str, optional): The name for the new masked VARCHAR column.
                                         Defaults to "masked_{base_column}".
        mask_left (int): The number of left digits/characters to mask. Must be non-negative.
        mask_right (int): The number of right digits/characters to mask. Must be non-negative.
        mask_char (Union[str, int]): The character/digit to use for masking.
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """

    # Determine new_column_name
    if new_column_name is None:
        new_column_name = f"masked_{base_column}"
    elif not isinstance(new_column_name, str) or not new_column_name:
        raise ValueError(f"add_masked_column: new_column_name must be a non-empty string or None.")

    # Validate base_column
    if not isinstance(base_column, str) or not base_column:
        raise ValueError(f"add_masked_column: base_column must be a non-empty string.")

    # Process mask_char for consistency and validation
    processed_mask_char: str
    if isinstance(mask_char, int):
        processed_mask_char = str(mask_char)
    elif isinstance(mask_char, str) and len(mask_char) > 0:
        processed_mask_char = mask_char[0]
    else:
        raise ValueError(f"add_masked_column: mask_char must be a non-empty string or an integer.")

    # Validate mask_left and mask_right (Python-side validation)
    if not isinstance(mask_left, int) or mask_left < 0:
        raise ValueError(f"add_masked_column: mask_left must be a non-negative integer.")
    if not isinstance(mask_right, int) or mask_right < 0:
        raise ValueError(f"add_masked_column: mask_right must be a non-negative integer.")

    # Ensure self.data is not None before accessing .columns
    if self.data is None:
        raise ValueError(f"add_masked_column: Data has not been loaded into self.data yet. Cannot process columns.")

    # Get actual column names from the relation, converted to lowercase for case-insensitive comparison
    data_column_names_lower = [col.lower() for col in self.data.columns]

    if base_column.lower() not in data_column_names_lower:
        raise ValueError(f"Column '{base_column}' not found in the table.")

    # Get the data type of the base_column
    column_type_query = f"""
        SELECT data_type
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}' AND column_name = '{base_column}'
    """
    if verbose:
        print(f"add_masked_column: {column_type_query=}", file=sys.stderr)
    result = self.cx.execute(column_type_query).fetchall()

    if not result:
         raise ValueError(f"add_masked_column: Could not retrieve type for column '{base_column}'. This indicates an unexpected state.")

    column_data_type = result[0][0].upper() # Convert to uppercase for consistent comparison

    # --- Determine UDF name and type flag based on column_data_type ---
    int_version_fn = False
    fn_name_in_duckdb: str

    if "INT" in column_data_type: # matches BIGINT, INTEGER, SMALLINT etc.
        fn_name_in_duckdb = 'mask_int_val'
        int_version_fn = True
    elif column_data_type in ("CHAR", "VARCHAR", "STRING"): # Also includes CHAR
        fn_name_in_duckdb = 'mask_char_val'
        int_version_fn = False # Explicitly set to False for char/varchar/string
    else:
        raise ValueError(f"Column '{base_column}' is not of an allowed type (INTEGER or VARCHAR/CHAR/STRING). It is '{column_data_type}'.")

    # --- Register the appropriate UDF if not already registered ---
    get_fn_query = f"SELECT COUNT(*) FROM duckdb_functions() WHERE function_name='{fn_name_in_duckdb}'"
    if verbose:
        print(f"add_masked_column: {get_fn_query=}", file=sys.stderr)
    fn_count = self.cx.execute(get_fn_query).fetchall()[0][0]

    if fn_count == 0:
        if int_version_fn:
            self.cx.create_function(
                fn_name_in_duckdb,
                _mask_value_for_duckdb, # Use the generic Python function
                [BIGINT, INTEGER, INTEGER, VARCHAR], # DuckDB expects BIGINT if the column is an integer type
                VARCHAR
            )
            if verbose:
                print(f"add_masked_column: UDF {fn_name_in_duckdb=} (INT version) created.", file=sys.stderr)
        else: # not int_version_fn, meaning it's a CHAR/VARCHAR/STRING type
            self.cx.create_function(
                fn_name_in_duckdb,
                _mask_value_for_duckdb, # Use the generic Python function
                [VARCHAR, INTEGER, INTEGER, VARCHAR], # DuckDB expects VARCHAR for string types
                VARCHAR
            )
            if verbose:
                print(f"add_masked_column: UDF {fn_name_in_duckdb=} (CHAR version) created.", file=sys.stderr)
    else:
        if verbose:
            print(f"add_masked_column: UDF {fn_name_in_duckdb=} already registered.", file=sys.stderr)

    # Update self.data to the new relation which includes the masked column
    alter_add_column = f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" VARCHAR;"
    if verbose:
        print(f"{alter_add_column=}", file=sys.stderr)
    self.cx.execute(alter_add_column)
    update_new_column = f"""UPDATE \"{self._tablename}\"
                            SET \"{new_column_name}\"={fn_name_in_duckdb}(\"{base_column}\", {mask_left}, {mask_right}, '{processed_mask_char}')
                           """
    if verbose:
        print(f"{update_new_column=}", file=sys.stderr)
    self.cx.execute(update_new_column)
    self.data = self.cx.table(self._tablename)
    if verbose:
        print(f"{self.data.columns=}", file=sys.stderr)
        self.data.show()

    return self

def _escape_regex_for_literal_match(self, s: str) -> str:
    """Escapes special regex characters in a string for a literal match."""
    special_chars = r".\+*?[()]|{}<>"
    escaped_s = "".join(["\\" + char if char in special_chars else char for char in s])
    return escaped_s

def add_masked_mail_column(self,
                            base_column: str,
                            new_column_name: Optional[str] = None,
                            mask_user: Optional[Union[bool, str]] = False,
                            mask_domain: Optional[Union[bool, str]] = False,
                            domain_choices: Optional[Union[bool, str, List[str]]] = False,
                            verbose: bool = False) -> 'CS':
    """
    Adds a string column of masked values to the .data member by replacing the user part of email
    if chosen, replacing the domain part if chosen, and replacing from a list of domains if given.
    When domain_choices is a list, it masks 1/n of the eligible unmasked rows for each choice.

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        new_column_name (str, optional): The name for the new masked VARCHAR column.
                                         Defaults to "masked_{base_column}".
        mask_user (Optional[Union[bool, str]]): Or True (replaces with a default) or a string with
                                                the replacement (for all values in base_column).
                                                Defaults to False (don't replace).
        mask_domain (Optional[Union[bool, str]]): Or True (replaces with something) or a string with
                                                  the replacement (for all values in base_column).
                                                  Defaults to False (don't replace).
        domain_choices (Optional[Union[bool, str, List[str]]]): True (replaces with something), a string
                                                                (replaces all domains with this string),
                                                                or a list of strings (replaces 1/n of eligible
                                                                rows for each choice in the list).
                                                                Defaults to False.
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """

    # --- Argument Validation (copied from your original code) ---
    if new_column_name is None:
        new_column_name = f"masked_{base_column}"
    elif not isinstance(new_column_name, str) or not new_column_name:
        raise ValueError(f"add_masked_column: new_column_name must be a non-empty string or None.")

    if not isinstance(base_column, str) or not base_column:
        raise ValueError(f"add_masked_column: base_column must be a non-empty string.")

    if not isinstance(mask_user, bool) and not isinstance(mask_user, str):
        raise ValueError(f"add_masked_column: mask_user must be boolean or a string.")
    if not isinstance(mask_domain, bool) and not isinstance(mask_domain, str):
        raise ValueError(f"add_masked_column: mask_domain must be boolean or a string.")
    if not isinstance(domain_choices, bool) and \
       (not isinstance(domain_choices, str) and not isinstance(domain_choices, List)):
        raise ValueError(f"add_masked_column: domain_choices must be boolean or a string or a list of strings.")

    if self.data is None:
        raise ValueError(f"add_masked_column: Data has not been loaded into self.data yet. Cannot process columns.")

    data_column_names_lower = [col.lower() for col in self.data.columns]

    if base_column.lower() not in data_column_names_lower:
        raise ValueError(f"Column '{base_column}' not found in the table.")

    column_type_query = f"""
        SELECT data_type
        FROM duckdb_columns()
        WHERE table_name = '{self._tablename}' AND column_name = '{base_column}'
    """
    if verbose:
        print(f"add_masked_column: {column_type_query=}", file=sys.stderr)
    result = self.cx.execute(column_type_query).fetchall()

    if not result:
        raise ValueError(f"add_masked_column: Could not retrieve type for column '{base_column}'. This indicates an unexpected state.")

    column_data_type = result[0][0].upper()

    if column_data_type not in ("CHAR", "VARCHAR", "STRING"):
        raise ValueError(f"Column '{base_column}' is not of an allowed type (VARCHAR/CHAR/STRING). It is '{column_data_type}'.")

    # --- Add new_column_name if it doesn't exist ---
    if new_column_name.lower() not in data_column_names_lower:
        alter_add_column: str = f"ALTER TABLE \"{self._tablename}\" ADD COLUMN \"{new_column_name}\" VARCHAR;"
        if verbose:
            print(f"add_masked_mail_column: {alter_add_column=}", file=sys.stderr)
        self.cx.execute(alter_add_column)
        # Ensure the new column is initially NULL to correctly track unmasked rows
        self.cx.execute(f"UPDATE \"{self._tablename}\" SET \"{new_column_name}\" = NULL WHERE \"{new_column_name}\" IS NULL;")
        self.data = self.cx.table(self._tablename) # Refresh relation

    # --- Masking Logic ---
    user_mask_str: Optional[str] = None
    domain_mask_str: Optional[str] = None

    # Determine user_mask_str
    if mask_user:
        user_mask_str = '????????' if mask_user is True else mask_user

    # Determine domain_mask_str for mask_domain=True or a string
    if mask_domain:
        if mask_domain is True:
            # If domain_choices is not specified or is False, use default choices
            if not domain_choices: # domain_choices could be False, True, str, or List
                # domain_mask_str = secrets.choice([
                domain_mask_str = random.choice([
                    "example.com", "example.org", "example.net",
                    "example.edu", "example.co"
                ])
            elif isinstance(domain_choices, str): # If domain_choices is a single string
                domain_mask_str = domain_choices
            elif isinstance(domain_choices, list): # If domain_choices is a list, don't set here
                # This case will be handled by the specific domain_choices loop later
                domain_mask_str = None
            else: # domain_choices is True, but not a list or string, fall back to default
                # domain_mask_str = secrets.choice([
                domain_mask_str = random.choice([
                    "example.com", "example.org", "example.net",
                    "example.edu", "example.co"
                ])
        else: # mask_domain is a specific string
            domain_mask_str = mask_domain

    # --- Build resrc and remsk for general masking (mask_user or mask_domain as single value) ---
    resrc_general: Optional[str] = None
    remsk_general: Optional[str] = None

    if user_mask_str and domain_mask_str:
        resrc_general = "^([\w._-]+)@([\w._-]+)"
        remsk_general = f"{user_mask_str}@{domain_mask_str}"
    elif user_mask_str:
        resrc_general = "^([\w._-]+)@"
        remsk_general = f"{user_mask_str}@"
    elif domain_mask_str: # Only mask_domain is true and not handling domain_choices list yet
        resrc_general = "@([\w._-]+)"
        remsk_general = f"@{domain_mask_str}"

    # Apply general masking if applicable
    if resrc_general and remsk_general:
        update_general_mask: str = f"""
            UPDATE \"{self._tablename}\"
            SET \"{new_column_name}\" = regexp_replace({base_column}, '{resrc_general}', '{remsk_general}')
            WHERE \"{new_column_name}\" IS NULL; -- Only update unmasked rows
        """
        if verbose:
            print(f"add_masked_mail_column: {update_general_mask=}", file=sys.stderr)
        self.cx.execute(update_general_mask)
        self.data = self.cx.table(self._tablename) # Refresh relation
        if verbose:
            print(f"Applied general mask to {base_column} into {new_column_name}.")

    # --- Handle domain_choices as a list for randomized, segmented updates ---
    if isinstance(domain_choices, list) and domain_choices:
        n = len(domain_choices)
        if n == 0:
            if verbose:
                print("add_masked_mail_column: domain_choices list is empty. No specific domain masking performed.", file=sys.stderr)
        else:
            resrc_for_domain_choice = "@([\w._-]+)" # Matches "@domain" part

            exclusion_conditions = []
            for choice in domain_choices:
                # Escape each choice for literal matching in REGEXP
                escaped_choice = self._escape_regex_for_literal_match(choice)
                exclusion_conditions.append(f"regexp_matches({base_column}, '.*{escaped_choice}$')") # Check if email ends with this domain
        
            # Combine exclusion conditions using OR, then negate the whole thing
            # i.e., NOT (base_column matches choice1 OR base_column matches choice2)
            exclusion_clause_list = " OR ".join(exclusion_conditions)
            if exclusion_clause_list:
                # Exclude rows where base_column already contains one of the `domain_choices` domains
                final_exclusion_where = f"AND NOT ({exclusion_clause_list})"
            else:
                final_exclusion_where = "" # No exclusion if domain_choices list is empty or boolean

            for i, one_item_remsk in enumerate(domain_choices):
                # Ensure one_item_remsk is a string, if not, convert or raise error.
                if not isinstance(one_item_remsk, str):
                    if verbose:
                        print(f"add_masked_mail_column: Skipping non-string item in domain_choices: {one_item_remsk}", file=sys.stderr)
                    continue

                # The `regexp_replace` will only act if the `resrc_for_domain_choice` pattern is found.
                # The `remsk` will be the `one_item_remsk`
                remsk_current_iteration = f"@{one_item_remsk}" # Format for replacement

                # Use new_column_name for update target
                update_sql_base = f"""
                    WITH EligibleRows AS (
                        SELECT
                            id,
                            {base_column}
                        FROM \"{self._tablename}\"
                        WHERE \"{new_column_name}\" IS NULL -- Only consider unmasked rows
                        {final_exclusion_where}
                    )
                """

                if i == n - 1: # Last item in the list
                    # Update all remaining eligible unmasked rows
                    update_sql = f"""
                        {update_sql_base}
                        UPDATE \"{self._tablename}\"
                        SET \"{new_column_name}\" = regexp_replace(\"{self._tablename}\".\"{base_column}\", '{resrc_for_domain_choice}', '{remsk_current_iteration}')
                        FROM EligibleRows
                        WHERE \"{self._tablename}\".id = EligibleRows.id;
                    """
                    if verbose:
                        print(f"add_masked_mail_column: Executing final update for domain '{one_item_remsk}' (all remaining eligible unmasked rows).", file=sys.stderr)
                    self.cx.execute(update_sql)
                else:
                    # Update approximately 1/n of the remaining eligible unmasked rows
                    update_sql = f"""
                        {update_sql_base},
                        NumberedEligibleRows AS (
                            SELECT
                                id,
                                {base_column},
                                ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn -- Assign a random row number
                            FROM EligibleRows
                        )
                        UPDATE \"{self._tablename}\"
                        SET \"{new_column_name}\" = regexp_replace(\"{self._tablename}\".\"{base_column}\", '{resrc_for_domain_choice}', '{remsk_current_iteration}')
                        FROM NumberedEligibleRows
                        WHERE \"{self._tablename}\".id = NumberedEligibleRows.id
                          AND NumberedEligibleRows.rn % {n} = {i}; -- Select ~1/n of rows for this iteration
                    """
                    if verbose:
                        print(f"add_masked_mail_column: Executing update for domain '{one_item_remsk}' (segment {i+1}/{n}).", file=sys.stderr)
                        print(f"{update_sql}", file=sys.stderr)
                    self.cx.execute(update_sql)

                self.data = self.cx.table(self._tablename) # Refresh relation after each update
                if verbose:
                    print(f"add_masked_mail_column: Updated with '{one_item_remsk}' for segment {i+1}/{n}.", file=sys.stderr)
                    self.data.show()

    self.data = self.cx.table(self._tablename) # Final refresh
    return self

def mask_column(self,
                base_column: str,
                mask_left: int = 0,
                mask_right: int = 0,
                mask_char: Union[str, int] = "×",
                verbose: bool = False) -> CS:
    """
    Converts a column to a column of masked values by applying a mask to an INTEGER or VARCHAR base_column.

    The `mask_left` leftmost characters and `mask_right` rightmost characters of
    the `base_column` are replaced by `mask_char`.
    The function uses only the first character of `mask_char`.

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        mask_left (int): The number of left digits/characters to mask. Must be non-negative.
        mask_right (int): The number of right digits/characters to mask. Must be non-negative.
        mask_char (Union[str, int]): The character/digit to use for masking.
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """
    new_column_name: str = f"masked_{base_column}"                   
    self.add_masked_column(base_column,
                           new_column_name=new_column_name,
                           mask_left=mask_left,
                           mask_right=mask_right,
                           mask_char=mask_char,
                           verbose=verbose)
    self.replace_column(base_column, new_column_name, verbose=verbose)
    self.drop_column(new_column_name, verbose=verbose)
    return self

def mask_mail_column(self,
                            base_column: str,
                            mask_user: Optional[Union[bool, str]] = False,
                            mask_domain: Optional[Union[bool, str]] = False,
                            domain_choices: Optional[Union[bool, str, List[str]]] = False,
                            verbose: bool = False) -> 'CS':
    """
    Converts a string column to a column of masked emails by replacing the user part of email
    if chosen, replacing the domain part if chosen, and replacing from a list of domains if given.
    When domain_choices is a list, it masks 1/n of the eligible unmasked rows for each choice.

    Args:
        base_column (str): The name of the existing INTEGER or VARCHAR column to mask.
        mask_user (Optional[Union[bool, str]]): Or True (replaces with a default) or a string with
                                                 the replacement (for all values in base_column).
                                                 Defaults to False (don't replace).
        mask_domain (Optional[Union[bool, str]]): Or True (replaces with something) or a string with
                                                   the replacement (for all values in base_column).
                                                   Defaults to False (don't replace).
        domain_choices (Optional[Union[bool, str, List[str]]]): True (replaces with something), a string
                                                                 (replaces all domains with this string),
                                                                 or a list of strings (replaces 1/n of eligible
                                                                 rows for each choice in the list).
                                                                 Defaults to False.
        verbose (bool): If True, will show debug messages.

    Returns:
        CS: The instance of the CS class, allowing for method chaining.

    Raises:
        ValueError: If input arguments are invalid or if base_column is not INTEGER or VARCHAR.
    """
    new_column_name: str = f"masked_{base_column}"                   
    self.add_masked_mail_column(base_column,
                                new_column_name=new_column_name,
                                mask_user=mask_user,
                                mask_domain=mask_domain,
                                domain_choices=domain_choices,
                                verbose=verbose)
    self.replace_column(base_column, new_column_name, verbose=verbose)
    self.drop_column(new_column_name, verbose=verbose)
    return self
