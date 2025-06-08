
import math
import os
# import random
import secrets
import string

def letters_for(n_digits: int) -> str:
  """
  Calculates the minimum number of latin letters needed to represent a number of n digits. Ex: 3 letter for 4 digits
  
  Args:
      n_digits: A number (integer) of digits
  Returns:
      the minimun numbers of latin (capital) letters needed to represent a number of n_digits
  """
  LETTERS=26 # Only use basic latin (no euro) alphabet
  log_10=math.log10(LETTERS)
  n_letters=math.ceil(n_digits/log_10) # round up
  return n_letters

def random_code(n_letters: int, upper: bool = True) -> str:
  """
  Return a "code" of n random capital letters
  
  Args:
      n_letters: a number (integer) of (how-many) letters
  Returns:
      a string of random letters of length n_letters
  """
  if upper:
      # return "".join(random.choices(string.ascii_uppercase, k=n_letters))
      return ''.join(secrets.choice(string.ascii_uppercase) for _ in range(n_letters))
  else:
      # return "".join(random.choices(string.ascii_lowercase, k=n_letters))
      return ''.join(secrets.choice(string.ascii_lowercase) for _ in range(n_letters))

def generate_kb_code(n_kb: int) -> str:
  """
  Generate a KB of random uppercase 32 letter codes
  
  Args:
      n_kb: a number (integer) of KB of data to generate
  Returns:
      a string of n_kb KB of length
  """
  n_codes=(1024)//32
  kb_block=random_code(32)*n_codes
  return kb_block*n_kb

def generate_mb_code(n_mb: int) -> str:
  """
  Generate a MB of random uppercase 32 letter codes
  
  Args:
      n_mb: a number (integer) of MB of data to generate
  Returns:
      a string of n_mb MB of length
  """
  n_codes=(1024*1024)//128
  mb_block=random_code(128)*n_codes
  return mb_block*n_mb

def get_file_size(filename: str) -> int:
  """
  Get file size of file in bytes
  
  Args:
      filename: the name of the file
  Returns:
      the size in bytes
  """
  if isinstance(filename, str) and os.path.isfile(filename):
    return os.path.getsize(filename)
  else:
    return 0
  
from typing import Union, Any # Any might not be strictly needed, but useful for initial flexibility

from typing import Union

def mask_val(
    value: Union[int, str],
    mask_left: int = 0,
    mask_right: int = 0,
    mask_char: Union[str, int] = "0",
    silent_fail: bool = False) -> str:
    """
    Return a string equivalent to the given value with the specified left and right
    digits replaced by mask_char.

    Args:
        value (Union[int, str]): The value to mask. Can be an integer or a string.
        mask_left (int): The number of left digits to mask. Must be non-negative.
        mask_right (int): The number of right digits to mask. Must be non-negative.
        mask_char (Union[str, int]): The character/digit to use for masking.
                                      Only the first character is used if a string.
        silent_fail (bool): If True, validation errors will return the original
                            value as a string instead of raising a ValueError.

    Returns:
        str: The masked string, or the original value as a string if silent_fail is True
             and a validation error occurs.
    """
    original_str_value = str(value) # Convert original value to string for fallback

    # Type Conversion and Basic Validation for mask_char
    processed_mask_char_single: str
    if isinstance(mask_char, int):
        processed_mask_char_single = str(mask_char)
    elif isinstance(mask_char, str) and len(mask_char) > 0:
        processed_mask_char_single = mask_char[0]
    else:
        if silent_fail:
            return original_str_value
        raise ValueError("mask_char must be a non-empty string or an integer.")

    # Convert value to string and handle sign if integer
    sign = 1
    processed_value_str = original_str_value

    if original_str_value.startswith("-"):
        sign = -1
        processed_value_str = original_str_value[1:] # Remove sign for internal processing

    current_length = len(processed_value_str)

    # Validate mask_left and mask_right
    if not isinstance(mask_left, int) or mask_left < 0:
        if silent_fail:
            return original_str_value
        raise ValueError("mask_left must be a non-negative integer.")
    if not isinstance(mask_right, int) or mask_right < 0:
        if silent_fail:
            return original_str_value
        raise ValueError("mask_right must be a non-negative integer.")

    # Additional validation: Ensure mask_left and mask_right don't mask too much
    if mask_left + mask_right > current_length:
        if silent_fail:
            return f"{mask_char}{original_str_value}{mask_char}"
        raise ValueError(
            f"Combined mask_left ({mask_left}) and mask_right ({mask_right}) "
            f"exceed the effective length of the value ({current_length})."
        )

    # Apply Masking Logic
    masked_string_parts = list(processed_value_str) # Convert to list to modify characters

    # Mask left part
    for i in range(mask_left):
        masked_string_parts[i] = processed_mask_char_single

    # Mask right part
    for i in range(current_length - mask_right, current_length):
        masked_string_parts[i] = processed_mask_char_single
    
    final_masked_str = "".join(masked_string_parts)

    # Add back the sign if it was negative
    if sign < 0:
        final_masked_str = "-" + final_masked_str

    return final_masked_str