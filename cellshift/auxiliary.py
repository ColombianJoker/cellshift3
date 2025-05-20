#!/usr/bin/env python3 
import math
import os
import random
import string

def letters_for(n_digits: int) -> str:
  """
  Calculates the minimum number of latin letters needed to represent a number of n digits.
  Ex: 3 letter for 4 digits
  
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
      return "".join(random.choices(string.ascii_uppercase, k=n_letters))
  else:
      return "".join(random.choices(string.ascii_lowercase, k=n_letters))

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
  
