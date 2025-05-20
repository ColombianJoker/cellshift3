#!/usr/bin/env python3 
import math
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
  LETTERS=26
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
