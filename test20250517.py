#!/usr/bin/env python3

from cellshift import CS
from IPython import display
import sys

try:
  r1 = CS("seattle_pets.csv")
  print(f"{type(r1)=}\n")
  print(f"{r1=}\n")
  print(f"{dir(r1)=}\n")
  print(f"{type(r1.data)=}\n")
  print(f"{r1.get_tablename()=}\n")
  print(f"{type(r1.to_pandas())=}\n")
  r1.to_csv("copy.csv")
  df1 = r1.to_pandas()
  print(f"{type(df1)=}\n")
except Exception as e:
  print(f"{e=}", file=sys.stderr)
