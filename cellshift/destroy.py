
from math import ceil
from tqdm import tqdm
import mmap
import os
import sys
from .auxiliary import get_file_size, generate_kb_code, generate_mb_code

def fast_overwrite(filename: str, verbose: bool = False) -> bool:
  """
  Fast overwrite of file with MMAP
  
  Args:
      filename: name of the file to overwrite
      verbose: if to show progress with tqdm
  Returns:
      if the file was overwritten
  """
  if isinstance(filename, str) and os.path.isfile(filename):
    kb, mb = 1024, 1024*1024
    four_kb, four_mb = 4*kb, 4*mb
    file_size=get_file_size(filename)
    if file_size>0:
      with open(filename, "r+b") as f:
        # Map all file (0 is all file)
        mapped_file=mmap.mmap(f.fileno(),0)
        if file_size>=mb:
          # Overwrite it with MB blocks
          four_mb_code=generate_mb_code(4).encode()
          file_size_in_4mb=ceil(file_size/four_mb)
          for i in tqdm(range(file_size_in_4mb), disable=not verbose):
            block_start=i*four_mb
            block_end=(i+1)*four_mb
            if file_size>=block_end:
              mapped_file[block_start:block_end]=four_mb_code
            else:
              mapped_file[block_start:file_size]=four_mb_code[:(file_size%four_mb)]
            mapped_file.flush()
        else:
          four_kb_code=generate_kb_code(4).encode()
          file_size_in_4kb=ceil(file_size/(4*1024))
          for i in tqdm(range(file_size_in_4kb), disable=not verbose):
            mapped_file[i*four_kb:(i+1)*four_kb]=four_kb_code
          mapped_file.flush()
      return True
    else:
      return False
  return False

def destroy(filename: str, verbose: bool = False) -> bool:
  """
  Destroy, by overwriting and removing the given file
  
  Args:
      filename: name of the file to destroy (overwrite then remove)
      verbose: if to show progress with tqdm
  Returns:
    if the file was destroyed
  """
  if isinstance(filename, str) and os.path.isfile(filename):
    if fast_overwrite(filename, verbose):
      try:
        os.unlink(filename)
        if verbose:
          print(f"{filename} destroyed.")
        return True
      except Exception as err:
        print(f"Error {err} on {filename}", file=sys.stderr)
        return False
    else:
      if verbose:
        print(f"Could not overwrite {filename}", file=sys.stderr)
      return False
  return False
