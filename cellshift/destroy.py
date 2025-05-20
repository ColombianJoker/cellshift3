
from math import ceil
from tqdm import tqdm
import mmap
import os
import sys
from .auxiliary import get_file_size, generate_kb_code, generate_mb_code
from typing import Union, Optional


def fast_overwrite(filename: Union[str, list, tuple], verbose: bool = False) -> bool:
    """
    Fast overwrite files with MMAP
  
    Args:
        filename: name of the file to overwrite, can be a list or tuple of strings, to overwrite many files
        verbose: if to show progress with tqdm
    Returns:
        if all the files were overwritten
    """
  
    def overwrite_one(a_filename: str, verbose: bool = False) -> bool:
        """
        Fast overwrite of ONE file with MMAP
    
        Args:
            filename: name of the file to overwrite
        Returns:
            if the file was overwritten
        """
        kb, mb = 1024, 1024*1024
        four_kb, four_mb = 4*kb, 4*mb
        file_size=get_file_size(a_filename)
        if verbose:
          print(f"× {a_filename} to be overwritten.", file=sys.stderr)
        if file_size>0:
            try:
                with open(a_filename, "r+b") as f:
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
            except:
                return False
            return True
        else:
            return True
    # end overwrite_one()
  
    if isinstance(filename, str) and os.path.isfile(filename):
        if verbose:
            print(f"× overwrite {filename}", file=sys.stderr)
        return overwrite_one(filename, verbose)
    else:
        all_true = False
        if isinstance(filename, (list, tuple)):
            filename_list = filename
            if verbose:
                print(f"× overwrite files {filename_list}", file=sys.stderr)
            return_codes = []
            return_codes = [overwrite_one(item, verbose) for item in filename_list]
            all_true = all(return_codes)
            if verbose:
                print(f"{all_true=}", file=sys.stderr)
            return all_true
        else:
            return False

def destroy(filename: str, verbose: bool = False) -> bool:
    """
    Destroy, by overwriting and removing the given files
  
    Args:
        filename: name of the file to destroy (overwrite then remove). Can be a str or a list or tuple
                  of strings
        verbose: if to show progress with tqdm
    Returns:
        if the files were destroyed
    """
  
    def destroy_one(a_filename: str, verbose: bool = False) -> bool:
      """
      Destroy, by overwriting and removing the given ONE file
    
      Args:
          filename: name of the one file to destroy (overwrite then remove). A str
      Returns:
          if the file was destroyed
      """
      if fast_overwrite(a_filename, verbose):
          try:
              os.unlink(a_filename)
              if verbose:
                  print(f"{a_filename} destroyed.")
              return True
          except Exception as err:
              print(f"Error {err} on {a_filename}", file=sys.stderr)
              return False
      else:
          if verbose:
              print(f"Could not overwrite {a_filename}", file=sys.stderr)
          return False
    # end destroy_one()
    
    if isinstance(filename, str) and os.path.isfile(filename):
        if verbose:
            print(f"× destroy {filename}", file=sys.stderr)
        return destroy_one(filename, verbose)
    else:
        all_true = False
        if isinstance(filename, (list, tuple)):
            filename_list = filename
            if verbose:
                print(f"× destroy {filename_list}", file=sys.stderr)
            return_codes = []
            return_codes = [destroy_one(item, verbose) for item in filename_list]
            all_true = all(return_codes)
            if verbose:
                print(f"{all_true=}", file=sys.stderr)
            return all_true
        else:
          return False
