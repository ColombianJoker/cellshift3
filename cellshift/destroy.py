
import glob
from math import ceil
import mmap
import os
import sys
from tqdm import tqdm
from typing import Union, Optional, List, Tuple
from .auxiliary import get_file_size, generate_kb_code, generate_mb_code

def fast_overwrite(filename: Union[str, List[str], Tuple[str]], verbose: bool = False) -> bool:
    """
    Fast overwrite files with MMAP.
    Can accept a single file path, a list/tuple of file paths, or a glob pattern.

    Args:
        filename: Name of the file(s) to overwrite.
                  Can be a str (single file path or glob pattern) or a list/tuple of strings.
        verbose: If True, show progress with tqdm.
    Returns:
        True if all specified files were successfully overwritten, False otherwise.
    """

    def overwrite_one(a_filename: str, verbose: bool = False) -> bool:
        """
        Fast overwrite of ONE file with MMAP.

        Args:
            a_filename: Name of the file to overwrite.
            verbose: If True, show progress.
        Returns:
            True if the file was overwritten, False otherwise.
        """
      
        kb, mb = 1024, 1024 * 1024
        four_kb, four_mb = 4 * kb, 4 * mb
        file_size = get_file_size(a_filename)
        # if verbose:
        #     print(f"× {a_filename} to be overwritten.", file=sys.stderr)
        if file_size > 0:
            try:
                with open(a_filename, "r+b") as f:
                    # Map all file (0 is all file)
                    mapped_file = mmap.mmap(f.fileno(), 0)
                    if file_size >= mb:
                        # Overwrite it with MB blocks
                        four_mb_code = generate_mb_code(4).encode()
                        file_size_in_4mb = ceil(file_size / four_mb)
                        for i in tqdm(range(file_size_in_4mb), disable=not verbose):
                            block_start = i * four_mb
                            block_end = (i + 1) * four_mb
                            if file_size >= block_end:
                                mapped_file[block_start:block_end] = four_mb_code
                            else:
                                mapped_file[block_start:file_size] = four_mb_code[:(file_size % four_mb)]
                            mapped_file.flush()
                    else:
                        four_kb_code = generate_kb_code(4).encode()
                        file_size_in_4kb = ceil(file_size / (4 * 1024))
                        for i in tqdm(range(file_size_in_4kb), disable=not verbose):
                            mapped_file[i * four_kb:(i + 1) * four_kb] = four_kb_code
                        mapped_file.flush()
            except Exception as e: # Catch specific exception if possible, or broad Exception
                print(f"Error during overwrite of {a_filename}: {e}", file=sys.stderr)
                return False
            return True
        else:
            # File is empty or does not exist (get_file_size returned 0), consider it overwritten.
            return True
    # end overwrite_one()

    files_to_process: List[str] = []

    if isinstance(filename, str):
        # Check if the string contains glob special characters
        if any(char in filename for char in ['*', '?', '[']):
            found_files = glob.glob(filename)
            if not found_files:
                # if verbose:
                #     print(f"No files found matching glob pattern: '{filename}'", file=sys.stderr)
                return False # No files to overwrite
            files_to_process.extend(found_files)
            # if verbose:
            #     print(f"Expanding glob '{filename}' to: {found_files}", file=sys.stderr)
        else:
            # It's a single file path, not a glob pattern
            if os.path.isfile(filename):
                files_to_process.append(filename)
                # if verbose:
                #     print(f"Processing single file: '{filename}'", file=sys.stderr)
            else:
                if verbose:
                    print(f"File not found: '{filename}'", file=sys.stderr)
                return False # File not found
    elif isinstance(filename, (list, tuple)): # It's a list or tuple of file paths
        for f in filename:
            if not isinstance(f, str):
                # if verbose:
                #     print(f"Invalid item in list/tuple: '{f}' is not a string.", file=sys.stderr)
                return False # Invalid input type in list
            if os.path.isfile(f):
                files_to_process.append(f)
            # else:
            #     if verbose:
            #         print(f"File not found in list/tuple: '{f}' (skipping).", file=sys.stderr)
        if not files_to_process and verbose:
            print("No valid files found in the provided list/tuple to process.", file=sys.stderr)
            return False
    else:
        if verbose:
            print(f"Unsupported filename type: {type(filename)}", file=sys.stderr)
        return False

    if not files_to_process:
        if verbose:
            print("No files to overwrite after processing input.", file=sys.stderr)
        return False

    # Process all collected files
    return_codes = [overwrite_one(item, verbose) for item in files_to_process]
    all_true = all(return_codes) # True if all calls to overwrite_one returned True
    # if verbose:
    #     print(f"result: {all_true=}", file=sys.stderr)
    return all_true
    
def destroy(filename: Union[str, List[str], Tuple[str]], verbose: bool = False) -> bool:
    """
    Destroy, by overwriting and removing the given files.
    Can accept a single file path, a list/tuple of file paths, or a glob pattern.

    Args:
        filename: Name of the file(s) to destroy (overwrite then remove).
                  Can be a str (single file path or glob pattern) or a list/tuple of strings.
        verbose: If True, show progress and error messages.
    Returns:
        True if all specified files were successfully destroyed, False otherwise.
    """

    def destroy_one(a_filename: str, verbose: bool = False) -> bool:
        """
        Destroy, by overwriting and removing the given ONE file.

        Args:
            a_filename: Name of the one file to destroy (overwrite then remove). A str.
            verbose: If True, show progress.
        Returns:
            True if the file was destroyed, False otherwise.
        """
        if fast_overwrite(a_filename, verbose):
            try:
                os.unlink(a_filename) # Remove the file
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

    files_to_process: List[str] = []

    if isinstance(filename, str):
        # Check if the string contains glob special characters
        if any(char in filename for char in ['*', '?', '[']):
            found_files = glob.glob(filename)
            if not found_files:
                if verbose:
                    print(f"No files found matching glob pattern: '{filename}'", file=sys.stderr)
                return False # No files to destroy
            files_to_process.extend(found_files)
            # if verbose:
            #     print(f"Expanding glob '{filename}' to: {found_files}", file=sys.stderr)
        else:
            # It's a single file path, not a glob pattern
            if os.path.isfile(filename):
                files_to_process.append(filename)
                # if verbose:
                #     print(f"Processing single file: '{filename}'", file=sys.stderr)
            else:
                if verbose:
                    print(f"File not found: '{filename}'", file=sys.stderr)
                return False # File not found
    elif isinstance(filename, (list, tuple)):
        # It's a list or tuple of file paths
        for f in filename:
            if not isinstance(f, str):
                # if verbose:
                #     print(f"Invalid item in list/tuple: '{f}' is not a string.", file=sys.stderr)
                return False # Invalid input type in list
            if os.path.isfile(f):
                files_to_process.append(f)
            else:
                if verbose:
                    print(f"File not found in list/tuple: '{f}' (skipping).", file=sys.stderr)
        if not files_to_process and verbose:
            print("No valid files found in the provided list/tuple to process.", file=sys.stderr)
            return False
    else:
        if verbose:
            print(f"Unsupported filename type: {type(filename)}", file=sys.stderr)
        return False

    if not files_to_process:
        if verbose:
            print("No files to destroy after processing input.", file=sys.stderr)
        return False

    # Process all collected files
    return_codes = [destroy_one(item, verbose) for item in files_to_process]
    all_true = all(return_codes) # True if all calls to destroy_one returned True
    # if verbose:
    #     print(f"result: {all_true=}", file=sys.stderr)
    return all_true
