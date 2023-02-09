#libraries
from pathlib import Path

def main_trial_function():
    #get current working directory
    cwd = Path.cwd()
    print(f'Current working directory: {cwd}')
    #get parent directory
    parent = cwd.parent
    print(f'Parent directory: {parent}')