#libraries
from pathlib import Path

def main():
    #get current working directory
    cwd = Path.cwd()
    print(f'Current working directory: {cwd}')
    #get parent directory
    parent = cwd.parent
    print(f'Parent directory: {parent}')

if __name__ == '__main__':
    main()