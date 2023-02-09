from pathlib import Path
# get resolved path to parent directory
repo_path_resolved = Path.cwd().resolve().parent
repo_path_unresolved = Path.cwd().parent
print(f'The resolved path to the parent directory is: {repo_path_resolved}\n')
print(f'The unresolved path to the parent directory is: {repo_path_unresolved}\n')

import os
repo_list = os.listdir(repo_path_resolved)
print(f'The parent directory contains the following files and folders: {repo_list}\n')
#check if .gitignore is in the parent directory, this will ensure that the parent directory is the root directory of the repo
if '.gitignore' not in repo_list:
    raise Exception('The parent directory is not the root directory of the repo')
else:
    print('All good regarding paths\n')


# add repo_path to sys.path
import sys; sys.path.insert(0,str(repo_path_resolved))
print(f'current sys.path: {sys.path}\n')

# All the code above is to to be able to import from the parent directory, whici is included in the sys.path.
# sys.path is restarted every time you run the script, coming from the PYTHONPTH variable, so you need to add the parent directory to sys.path every time you run the script.


#import main_trial
from main_trial import main_trial_function

if __name__ == '__main__':
    main_trial_function()