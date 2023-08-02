# coding: utf-8
import os
import zipfile
from pathlib import Path
import shutil


# unzip
def unzip(file_path):
    parent_path = Path(file_path).parent.absolute()
    zf = zipfile.ZipFile(file_path, 'r')
    zf.extractall(path=parent_path)


# remove file
def remove_allfile_fromdir(directory_path):
    for file in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, file)):
            shutil.rmtree(os.path.join(directory_path, file))
        else:
            os.remove(os.path.join(directory_path, file))

def parse_boolean(s):
    return s == "True"