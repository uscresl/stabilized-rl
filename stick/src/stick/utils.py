from io import TextIOWrapper
from pathlib import Path
from typing import Union
import os
import sys
import fnmatch

PathIsh = Union[str, bytes, Path]
FileIsh = Union[None, PathIsh, TextIOWrapper]


def splitall(path: PathIsh) -> list[str]:
    """Split a path into the drive (on windows), the directories, and the file
    in one list."""
    # Files, dirs, or drives
    parts = []
    while path:
        head, tail = os.path.split(path)
        if tail:
            parts.append(tail)
        else:
            parts.append(head)
            break
        path = head
    return list(reversed(parts))


def is_instance_str(obj, type_names):
    """An isinstance check that does not require importing the type's module."""
    obj_type_str = f"{type(obj).__module__}.{type(obj).__name__}"
    if isinstance(type_names, str):
        return fnmatch.fnmatch(obj_type_str, type_names)
    else:
        for type_name in type_names:
            if fnmatch.fnmatch(obj_type_str, type_name):
                return True
        return False


class FileManager:
    def __init__(self, file: FileIsh = None):
        self.should_close = False
        if file is None:
            self.file = sys.stdout
        elif isinstance(file, (str, bytes, Path)):
            if file == "stderr":
                self.file = sys.stderr
            elif file == "stdout":
                self.file = sys.stdout
            else:
                path, _ = os.path.split(file)
                os.makedirs(path, exist_ok=True)
                self.file = open(file, "a+")
                self.should_close = True
        else:
            self.file = file

    def close(self):
        if self.should_close:
            self.file.close()
