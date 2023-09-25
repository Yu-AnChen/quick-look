import os
import pathlib
import platform


# handle Windows shortcut
def get_path(path):
    path = str(path)
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        path = os.path.realpath(path)
    elif system == "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        if path.endswith('lnk'):
            _path = path[:]
            path = shell.CreateShortCut(path).Targetpath
            assert path != '', (f"{_path} does not exist")
    else:
        print('Unknown os.')
    return pathlib.Path(path)