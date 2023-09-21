# ---------------------------------------------------------------------------- #
#                       Helper to handle Windows shortcut                      #
# ---------------------------------------------------------------------------- #
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
            path = shell.CreateShortCut(path).Targetpath
    else:
        print('Unknown os.')
    return pathlib.Path(path)