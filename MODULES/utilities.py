# Main data visualization modules
# (this should be finally implemented as a python package)

import os
import glob, re

# Hello
def hello():
    return "hello world"

# Clear terminal console
def clear_TerminalConsole():
    # For Windows
    if os.name == "nt":
        _ = os.system("cls")

    # For Mac & Linux (here, os.name is "posix")
    else:
        _ = os.system("clear")

# Increment output filenames
def outputFilename(output_path):
    numList = [0]
    currentOutputs = glob.glob(output_path + "/*.png")
    for img in currentOutputs:
        i = os.path.splitext(img)[0]
        try:
            num = re.findall("[0-9]+$", i)[0]
            numList.append(int(num))
        except IndexError:
            pass
    numList = sorted(numList)
    new_Num = numList[-1] + 1
    saveName = "output_%04d.png" % new_Num

    return saveName
