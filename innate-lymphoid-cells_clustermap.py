# InnateLymphoidCells data viz
# Datasets from IndicaLabs Halo : https://www.indicalab.com

# Coding: utf8

# MIT License

# Copyright (c) 2024 Ouatataz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Import necessary libraries
import os
import os
import glob, re
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import importlib
import warnings

# Get script directory filepath
dirname = os.path.dirname(__file__)

# See: https://stackoverflow.com/questions/27476642/matplotlib-get-rid-of-max-open-warning-output
mpl.rcParams.update({"figure.max_open_warning": 0}) # Get rid of max_open_warning output (matplotlib)

# Import custom utilities module
utilities_Path = dirname + "/MODULES/utilities.py"
spec = importlib.util.spec_from_file_location("utilities", utilities_Path)
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)

# Clear terminal console
utilities.clear_TerminalConsole()
###################################

# Global variables for visualization and formatting
extension = "svg"  # File extension for output images / Can be: "png", "svg", etc...
axe_label_color = "#333"  # Axis label color / Can be: "#ccc"...
axe_color = "#777"  # Axis color / Can be: "#ccc"...
labelsize = 6  # Font size for labels / Format: x pt
fontscale = 1  # Font scaling factor / Font scaling (1 = 100%)...

# Clinical dataset filename
clinical_dataset = "_clinical_data"

# Define datasets and their structure
datasets_dict = {
    "_ST&TdT_THs_Layer1+Cortex+Medulla": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
    },
    "_ST&TdT_THs_Cortex+Medulla": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
    },
    "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
    },
    "_ST&TdT_ILCs+THs_Cortex+Medulla": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
    },
    "_ALL_Merged_Layer1": {
        "multiplexMain": {
            "percentage": [
                "AA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_HighPlex_Query_Total_Summary_Results_Layer1",
                "SG_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
    },
    "_ALL_Splitted_Layer1": {
        "multiplexMain": {
            "percentage": [
                "AA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_HighPlex_Query_Total_Summary_Results_Layer1",
                "SG_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
    },
}

# Filter dataset columns
phenotypes_dict = {
    "_ST&TdT_THs_Layer1+Cortex+Medulla": {
        "percentage": [
            [
                #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                #"Early T-Cells - Density/mm²", # Immature T
                #"Mature T-Cells - Density/mm²", # Mature T
                "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "_ST&TdT_THs_Cortex+Medulla": {
        "percentage": [
            [
                #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                #"Early T-Cells - Density/mm²", # Immature T
                #"Mature T-Cells - Density/mm²", # Mature T
                "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla": {
        "percentage": [
            [
                #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                #"Early T-Cells - Density/mm²", # Immature T
                #"Mature T-Cells - Density/mm²", # Mature T
                "ILC1s - Density/mm²", # ILC1
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "ILC2 - Density/mm²", # ILC2
                "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "ILC3 - Density/mm²", # ILC3
                "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "_ST&TdT_ILCs+THs_Cortex+Medulla": {
        "percentage": [
            [
                #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                #"Early T-Cells - Density/mm²", # Immature T
                #"Mature T-Cells - Density/mm²", # Mature T
                "ILC1s - Density/mm²", # ILC1
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "ILC2 - Density/mm²", # ILC2
                "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "ILC3 - Density/mm²", # ILC3
                "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "_ALL_Merged_Layer1": {
        "multiplexMain": {
            "percentage": [
                [
                    #"% T-Cells:CD20-TCR+ Positive Cells", # T
                    "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                    "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                    "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                    "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
                ],
            ],
            "density": [
                [
                    #"T Cells - Density/mm²", # T
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC1s - Density/mm²", # ILC1
                    "TH1 - Density/mm²", # TH1
                    "ILC2 - Density/mm²", # ILC2
                    "TH2 - Density/mm²", # TH2
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 - Density/mm²", # TH17
                ],
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                [
                    #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                    #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                    "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                    "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                    "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                    "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                    "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                    "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                    "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                    "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                    "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                    "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
                ],
            ],
            "density": [
                [
                    #"Early T-Cells - Density/mm²", # Immature T
                    #"Mature T-Cells - Density/mm²", # Mature T
                    "ILC1s - Density/mm²", # ILC1
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                    "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                    "Immature TH1 - Density/mm²", # Immature TH1
                    "ILC2 - Density/mm²", # ILC2
                    "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                    "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                    "Immature TH2 - Density/mm²", # Immature TH2
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                    "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                    "Immature TH17 - Density/mm²", # Immature TH17
                ],
            ],
        },
    },
    "_ALL_Splitted_Layer1": {
        "multiplexMain": {
            "percentage": [
                [
                    "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                    "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
                ],
                [
                    #"% T-Cells:CD20-TCR+ Positive Cells", # T
                    "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                    "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                    "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
                ],
            ],
            "density": [
                [
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC1s - Density/mm²", # ILC1
                    "ILC2 - Density/mm²", # ILC2
                    "ILC3 - Density/mm²", # ILC3
                ],
                [
                    #"T Cells - Density/mm²", # T
                    "TH1 - Density/mm²", # TH1
                    "TH2 - Density/mm²", # TH2
                    "TH17 - Density/mm²", # TH17
                ],
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                [
                    "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                    "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                ],
                [
                    #"% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                    #"% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                    "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT-)
                    "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                    "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                    "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT-)
                    "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                    "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                    "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT-)
                    "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                    "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
                ],
            ],
            "density": [
                [
                    "ILC1s - Density/mm²", # ILC1
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC2 - Density/mm²", # ILC2
                    "ILC3 - Density/mm²", # ILC3
                ],
                [
                    #"Early T-Cells - Density/mm²", # Immature T
                    #"Mature T-Cells - Density/mm²", # Mature T
                    "TH1 TdT- - Density/mm²", # TH1 (TdT-)
                    "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                    "Immature TH1 - Density/mm²", # Immature TH1
                    "TH2 TdT- - Density/mm²", # TH2 (TdT-)
                    "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                    "Immature TH2 - Density/mm²", # Immature TH2
                    "TH17 TdT- - Density/mm²", # TH17 (TdT-)
                    "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                    "Immature TH17 - Density/mm²", # Immature TH17
                ],
            ],
        },
    },
}

# Rename dataset columns
phenotypes_renamed_dict = {
    "_ST&TdT_THs_Layer1+Cortex+Medulla": {
        "percentage": [
            [
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "_ST&TdT_THs_Cortex+Medulla": {
        "percentage": [
            [
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla": {
        "percentage": [
            [
                #"Immature T",
                #"Mature T",
                "ILC1",
                "NK/ILC1ie",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "ILC2",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "ILC3",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                #"Immature T",
                #"Mature T",
                "NK/ILC1ie",
                "ILC1",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "ILC2",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "ILC3",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "_ST&TdT_ILCs+THs_Cortex+Medulla": {
        "percentage": [
            [
                #"Immature T",
                #"Mature T",
                "ILC1",
                "NK/ILC1ie",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "ILC2",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "ILC3",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                #"Immature T",
                #"Mature T",
                "NK/ILC1ie",
                "ILC1",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "ILC2",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "ILC3",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "_ALL_Merged_Layer1": {
        "multiplexMain": {
            "percentage": [
                [
                    #"T",
                    "ILC1",
                    "NK/ILC1ie",
                    "TH1",
                    "ILC2",
                    "TH2",
                    "ILC3",
                    "TH17",
                ],
            ],
            "density": [
                [
                    #"T",
                    "NK/ILC1ie",
                    "ILC1",
                    "TH1",
                    "ILC2",
                    "TH2",
                    "ILC3",
                    "TH17",
                ],
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                [
                    #"Immature T",
                    #"Mature T",
                    "ILC1",
                    "NK/ILC1ie",
                    "TH1 (TdT-)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                    "ILC2",
                    "TH2 (TdT-)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                    "ILC3",
                    "TH17 (TdT-)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
            "density": [
                [
                    #"Immature T",
                    #"Mature T",
                    "NK/ILC1ie",
                    "ILC1",
                    "TH1 (TdT-)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                    "ILC2",
                    "TH2 (TdT-)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                    "ILC3",
                    "TH17 (TdT-)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
        },
    },
    "_ALL_Splitted_Layer1": {
        "multiplexMain": {
            "percentage": [
                [
                    "ILC1",
                    "NK/ILC1ie",
                    "ILC2",
                    "ILC3",
                ],
                [
                    #"T",
                    "TH1",
                    "TH2",
                    "TH17",
                ],
            ],
            "density": [
                [
                    "NK/ILC1ie",
                    "ILC1",
                    "ILC2",
                    "ILC3",
                ],
                [
                    #"T",
                    "TH1",
                    "TH2",
                    "TH17",
                ],
            ],
        },
        "multiplexAlt#2": {
            "percentage": [
                [
                    "ILC1",
                    "NK/ILC1ie",
                    "ILC2",
                    "ILC3",
                ],
                [
                    #"Immature T",
                    #"Mature T",
                    "TH1 (TdT-)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                    "TH2 (TdT-)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                    "TH17 (TdT-)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
            "density": [
                [
                    "NK/ILC1ie",
                    "ILC1",
                    "ILC2",
                    "ILC3",
                ],
                [
                    #"Immature T",
                    #"Mature T",
                    "TH1 (TdT-)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                    "TH2 (TdT-)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                    "TH17 (TdT-)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
        },
    },
}

# Order dataset columns
phenotypes_ordered_dict = {
    "_ST&TdT_THs_Layer1+Cortex+Medulla": [
        [
            #"Immature T",
            #"Mature T",
            "TH1 (TdT-)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT-)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT-)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "_ST&TdT_THs_Cortex+Medulla": [
        [
            #"Immature T",
            #"Mature T",
            "TH1 (TdT-)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT-)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT-)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
            #"Immature T",
            #"Mature T",
            "TH1 (TdT-)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT-)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT-)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "_ST&TdT_ILCs+THs_Cortex+Medulla": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
            #"Immature T",
            #"Mature T",
            "TH1 (TdT-)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT-)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT-)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "_ALL_Merged_Layer1": {
        "multiplexMain": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
                #"T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "multiplexAlt#2": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "_ALL_Splitted_Layer1": {
        "multiplexMain": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                #"T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "multiplexAlt#2": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                #"Immature T",
                #"Mature T",
                "TH1 (TdT-)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT-)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT-)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
}

# HELPER FUNCTIONS
# For data processing and visualization
def rename_dataframe_cols(df, renamed_cols):
    """
    Rename columns in the dataframe for better readability.
    """

    df["Analysis Region"] = df["Analysis Region"].replace("Layer 1", "Whole tissue")
    df["Analysis Region"] = df["Analysis Region"].replace("GerminalCenter", "Germinal center") # Interface...
    df["Analysis Region"] = df["Analysis Region"].replace("GC Excluded", "Extra-follicular zone")
    df["Analysis Region"] = df["Analysis Region"].replace("WhitePulp", "White pulp") # Interface...
    df["Analysis Region"] = df["Analysis Region"].replace("RedPulp", "Red pulp")
    df["Analysis Region"] = df["Analysis Region"].replace("Medulla", "Medulla") # Interface...
    df["Analysis Region"] = df["Analysis Region"].replace("Cortex", "Cortex")
    df.columns = ["Analysis Region"] + renamed_cols + ["Case ID"]
    return df

def get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index):
    """
    Load and filter clinical data based on tissue codes and clinical indices.
    """

    # Local dataset path
    file_path = dirname + "/DATASETS/" + clinical_dataset + ".csv"

    #filter dataset columns
    reader_columns = ["Tissue Code", "Case ID"] + clinical_index

    # Define dataset columns type
    dtypes_col = {"Tissue Code": str, "Case ID": str, "Age": np.float64, "Age Range": str, "Sex": str, "Histology": str, "Histology Bool": str}

    # Read dataset and set indexes with panda
    df = None # !important reset dataframe before reading csv
    df = pd.read_csv(file_path, sep=",", usecols=reader_columns, dtype=dtypes_col)
    #print(len(df.index))

    # Filter dataset by tissue
    df_clinical = pd.DataFrame() # Empty df
    for idx, code in enumerate(tissue_code):
        if code in df["Tissue Code"].values: df_filtered = df[df["Tissue Code"].str.contains(code)]
        df_clinical = pd.concat([df_clinical, df_filtered])
    
    df_clinical = df_clinical.set_index(["Case ID"])
    #print(df_clinical.to_string()) # Get whole dataframe
    #print(len(df_clinical.index))

    # Return dataframe
    return df_clinical

def create_phenotype_dataframes(dataset, dataset_keys, input_file):
    """
    Create phenotype dataframes for clustering and visualization.
    """
    
    # Dataset key values
    data_type = dataset_keys[3]
    tissue_selection = dataset_keys[2]
    clinical_index = dataset_keys[1]
    tissue_code = dataset_keys[0]
    means = dataset_keys[4]

    # Set "value_name"
    if "_ST&TdT_" in tissue_selection:
        if data_type == "percentage": value_name = "Standardized Cell Percentage"
        elif data_type == "density": value_name = "Standardized Cell Density"
    elif "_ALL_" in tissue_selection:
        if data_type == "percentage": value_name = "Standardized\nCell Percentage"
        elif data_type == "density": value_name = "Standardized\nCell Density"

    # Define some color palettes
    sns_pal_paired_blue = sns.color_palette("Paired") # Pick a seaborn color palette...
    sns_pal_paired_green = sns.color_palette("PRGn_r") # Pick a seaborn color palette...
    sns_pal_qualitative_circular = sns.color_palette("tab10") # Pick a seaborn color palette...
    sns_pal_sequential_purple = sns.color_palette("Purples_r") # Pick a seaborn color palette...

    # Data depends on dataset and phenotypes
    if tissue_selection == "_ST&TdT_THs_Layer1+Cortex+Medulla":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for _each_dataset in dataset[data_type]:
            dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
            dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict = []
        for _each_dataset in dataset[data_type]:
            input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

            phenotypes_tuple = phenotypes_dict[tissue_selection][data_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][data_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                #filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Some data cleaning
                df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                df["Case ID"] = df["Case ID"].astype(str)
                df = df.drop(columns=["Image Tag"])

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)
                unique_case_ids = df["Case ID"].unique() # Get unique Case IDs (see below)...

                # Print dataframe for safety...
                #print(df)

                # Format dataframe for clustermaps
                df = df.set_index(["Case ID", "Analysis Region"])
                df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                
                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                if means:
                    # Compute means for some phenotypes columns
                    if phenotypes_ordered == ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["T", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["T", "TH1", "TH2", "TH17"]
                    elif phenotypes_ordered == ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["TH1", "TH2", "TH17"]

                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                # Populate dataframes dict
                multi_df_dict.append(df_grouped)

        # Concatenate dataframes         
        multi_df = pd.concat(multi_df_dict)
        #print(multi_df)

        # Be sure each Case IDs match
        df_clinical = df_clinical.reset_index()
        df_clinical.drop(df_clinical[~df_clinical["Case ID"].isin(unique_case_ids)].index, inplace = True)
        df_clinical = df_clinical.set_index(["Case ID"])
        
        # Print dataframe for safety...
        #print(df_clinical)

        # Make Case IDs index unique
        multi_df = multi_df.reset_index()
        multi_df.insert(0, "Unique ID", range(1001, 1001 + len(multi_df)))
        multi_df["Case ID"] = multi_df["Case ID"] + "_" + multi_df["Unique ID"].astype(str)
        multi_df = multi_df.drop(columns=["Unique ID"])

        df_clinical = df_clinical.reset_index()
        df_clinical.insert(0, "Unique ID", range(1001, 1001 + len(df_clinical)))
        df_clinical["Case ID"] = df_clinical["Case ID"] + "_" + df_clinical["Unique ID"].astype(str)
        df_clinical = df_clinical.drop(columns=["Unique ID"])

        # Re-index dataframe
        multi_df = multi_df.set_index(["Case ID"])
        df_clinical = df_clinical.set_index(["Case ID"])

        # Print dataframe for safety...
        #print(multi_df)
        #print(df_clinical)

        # Join dataframes
        joined_df = multi_df.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df.index))
        #print(joined_df)

        continue_script = True
        if joined_df.isnull().values.any(): 
            warnings.warn("There is some unmatched data...")
            continue_script = False

        if continue_script:
            # Prepare a vector of colors mapped to the clinical indices columns
            if clinical_index == ["Sex", "Age Range", "Histology Bool"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" and "Mature T" in joined_df.columns: continuous_data = ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)
            elif clinical_index == ["Sex", "Age Range"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)

                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" in joined_df.columns and "Mature T" in joined_df.columns: continuous_data = ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 3 # Init with 3 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

            # Keep only columns with numerical values (for clustermap)
            if means:
                if "T" in df_scaled.columns: df_scaled = df_scaled[["T", "TH1", "TH2", "TH17"]]
                else: df_scaled = df_scaled[["TH1", "TH2", "TH17"]]
            else:
                if "Immature T" in df_scaled.columns and "Mature T" in df_scaled.columns: df_scaled = df_scaled[["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]
                else: df_scaled = df_scaled[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]

            # Rename dataframe columns
            if not means: df_scaled.rename(columns={"TH1 (TdT-)": "TH1\nTdT–", "TH1 (TdT+)": "TH1\nTdT+", "Immature TH1": "Immature\nTH1", "TH2 (TdT-)": "TH2\nTdT–", "TH2 (TdT+)": "TH2\nTdT+", "Immature TH2": "Immature\nTH2", "TH17 (TdT-)": "TH17\nTdT–", "TH17 (TdT+)": "TH17\nTdT+", "Immature TH17": "Immature\nTH17"}, inplace=True)

            # Return dataframe
            yield value_name, df_scaled, row_colors
    elif tissue_selection == "_ST&TdT_THs_Cortex+Medulla":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for _each_dataset in dataset[data_type]:
            dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
            dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict = []
        for _each_dataset in dataset[data_type]:
            input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

            phenotypes_tuple = phenotypes_dict[tissue_selection][data_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][data_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                #filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Some data cleaning
                df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                df["Case ID"] = df["Case ID"].astype(str)
                df = df.drop(columns=["Image Tag"])

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)
                unique_case_ids = df["Case ID"].unique() # Get unique Case IDs (see below)...

                # Print dataframe for safety...
                #print(df)

                # Format dataframe for clustermaps
                df = df.set_index(["Case ID", "Analysis Region"])
                df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                
                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                if means:
                    # Compute means for some phenotypes columns
                    if phenotypes_ordered == ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["T", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["T", "TH1", "TH2", "TH17"]
                    elif phenotypes_ordered == ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["TH1", "TH2", "TH17"]

                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                # Populate dataframes dict
                multi_df_dict.append(df_grouped)

        # Concatenate dataframes         
        multi_df = pd.concat(multi_df_dict)
        #print(multi_df)

        # Be sure each Case IDs match
        df_clinical = df_clinical.reset_index()
        df_clinical.drop(df_clinical[~df_clinical["Case ID"].isin(unique_case_ids)].index, inplace = True)
        df_clinical = df_clinical.set_index(["Case ID"])
        
        # Print dataframe for safety...
        #print(df_clinical)

        # Make Case IDs index unique
        multi_df = multi_df.reset_index()
        multi_df.insert(0, "Unique ID", range(1001, 1001 + len(multi_df)))
        multi_df["Case ID"] = multi_df["Case ID"] + "_" + multi_df["Unique ID"].astype(str)
        multi_df = multi_df.drop(columns=["Unique ID"])

        df_clinical = df_clinical.reset_index()
        df_clinical.insert(0, "Unique ID", range(1001, 1001 + len(df_clinical)))
        df_clinical["Case ID"] = df_clinical["Case ID"] + "_" + df_clinical["Unique ID"].astype(str)
        df_clinical = df_clinical.drop(columns=["Unique ID"])

        # Re-index dataframe
        multi_df = multi_df.set_index(["Case ID"])
        df_clinical = df_clinical.set_index(["Case ID"])

        # Print dataframe for safety...
        #print(multi_df)
        #print(df_clinical)

        # Join dataframes
        joined_df = multi_df.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df.index))
        #print(joined_df)

        continue_script = True
        if joined_df.isnull().values.any(): 
            warnings.warn("There is some unmatched data...")
            continue_script = False

        if continue_script:
            # Prepare a vector of colors mapped to the clinical indices columns
            if clinical_index == ["Sex", "Age Range", "Histology Bool"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" and "Mature T" in joined_df.columns: continuous_data = ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)
            elif clinical_index == ["Sex", "Age Range"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)

                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" in joined_df.columns and "Mature T" in joined_df.columns: continuous_data = ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 3 # Init with 3 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

            # Keep only columns with numerical values (for clustermap)
            if means:
                if "T" in df_scaled.columns: df_scaled = df_scaled[["T", "TH1", "TH2", "TH17"]]
                else: df_scaled = df_scaled[["TH1", "TH2", "TH17"]]
            else:
                if "Immature T" in df_scaled.columns and "Mature T" in df_scaled.columns: df_scaled = df_scaled[["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]
                else: df_scaled = df_scaled[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]

            # Rename dataframe columns
            if not means: df_scaled.rename(columns={"TH1 (TdT-)": "TH1\nTdT–", "TH1 (TdT+)": "TH1\nTdT+", "Immature TH1": "Immature\nTH1", "TH2 (TdT-)": "TH2\nTdT–", "TH2 (TdT+)": "TH2\nTdT+", "Immature TH2": "Immature\nTH2", "TH17 (TdT-)": "TH17\nTdT–", "TH17 (TdT+)": "TH17\nTdT+", "Immature TH17": "Immature\nTH17"}, inplace=True)

            # Return dataframe
            yield value_name, df_scaled, row_colors
    elif tissue_selection == "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for _each_dataset in dataset[data_type]:
            dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
            dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict = []
        for _each_dataset in dataset[data_type]:
            input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

            phenotypes_tuple = phenotypes_dict[tissue_selection][data_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][data_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                #filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Some data cleaning
                df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                df["Case ID"] = df["Case ID"].astype(str)
                df = df.drop(columns=["Image Tag"])

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)
                unique_case_ids = df["Case ID"].unique() # Get unique Case IDs (see below)...

                # Print dataframe for safety...
                #print(df)

                # Format dataframe for clustermaps
                df = df.set_index(["Case ID", "Analysis Region"])
                df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                
                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                if means:
                    # Compute means for some phenotypes columns
                    if phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    elif phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]

                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                # Populate dataframes dict
                multi_df_dict.append(df_grouped)

        # Concatenate dataframes         
        multi_df = pd.concat(multi_df_dict)
        #print(multi_df)

        # Be sure each Case IDs match
        df_clinical = df_clinical.reset_index()
        df_clinical.drop(df_clinical[~df_clinical["Case ID"].isin(unique_case_ids)].index, inplace = True)
        df_clinical = df_clinical.set_index(["Case ID"])
        
        # Print dataframe for safety...
        #print(df_clinical)

        # Make Case IDs index unique
        multi_df = multi_df.reset_index()
        multi_df.insert(0, "Unique ID", range(1001, 1001 + len(multi_df)))
        multi_df["Case ID"] = multi_df["Case ID"] + "_" + multi_df["Unique ID"].astype(str)
        multi_df = multi_df.drop(columns=["Unique ID"])

        df_clinical = df_clinical.reset_index()
        df_clinical.insert(0, "Unique ID", range(1001, 1001 + len(df_clinical)))
        df_clinical["Case ID"] = df_clinical["Case ID"] + "_" + df_clinical["Unique ID"].astype(str)
        df_clinical = df_clinical.drop(columns=["Unique ID"])

        # Re-index dataframe
        multi_df = multi_df.set_index(["Case ID"])
        df_clinical = df_clinical.set_index(["Case ID"])

        # Print dataframe for safety...
        #print(multi_df)
        #print(df_clinical)

        # Join dataframes
        joined_df = multi_df.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df.index))
        #print(joined_df)

        continue_script = True
        if joined_df.isnull().values.any(): 
            warnings.warn("There is some unmatched data...")
            continue_script = False

        if continue_script:
            # Prepare a vector of colors mapped to the clinical indices columns
            if clinical_index == ["Sex", "Age Range", "Histology Bool"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" and "Mature T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)
            elif clinical_index == ["Sex", "Age Range"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)

                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" in joined_df.columns and "Mature T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 3 # Init with 3 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

            # Keep only columns with numerical values (for clustermap)
            if means:
                if "T" in df_scaled.columns: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]]
                else: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]]
            else:
                if "Immature T" in df_scaled.columns and "Mature T" in df_scaled.columns: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]
                else: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]

            # Rename dataframe columns
            if not means: df_scaled.rename(columns={"TH1 (TdT-)": "TH1\nTdT–", "TH1 (TdT+)": "TH1\nTdT+", "Immature TH1": "Immature\nTH1", "TH2 (TdT-)": "TH2\nTdT–", "TH2 (TdT+)": "TH2\nTdT+", "Immature TH2": "Immature\nTH2", "TH17 (TdT-)": "TH17\nTdT–", "TH17 (TdT+)": "TH17\nTdT+", "Immature TH17": "Immature\nTH17"}, inplace=True)

            # Return dataframe
            yield value_name, df_scaled, row_colors
    elif tissue_selection == "_ST&TdT_ILCs+THs_Cortex+Medulla":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for _each_dataset in dataset[data_type]:
            dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
            dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict = []
        for _each_dataset in dataset[data_type]:
            input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

            phenotypes_tuple = phenotypes_dict[tissue_selection][data_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][data_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                #filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Some data cleaning
                df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                df["Case ID"] = df["Case ID"].astype(str)
                df = df.drop(columns=["Image Tag"])

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)
                unique_case_ids = df["Case ID"].unique() # Get unique Case IDs (see below)...

                # Print dataframe for safety...
                #print(df)

                # Format dataframe for clustermaps
                df = df.set_index(["Case ID", "Analysis Region"])
                df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                
                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                if means:
                    # Compute means for some phenotypes columns
                    if phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    elif phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                        df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                        df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                        df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                        df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                        df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]

                # Print dataframe for safety...
                #print(len(df_grouped.index))
                #print(df_grouped)

                # Populate dataframes dict
                multi_df_dict.append(df_grouped)

        # Concatenate dataframes         
        multi_df = pd.concat(multi_df_dict)
        #print(multi_df)

        # Be sure each Case IDs match
        df_clinical = df_clinical.reset_index()
        df_clinical.drop(df_clinical[~df_clinical["Case ID"].isin(unique_case_ids)].index, inplace = True)
        df_clinical = df_clinical.set_index(["Case ID"])
        
        # Print dataframe for safety...
        #print(df_clinical)

        # Make Case IDs index unique
        multi_df = multi_df.reset_index()
        multi_df.insert(0, "Unique ID", range(1001, 1001 + len(multi_df)))
        multi_df["Case ID"] = multi_df["Case ID"] + "_" + multi_df["Unique ID"].astype(str)
        multi_df = multi_df.drop(columns=["Unique ID"])

        df_clinical = df_clinical.reset_index()
        df_clinical.insert(0, "Unique ID", range(1001, 1001 + len(df_clinical)))
        df_clinical["Case ID"] = df_clinical["Case ID"] + "_" + df_clinical["Unique ID"].astype(str)
        df_clinical = df_clinical.drop(columns=["Unique ID"])

        # Re-index dataframe
        multi_df = multi_df.set_index(["Case ID"])
        df_clinical = df_clinical.set_index(["Case ID"])

        # Print dataframe for safety...
        #print(multi_df)
        #print(df_clinical)

        # Join dataframes
        joined_df = multi_df.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df.index))
        #print(joined_df)

        continue_script = True
        if joined_df.isnull().values.any(): 
            warnings.warn("There is some unmatched data...")
            continue_script = False

        if continue_script:
            # Prepare a vector of colors mapped to the clinical indices columns
            if clinical_index == ["Sex", "Age Range", "Histology Bool"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" and "Mature T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)
            elif clinical_index == ["Sex", "Age Range"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Analysis Region"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Analysis Region"].unique())]))
                row_color = joined_df["Analysis Region"].map(row_color_palette)
                row_colors.append(row_color)

                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if means:
                    if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]
                else:
                    if "Immature T" in joined_df.columns and "Mature T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]
                    categorical_data = ["Analysis Region", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 3 # Init with 3 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Analysis Region": "STRUCTURE", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

            # Keep only columns with numerical values (for clustermap)
            if means:
                if "T" in df_scaled.columns: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]]
                else: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]]
            else:
                if "Immature T" in df_scaled.columns and "Mature T" in df_scaled.columns: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]
                else: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]]

            # Rename dataframe columns
            if not means: df_scaled.rename(columns={"TH1 (TdT-)": "TH1\nTdT–", "TH1 (TdT+)": "TH1\nTdT+", "Immature TH1": "Immature\nTH1", "TH2 (TdT-)": "TH2\nTdT–", "TH2 (TdT+)": "TH2\nTdT+", "Immature TH2": "Immature\nTH2", "TH17 (TdT-)": "TH17\nTdT–", "TH17 (TdT+)": "TH17\nTdT+", "Immature TH17": "Immature\nTH17"}, inplace=True)

            # Return dataframe
            yield value_name, df_scaled, row_colors
    elif tissue_selection == "_ALL_Merged_Layer1":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for multiplex in ["multiplexMain", "multiplexAlt#2"]:
            for _each_dataset in dataset[multiplex][data_type]:
                dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
                dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict = []
        for multiplex in ["multiplexMain", "multiplexAlt#2"]:
            for _each_dataset in dataset[multiplex][data_type]:
                input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

                phenotypes_tuple = phenotypes_dict[tissue_selection][multiplex][data_type]
                phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][multiplex][data_type]
                phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection][multiplex]

                for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                    #filter dataset columns
                    reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                    # Define dataset columns type
                    dtypes_col = {"Image Tag": str, "Analysis Region": str}
                    for pheno in phenotypes:
                        dtypes_col[pheno] = np.float64

                    # Read dataset and set indexes with panda
                    df = None # !important reset dataframe before reading csv
                    df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                    # Some data cleaning
                    df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                    df["Case ID"] = df["Case ID"].astype(str)
                    df = df.drop(columns=["Image Tag"])

                    # Rename columns
                    df = rename_dataframe_cols(df, phenotypes_renamed)

                    # Print dataframe for safety...
                    #print(df)

                    # Format dataframe for clustermaps
                    df = df.set_index(["Case ID", "Analysis Region"])
                    df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                    df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                    
                    # Print dataframe for safety...
                    #print(len(df_grouped.index))
                    #print(df_grouped)

                    # Compute means for some phenotypes columns
                    if (multiplex == "multiplexAlt#2"):
                        if phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                            df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                            df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                            df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                            df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                            df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                            df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"])

                            # Reset "phenotypes_ordered"
                            phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                        elif phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                            df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                            df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                            df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                            df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                            df_grouped = df_grouped.reindex(columns=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"])

                            # Reset "phenotypes_ordered"
                            phenotypes_ordered = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]

                    # Print dataframe for safety...
                    #print(len(df_grouped.index))
                    #print(df_grouped)

                    # Populate dataframes dict
                    multi_df_dict.append(df_grouped)

        # Concatenate dataframes         
        multi_df = pd.concat(multi_df_dict)
        #print(multi_df)

        # Join dataframes
        joined_df = multi_df.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df.index))
        #print(joined_df)

        continue_script = True
        if joined_df.isnull().values.any(): 
            warnings.warn("There is some unmatched data...")
            continue_script = False

        if continue_script:
            # Prepare a vector of colors mapped to the clinical indices columns
            if clinical_index == ["Sex", "Age Range", "Histology Bool"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Tissue Code"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Tissue Code"].unique())]))
                row_color = joined_df["Tissue Code"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                categorical_data = ["Tissue Code", "Age Range", "Sex", "Histology Bool"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Tissue Code": "TISSUE TYPE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)
            elif clinical_index == ["Sex", "Age Range"]:
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Tissue Code"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Tissue Code"].unique())]))
                row_color = joined_df["Tissue Code"].map(row_color_palette)
                row_colors.append(row_color)
                
                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_1_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                if "T" in joined_df.columns: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]
                else: continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]
                categorical_data = ["Tissue Code", "Age Range", "Sex"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]
                
                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Tissue Code": "TISSUE TYPE", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

            # Keep only columns with numerical values (for clustermap)
            if "T" in df_scaled.columns: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "T", "TH1", "TH2", "TH17"]]
            else: df_scaled = df_scaled[["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"]]

            # Return dataframe
            yield value_name, df_scaled, row_colors
    elif tissue_selection == "_ALL_Splitted_Layer1":
        # Get clinical data (+ filter clinical dataframe)
        df_clinical = get_clinical_dataframe_rows(clinical_dataset, tissue_code, clinical_index)
        #print(len(df_clinical.index))

        dataset_tissue_codes = []
        for multiplex in ["multiplexMain", "multiplexAlt#2"]:
            for _each_dataset in dataset[multiplex][data_type]:
                dataset_tissue_code = _each_dataset[0:2] # Filter by dataset tisue code
                dataset_tissue_codes.append(dataset_tissue_code)

        df_clinical_multiple = []
        for dataset_tissue_code in dataset_tissue_codes:
            if dataset_tissue_code in df_clinical["Tissue Code"].values: df_clinical_filtered = df_clinical[df_clinical["Tissue Code"].str.contains(dataset_tissue_code)]
            df_clinical_multiple.append(df_clinical_filtered)

        df_clinical = pd.concat(df_clinical_multiple)
        #df_clinical = df_clinical.drop(columns=["Tissue Code"]) # Keep tissue code for row colors

        # Print dataframe for safety...
        #print(df_clinical.to_string()) # Get whole dataframe

        multi_df_dict_ths = []
        multi_df_dict_ilcs = []
        for multiplex in ["multiplexMain", "multiplexAlt#2"]:
            for _each_dataset in dataset[multiplex][data_type]:
                input_file = dirname + "/DATASETS/" + _each_dataset + ".csv"

                phenotypes_tuple = phenotypes_dict[tissue_selection][multiplex][data_type]
                phenotypes_renamed_tuple = phenotypes_renamed_dict[tissue_selection][multiplex][data_type]
                phenotypes_ordered_tuple = phenotypes_ordered_dict[tissue_selection][multiplex]

                for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                    #filter dataset columns
                    reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                    # Define dataset columns type
                    dtypes_col = {"Image Tag": str, "Analysis Region": str}
                    for pheno in phenotypes:
                        dtypes_col[pheno] = np.float64

                    # Read dataset and set indexes with panda
                    df = None # !important reset dataframe before reading csv
                    df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                    # Some data cleaning
                    df["Case ID"] = df["Image Tag"].apply(lambda x : re.match(r"\d{6}_.*_(\d{2}[A-Z]\d{6})", x)[1])
                    df["Case ID"] = df["Case ID"].astype(str)
                    df = df.drop(columns=["Image Tag"])

                    # Rename columns
                    df = rename_dataframe_cols(df, phenotypes_renamed)

                    # Print dataframe for safety...
                    #print(df)

                    # Format dataframe for clustermaps
                    df = df.set_index(["Case ID", "Analysis Region"])
                    df_grouped = df.groupby(by=["Case ID", "Analysis Region"], dropna=True).mean()
                    df_grouped = df_grouped.reindex(columns=phenotypes_ordered)
                    
                    # Print dataframe for safety...
                    #print(len(df_grouped.index))
                    #print(df_grouped)

                    # Compute means for some phenotypes columns
                    if (multiplex == "multiplexAlt#2"):
                        if phenotypes_ordered == ["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                            df_grouped["T"] = df_grouped[["Immature T", "Mature T"]].mean(axis=1).round(8)
                            df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                            df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                            df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                            df_grouped = df_grouped.drop(columns=["Immature T", "Mature T", "TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                            df_grouped = df_grouped.reindex(columns=["T", "TH1", "TH2", "TH17"])

                            # Reset "phenotypes_ordered"
                            phenotypes_ordered = ["T", "TH1", "TH2", "TH17"]
                        elif phenotypes_ordered == ["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]:
                            df_grouped["TH1"] = df_grouped[["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                            df_grouped["TH2"] = df_grouped[["TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                            df_grouped["TH17"] = df_grouped[["TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                            df_grouped = df_grouped.drop(columns=["TH1 (TdT-)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT-)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT-)", "TH17 (TdT+)", "Immature TH17"])
                            df_grouped = df_grouped.reindex(columns=["TH1", "TH2", "TH17"])

                            # Reset "phenotypes_ordered"
                            phenotypes_ordered = ["TH1", "TH2", "TH17"]

                    # Print dataframe for safety...
                    #print(len(df_grouped.index))
                    #print(df_grouped)

                    # Populate dataframes dict
                    if phenotypes_ordered == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]: multi_df_dict_ilcs.append(df_grouped)
                    elif phenotypes_ordered == ["T", "TH1", "TH2", "TH17"] or phenotypes_ordered == ["TH1", "TH2", "TH17"]: multi_df_dict_ths.append(df_grouped)

        # Concatenate dataframes         
        multi_df_ilcs = pd.concat(multi_df_dict_ilcs)
        multi_df_ths = pd.concat(multi_df_dict_ths)
        #print(multi_df_ilcs)
        #print(multi_df_ths)

        # Join dataframes
        joined_df_ilcs = multi_df_ilcs.join(df_clinical)
        joined_df_ths = multi_df_ths.join(df_clinical)

        # Print dataframe for safety...
        #print(len(joined_df_ilcs.index))
        #print(len(joined_df_ths.index))
        #print(joined_df_ilcs)
        #print(joined_df_ths)

        for idx, joined_df in enumerate([joined_df_ilcs, joined_df_ths]):
            continue_script = True
            if joined_df.isnull().values.any(): 
                warnings.warn("There is some unmatched data...")
                continue_script = False

            if continue_script:
                # Prepare a vector of colors mapped to the clinical indices columns
                row_colors = []
                for c_idx in clinical_index:
                    if (c_idx == "Sex"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_green.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Age Range"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_sequential_purple.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)
                    elif (c_idx == "Histology Bool"):
                        row_color_palette = dict(zip(joined_df[c_idx].unique(), sns_pal_paired_blue.as_hex()[0:len(joined_df[c_idx].unique())]))
                        row_color = joined_df[c_idx].map(row_color_palette)
                        row_colors.append(row_color)

                row_color_palette = dict(zip(joined_df["Tissue Code"].unique(), sns_pal_qualitative_circular.as_hex()[0:len(joined_df["Tissue Code"].unique())]))
                row_color = joined_df["Tissue Code"].map(row_color_palette)
                row_colors.append(row_color)

                # Export dataframe to csv
                output_dir = dirname + "/OUTPUT"
                output_filename = "dataframe_{}".format(idx + 1) + "_" + data_type
                outpath = os.path.join(output_dir, output_filename + ".csv")
                #joined_df.to_csv(outpath)

                # Describe dataframe (try to understand data)
                # Get continuous and categorical data (for dataframe description)
                categorical_data = ["Tissue Code", "Age Range", "Sex", "Histology Bool"]
                if idx == 0:
                    continuous_data = ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]
                elif idx == 1:
                    if "T" in joined_df.columns: continuous_data = ["T", "TH1", "TH2", "TH17"]
                    else: continuous_data = ["TH1", "TH2", "TH17"]

                if describe_data:
                    describe_dataframe(joined_df, continuous_data, categorical_data, True, True)

                # Scale dataframe (try to get a gaussian distribution)
                df_scaled = scale_dataframe(joined_df, continuous_data, "PowerTransformer", False)

                # Clustering analysis
                n_clusters = 4 # Init with 4 clusters...
                clustering_analyis(df_scaled, clustering_analysis_print_output, clustering_analysis_show_plot) # Find the number of clusters
                clusters = pca_analyis(df_scaled, len(continuous_data), n_clusters, pca_analysis_show_plots) # Get clusters

                # Add clusters to "row_colors"
                row_color_palette = dict(zip(np.unique(clusters), "rgby"))
                row_color = pd.Series(clusters, name="Clusters", index=df_scaled.index).map(row_color_palette)
                row_colors.append(row_color)

                # Reorder "row_colors"
                row_colors_order =[4, 3, 2, 0, 1]
                row_colors = [row_colors[i] for i in row_colors_order]

                row_colors = pd.concat(row_colors, axis=1) # Must be a dataframe (not a list) to get the labels in the clustermap...
                row_colors.rename(columns={"Clusters": "CLUSTERS", "Tissue Code": "TISSUE TYPE", "Histology Bool": "HISTOLOGY", "Age Range": "AGE", "Sex": "GENDER"}, inplace=True)

                # Keep only columns with numerical values (for clustermap)
                if idx == 0:
                    joined_df = joined_df[["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]]
                elif idx == 1:
                    if "T" in joined_df.columns: joined_df = joined_df[["T", "TH1", "TH2", "TH17"]]
                    else: joined_df = joined_df[["TH1", "TH2", "TH17"]]

                # Return dataframe
                yield value_name, joined_df, row_colors

def pca_analyis(df_scaled, n_components, n_clusters, show_plot=False):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = KMeans(n_clusters=n_clusters, random_state=5)
        y = model.fit_predict(df_scaled)

    pca = PCA(n_components=n_components, svd_solver="full").fit(df_scaled)
    pca_df = pd.DataFrame(pca.transform(df_scaled))
    pca_df.columns = [f"PC_{i}" for i in pca_df.columns]

    if show_plot:
        # Plot PCA
        plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)

        plt.title("PCA Analysis")

        plt.tight_layout()
        plt.show()

        # Plot PCA 3D
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        xdata = pca_df.iloc[:,0]
        ydata = pca_df.iloc[:,1]
        zdata = pca_df.iloc[:,2]

        ax.scatter3D(xdata, ydata, zdata, c=y, cmap="tab10_r")

        plt.title("3D PCA Analysis")

        plt.tight_layout()
        plt.show()

    return y

def clustering_analyis(df_scaled, print_output=False, show_plot=False):
    from sklearn.cluster import KMeans
    from collections import Counter

    # Within-Cluster Sum of Square
    WCSS = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(1, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(df_scaled)
            WCSS.append(model.inertia_)

        for j in range(3, 6): # Range defined from WCSS plot (curve inflection points)
            if print_output: print(f"--{j}--")

            for k in range(1, 10):
                model = KMeans(n_clusters=j, random_state=k)
                y = model.fit_predict(df_scaled)
                if print_output: print(Counter(y))

    if show_plot:
        plt.plot(range(1, 11), WCSS)
        plt.xlabel("cluster")
        plt.ylabel("WCSS")

        plt.title("Clustering Analysis")

        plt.tight_layout()
        plt.show()

def scale_dataframe(df, continuous_data, preprocessing="StandardScaler", show_plot=False):
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    X = df[continuous_data]
    X_offset = X + 0.01

    if preprocessing == "StandardScaler":
        scaler = StandardScaler().fit(X)
        X_scaled = pd.DataFrame(scaler.transform(X))
        X_scaled.columns = X.columns # Set column names
    elif preprocessing == "MinMaxScaler":
        scaler = MinMaxScaler().fit(X)
        X_scaled = pd.DataFrame(scaler.transform(X))
        X_scaled.columns = X.columns # Set column names
    elif preprocessing == "PowerTransformer":
        scaler = PowerTransformer(method="box-cox")
        X_scaled = pd.DataFrame(scaler.fit_transform(X_offset))
        X_scaled.columns = X.columns # Set column names

    if show_plot:
        X_scaled.hist() # Print histograms
        plt.tight_layout()
        plt.show()

    X_scaled = X_scaled.set_index(X.index)

    return X_scaled

def describe_dataframe(df, continuous_data, categorical_data, show_swarmplot=False, categorical_count=False):
    print(df.describe()) # Get main statistics

    df.hist() # Print histograms
    plt.tight_layout()
    plt.show()

    if show_swarmplot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.swarmplot(x="variable", y="value", data=pd.melt(df, value_vars=continuous_data), size=2.5, log_scale=True)
        plt.xticks(rotation=90)

        plt.title("Swarmplot of Continuous Data")
        
        plt.tight_layout()
        plt.show()

    if categorical_count:
        sns.countplot(x="value", data=pd.melt(df, value_vars=categorical_data))
        plt.xticks(rotation=90)

        plt.title("Countplot of Categorical Data")

        plt.tight_layout()
        plt.show()

def plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends=False, save=False):
    # See: https://seaborn.pydata.org/generated/seaborn.clustermap.html / https://youtu.be/crQkHHhY7aY?feature=shared

    # Dataset key values
    data_type = dataset_keys[3]
    tissue_selection = dataset_keys[2]
    clinical_index = dataset_keys[1]
    tissue_code = dataset_keys[0]

    # Set aesthetic parameters
    sns.set_context("paper")
    sns.set(font_scale=fontscale)

    # Get fonts as font in postscript filetypes
    plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]

    if tissue_selection == "_ST&TdT_THs_Layer1+Cortex+Medulla":
        # Style y axis left ticks
        plt.rcParams["ytick.right"] = False
        plt.rcParams["xtick.bottom"] = False
        plt.rcParams["ytick.color"] = axe_label_color
        plt.rcParams["xtick.color"] = axe_label_color

        plt.rcParams["ytick.major.size"] = 0
        plt.rcParams["ytick.major.width"] = 0

        # Figure shaping
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        figsize = (18*cm, 5.5*cm)
        col_cluster = True
        row_cluster = True
        annot = False
        cmap = "OrRd"

        # Custom stuffs
        colors_ratio = 0.0125
        cbar_kws = {"orientation": "horizontal"}
        #cbar_pos = None #(0.02, 0.8, 0.05, 0.18)
        cbar_pos = (0.02, 0.8, 0.05, 0.18)
        tight_layout_h_pad = -0.45
        tight_layout_w_pad = 0.1

        # Standardize and/or normalize across columns or rows (parameters mutually exclusive)
        standard_scale = 1 # "0" => across rows / "1" => across columns
        z_score = 0 # "0" => across rows / "1" => across columns

        # Metric and method
        metric = "euclidean" # "correlation"
        method = "ward" # "single"
    else:
        if tissue_selection == "_ST&TdT_THs_Cortex+Medulla":
            # Style y axis left ticks
            plt.rcParams["ytick.right"] = False
            plt.rcParams["xtick.bottom"] = False
            plt.rcParams["ytick.color"] = axe_label_color
            plt.rcParams["xtick.color"] = axe_label_color

            plt.rcParams["ytick.major.size"] = 0
            plt.rcParams["ytick.major.width"] = 0

            # Figure shaping
            cm = 1/2.54  # Centimeters in inches (for "figsize")
            figsize = (18*cm, 4.5*cm)
            col_cluster = True
            row_cluster = True
            annot = False
            cmap = "OrRd"

            # Custom stuffs
            colors_ratio = 0.0125
            cbar_kws = {"orientation": "horizontal"}
            #cbar_pos = None #(0.02, 0.8, 0.05, 0.18)
            cbar_pos = (0.02, 0.8, 0.05, 0.18)
            tight_layout_h_pad = -0.65
            tight_layout_w_pad = 0.1

            # Standardize and/or normalize across columns or rows (parameters mutually exclusive)
            standard_scale = 1 # "0" => across rows / "1" => across columns
            z_score = 0 # "0" => across rows / "1" => across columns

            # Metric and method
            metric = "euclidean" # "correlation"
            method = "ward" # "single"
        else:
            if tissue_selection == "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla":
                # Style y axis left ticks
                plt.rcParams["ytick.right"] = False
                plt.rcParams["xtick.bottom"] = False
                plt.rcParams["ytick.color"] = axe_label_color
                plt.rcParams["xtick.color"] = axe_label_color

                plt.rcParams["ytick.major.size"] = 0
                plt.rcParams["ytick.major.width"] = 0

                # Figure shaping
                cm = 1/2.54  # Centimeters in inches (for "figsize")
                figsize = (18*cm, 5.5*cm)
                col_cluster = True
                row_cluster = True
                annot = False
                cmap = "OrRd"

                # Custom stuffs
                colors_ratio = 0.0125
                cbar_kws = {"orientation": "horizontal"}
                #cbar_pos = None #(0.02, 0.8, 0.05, 0.18)
                cbar_pos = (0.02, 0.8, 0.05, 0.18)
                tight_layout_h_pad = -0.45
                tight_layout_w_pad = 0.1

                # Standardize and/or normalize across columns or rows (parameters mutually exclusive)
                standard_scale = 1 # "0" => across rows / "1" => across columns
                z_score = 0 # "0" => across rows / "1" => across columns

                # Metric and method
                metric = "euclidean" # "correlation"
                method = "ward" # "single"
            else:
                if tissue_selection == "_ST&TdT_ILCs+THs_Cortex+Medulla":
                    # Style y axis left ticks
                    plt.rcParams["ytick.right"] = False
                    plt.rcParams["xtick.bottom"] = False
                    plt.rcParams["ytick.color"] = axe_label_color
                    plt.rcParams["xtick.color"] = axe_label_color

                    plt.rcParams["ytick.major.size"] = 0
                    plt.rcParams["ytick.major.width"] = 0

                    # Figure shaping
                    cm = 1/2.54  # Centimeters in inches (for "figsize")
                    figsize = (18*cm, 4.5*cm)
                    col_cluster = True
                    row_cluster = True
                    annot = False
                    cmap = "OrRd"

                    # Custom stuffs
                    colors_ratio = 0.0125
                    cbar_kws = {"orientation": "horizontal"}
                    #cbar_pos = None #(0.02, 0.8, 0.05, 0.18)
                    cbar_pos = (0.02, 0.8, 0.05, 0.18)
                    tight_layout_h_pad = -0.65
                    tight_layout_w_pad = 0.1

                    # Standardize and/or normalize across columns or rows (parameters mutually exclusive)
                    standard_scale = 1 # "0" => across rows / "1" => across columns
                    z_score = 0 # "0" => across rows / "1" => across columns

                    # Metric and method
                    metric = "euclidean" # "correlation"
                    method = "ward" # "single"
                else:
                    # Style y axis left ticks
                    plt.rcParams["ytick.right"] = False
                    plt.rcParams["xtick.bottom"] = False
                    plt.rcParams["ytick.color"] = axe_label_color
                    plt.rcParams["xtick.color"] = axe_label_color

                    plt.rcParams["ytick.major.size"] = 0
                    plt.rcParams["ytick.major.width"] = 0

                    # Figure shaping
                    cm = 1/2.54  # Centimeters in inches (for "figsize")
                    figsize = (18*cm, 10*cm)
                    col_cluster = True
                    row_cluster = True
                    annot = False
                    cmap = "OrRd"

                    # Custom stuffs
                    colors_ratio = 0.0125
                    cbar_kws = {"orientation": "vertical"}
                    #cbar_pos = None #(0.02, 0.8, 0.05, 0.18)
                    cbar_pos = (0.02, 0.8, 0.05, 0.18)
                    tight_layout_h_pad = -0.75
                    tight_layout_w_pad = 0.1

                    # Standardize and/or normalize across columns or rows (parameters mutually exclusive)
                    standard_scale = 1 # "0" => across rows / "1" => across columns
                    z_score = 0 # "0" => across rows / "1" => across columns

                    # Metric and method
                    metric = "euclidean" # "correlation"
                    method = "ward" # "single"

    # Metric => See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    # Method => See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    sns_plot = sns.clustermap(df, method=method, metric=metric, standard_scale=standard_scale, col_cluster=col_cluster, row_cluster=row_cluster, row_colors=row_colors, colors_ratio=colors_ratio, cmap=cmap, annot=annot, linewidths=.25, figsize=figsize, xticklabels=1, yticklabels=False, cbar_pos=cbar_pos, cbar_kws=cbar_kws)
    #sns_plot = sns.heatmap(df, cmap=cmap)

    # Custom draw settings
    axes = sns_plot.ax_heatmap

    if tissue_selection == "_ST&TdT_THs_Layer1+Cortex+Medulla":
        # Customize row color labels
        # => https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
        # => https://copyprogramming.com/howto/additional-row-colors-in-seaborn-cluster-map

        ax_row_colors = sns_plot.ax_row_colors
        ax_row_colors.tick_params(axis="x", labelsize=labelsize-2.5, labelcolor=axe_label_color, pad=-3)

        # Customize other plot parameters
        sns_plot.ax_cbar.set_xlabel(value_name, size=0, color="#ffffff")
        sns_plot.ax_cbar.set_title(value_name, size=labelsize-1.5, pad=2.8)
        sns_plot.ax_cbar.tick_params(axis='x', length=0, labelsize=labelsize-2)
        sns_plot.ax_cbar.set_aspect(0.035)

        xlabel_rotation = 0 # Horizontal x labels

        axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
        axes.tick_params(axis="x", labelsize=labelsize-1.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
        axes.set_ylabel("CASE ID", size=labelsize, weight="bold", labelpad=3)
        axes.set_xlabel("")
    else:
        if tissue_selection == "_ST&TdT_THs_Cortex+Medulla":
            # Customize row color labels
            # => https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
            # => https://copyprogramming.com/howto/additional-row-colors-in-seaborn-cluster-map

            ax_row_colors = sns_plot.ax_row_colors
            ax_row_colors.tick_params(axis="x", labelsize=labelsize-2.5, labelcolor=axe_label_color, pad=-3)

            # Customize other plot parameters
            sns_plot.ax_cbar.set_xlabel(value_name, size=0, color="#ffffff")
            sns_plot.ax_cbar.set_title(value_name, size=labelsize-1.5, pad=2.8)
            sns_plot.ax_cbar.tick_params(axis='x', length=0, labelsize=labelsize-2)
            sns_plot.ax_cbar.set_aspect(0.035)

            xlabel_rotation = 0 # Horizontal x labels

            axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
            axes.tick_params(axis="x", labelsize=labelsize-1.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
            axes.set_ylabel("CASE ID", size=labelsize, weight="bold", labelpad=3)
            axes.set_xlabel("")
        else:
            if tissue_selection == "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla":
                # Customize row color labels
                # => https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
                # => https://copyprogramming.com/howto/additional-row-colors-in-seaborn-cluster-map

                ax_row_colors = sns_plot.ax_row_colors
                ax_row_colors.tick_params(axis="x", labelsize=labelsize-2.5, labelcolor=axe_label_color, pad=-3)

                # Customize other plot parameters
                sns_plot.ax_cbar.set_xlabel(value_name, size=0, color="#ffffff")
                sns_plot.ax_cbar.set_title(value_name, size=labelsize-1.5, pad=2.8)
                sns_plot.ax_cbar.tick_params(axis='x', length=0, labelsize=labelsize-2)
                sns_plot.ax_cbar.set_aspect(0.035)

                xlabel_rotation = 0 # Horizontal x labels

                axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                axes.tick_params(axis="x", labelsize=labelsize-1.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                axes.set_ylabel("CASE ID", size=labelsize, weight="bold", labelpad=3)
                axes.set_xlabel("")
            else:
                if tissue_selection == "_ST&TdT_ILCs+THs_Cortex+Medulla":
                    # Customize row color labels
                    # => https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
                    # => https://copyprogramming.com/howto/additional-row-colors-in-seaborn-cluster-map

                    ax_row_colors = sns_plot.ax_row_colors
                    ax_row_colors.tick_params(axis="x", labelsize=labelsize-2.5, labelcolor=axe_label_color, pad=-3)

                    # Customize other plot parameters
                    sns_plot.ax_cbar.set_xlabel(value_name, size=0, color="#ffffff")
                    sns_plot.ax_cbar.set_title(value_name, size=labelsize-1.5, pad=2.8)
                    sns_plot.ax_cbar.tick_params(axis='x', length=0, labelsize=labelsize-2)
                    sns_plot.ax_cbar.set_aspect(0.035)

                    xlabel_rotation = 0 # Horizontal x labels

                    axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes.tick_params(axis="x", labelsize=labelsize-1.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    axes.set_ylabel("CASE ID", size=labelsize, weight="bold", labelpad=3)
                    axes.set_xlabel("")
                else:
                    # Customize row color labels
                    # => https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
                    # => https://copyprogramming.com/howto/additional-row-colors-in-seaborn-cluster-map

                    ax_row_colors = sns_plot.ax_row_colors
                    ax_row_colors.tick_params(axis="x", labelsize=labelsize-2.5, labelcolor=axe_label_color, pad=-3)

                    # Customize other plot parameters
                    sns_plot.ax_cbar.set_ylabel(value_name, size=labelsize-1.5)
                    sns_plot.ax_cbar.tick_params(labelsize=labelsize-2)
                    sns_plot.ax_cbar.set_aspect(10)

                    xlabel_rotation = 0 # Horizontal x labels
                    
                    axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes.tick_params(axis="x", labelsize=labelsize, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    axes.set_ylabel("CASE ID", size=labelsize, weight="bold", labelpad=3)
                    axes.set_xlabel("")

    # Update outpath with clustermap method and metric
    outpath_leftpart = outpath.split("__")[0]
    outpath_rightpart = outpath.split("__")[1]
    outpath = outpath_leftpart + "__" + metric + "_" + method + "_" + outpath_rightpart

    # Custom legends
    if custom_legends:
        for categorical_var in list(row_colors.columns):
            xx = [] # Initialize an empty "artists" list...
            for color in row_colors[categorical_var].unique():
                x = sns_plot.ax_row_dendrogram.bar(0, 0, color=color, label=categorical_var, linewidth=0)
                xx.append(x)

            # Full manual implementation...
            if tissue_selection == "_ST&TdT_THs_Layer1+Cortex+Medulla":
                if categorical_var == "AGE":
                    items = ["0-25", "26-50", "51-75", "76+"]
                    xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.116), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                    plt.gca().add_artist(xxx)
                elif categorical_var == "GENDER":
                    items = ["MALE", "FEMALE"]
                    xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.088), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                    plt.gca().add_artist(xxx)
                elif categorical_var == "STRUCTURE":
                    items = ["WHOLE TISSUE", "CORTEX", "MEDULLA"]
                    xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.060), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                    plt.gca().add_artist(xxx)
            else:
                if tissue_selection == "_ST&TdT_THs_Cortex+Medulla":
                    if categorical_var == "AGE":
                        items = ["0-25", "26-50", "51-75", "76+"]
                        xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.130), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                        plt.gca().add_artist(xxx)
                    elif categorical_var == "GENDER":
                        items = ["MALE", "FEMALE"]
                        xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.098), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                        plt.gca().add_artist(xxx)
                    elif categorical_var == "STRUCTURE":
                        items = ["CORTEX", "MEDULLA"]
                        xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.066), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                        plt.gca().add_artist(xxx)
                else:
                    if tissue_selection == "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla":
                        if categorical_var == "AGE":
                            items = ["0-25", "26-50", "51-75", "76+"]
                            xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.116), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                            plt.gca().add_artist(xxx)
                        elif categorical_var == "GENDER":
                            items = ["MALE", "FEMALE"]
                            xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.088), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                            plt.gca().add_artist(xxx)
                        elif categorical_var == "STRUCTURE":
                            items = ["WHOLE TISSUE", "CORTEX", "MEDULLA"]
                            xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.060), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                            plt.gca().add_artist(xxx)
                    else:
                        if tissue_selection == "_ST&TdT_ILCs+THs_Cortex+Medulla":
                            if categorical_var == "AGE":
                                items = ["0-25", "26-50", "51-75", "76+"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.130), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
                            elif categorical_var == "GENDER":
                                items = ["MALE", "FEMALE"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.098), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
                            elif categorical_var == "STRUCTURE":
                                items = ["CORTEX", "MEDULLA"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.066), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
                        else:
                            if categorical_var == "TISSUE TYPE":
                                items = ["TONSIL", "APPENDIX", "ILEUM", "LYMPH NODE", "SPLEEN", "THYMUS"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.074), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
                            elif categorical_var == "AGE":
                                items = ["0-25", "26-50", "51-75", "76+"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.058), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
                            elif categorical_var == "GENDER":
                                items = ["MALE", "FEMALE"]
                                xxx = plt.legend(xx, items, ncol=len(items), title="", shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, loc="center", bbox_to_anchor=(0.615, 0.042), bbox_transform=plt.gcf().transFigure, title_fontsize=labelsize, fontsize=labelsize-2.5, labelcolor=axe_label_color)
                                plt.gca().add_artist(xxx)
    
    if save:
        plt.tight_layout(h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
        sns_plot.figure.savefig(outpath, dpi=300, format=extension)
    else:
        plt.tight_layout(h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
        plt.show()

# DRAW FUNCTION
def draw(dataset_dict, dataset_keys, custom_legends=False, save=False):
    """
    Main function to generate clustermaps for the given datasets.
    """
    
    # Dataset key values
    data_type = dataset_keys[3]
    tissue_selection = dataset_keys[2]
    
    # Draw...
    if tissue_selection == "_ST&TdT_THs_Layer1+Cortex+Medulla":
        # Local dataset path
        dataset = dataset_dict[tissue_selection]
        for phenotype_dataframes in create_phenotype_dataframes(dataset, dataset_keys, False):
            value_name, df, row_colors = phenotype_dataframes

            # Output directory + filename + path
            output_dir = dirname + "/OUTPUT"
            output_filename = "clustermap_" + tissue_selection + "_" + data_type
            outpath = os.path.join(output_dir, output_filename + "." + extension)

            plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
    else:
        if tissue_selection == "_ST&TdT_THs_Cortex+Medulla":
            # Local dataset path
            dataset = dataset_dict[tissue_selection]
            for phenotype_dataframes in create_phenotype_dataframes(dataset, dataset_keys, False):
                value_name, df, row_colors = phenotype_dataframes

                # Output directory + filename + path
                output_dir = dirname + "/OUTPUT"
                output_filename = "clustermap_" + tissue_selection + "_" + data_type
                outpath = os.path.join(output_dir, output_filename + "." + extension)

                plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
        else:
            if tissue_selection == "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla":
                # Local dataset path
                dataset = dataset_dict[tissue_selection]
                for phenotype_dataframes in create_phenotype_dataframes(dataset, dataset_keys, False):
                    value_name, df, row_colors = phenotype_dataframes

                    # Output directory + filename + path
                    output_dir = dirname + "/OUTPUT"
                    output_filename = "clustermap_" + tissue_selection + "_" + data_type
                    outpath = os.path.join(output_dir, output_filename + "." + extension)

                    plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
            else:
                if tissue_selection == "_ST&TdT_ILCs+THs_Cortex+Medulla":
                    # Local dataset path
                    dataset = dataset_dict[tissue_selection]
                    for phenotype_dataframes in create_phenotype_dataframes(dataset, dataset_keys, False):
                        value_name, df, row_colors = phenotype_dataframes

                        # Output directory + filename + path
                        output_dir = dirname + "/OUTPUT"
                        output_filename = "clustermap_" + tissue_selection + "_" + data_type
                        outpath = os.path.join(output_dir, output_filename + "." + extension)

                        plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
                else:
                    if tissue_selection == "_ALL_Merged_Layer1":
                        # Local dataset path
                        dataset = dataset_dict[tissue_selection]
                        for phenotype_dataframes in create_phenotype_dataframes(dataset, dataset_keys, False):
                            value_name, df, row_colors = phenotype_dataframes

                            # Output directory + filename + path
                            output_dir = dirname + "/OUTPUT"
                            output_filename = "clustermap_" + tissue_selection + "_" + data_type
                            outpath = os.path.join(output_dir, output_filename + "." + extension)

                            # Output directory (csv)
                            source_data = 'sourceData_'
                            csv_output_filename = f"{source_data}" + output_filename
                            csv_outpath = os.path.join(output_dir, csv_output_filename + ".csv")

                            # Get data source...
                            if _get_source_data:
                                df_csv = df.dropna(subset=["NK/ILC1ie", "ILC1", "ILC2", "ILC3", "TH1", "TH2", "TH17"])

                                df_csv = df_csv.reset_index(drop=True)
                                df_csv.index.name = "Case ID"
                                                                
                                #print(df_csv)

                                df_csv.to_csv(csv_outpath, sep=",", encoding="utf-8", index=True, header=True) # Use "na_rep='NaN'" to keep NaN values in the CSV file 
                            # ----------------------------------------------------------------------------------------------------------------------------------------

                            plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
                    else:
                        if tissue_selection == "_ALL_Splitted_Layer1":
                            # Local dataset path
                            dataset = dataset_dict[tissue_selection]
                            for idx, phenotype_dataframes in enumerate(create_phenotype_dataframes(dataset, dataset_keys, False)):
                                value_name, df, row_colors = phenotype_dataframes

                                # Output directory + filename + path
                                output_dir = dirname + "/OUTPUT"
                                output_filename = "clustermap_" + tissue_selection + "_{}".format(idx + 1) + "_" + data_type
                                outpath = os.path.join(output_dir, output_filename + "." + extension)

                                plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)
                        else:
                            for dataset in dataset_dict[tissue_selection][data_type]:
                                # Local dataset path
                                file_path = dirname + "/DATASETS/" + dataset + ".csv"
                                
                                for idx, phenotype_dataframes in enumerate(create_phenotype_dataframes(dataset, dataset_keys, file_path)):
                                    value_name, df, row_colors = phenotype_dataframes

                                    # Output directory + filename + path
                                    output_dir = dirname + "/OUTPUT"
                                    output_filename = "clustermap__" + tissue_selection + "_{}".format(idx + 1) + "_" + dataset.replace("_HighPlex_Query_Total_Summary_Results_", "_") + "_" + data_type
                                    outpath = os.path.join(output_dir, output_filename + "." + extension)

                                    plot_dataframe_to_file(value_name, dataset_keys, df, row_colors, outpath, custom_legends, save)

#-------
# DRAW !
#-------

# Global vars
_get_source_data = True # Set to "True" to get the source data...
describe_data = False # Set to "True" to describe continuous and categorical data
pca_analysis_show_plots = False # Set to "True" to show PCA analysis plots interactively
clustering_analysis_show_plot = False # Set to "True" to show clustering analysis plots interactively
clustering_analysis_print_output = False # Set to "True" to print clustering analysis output interactively

# Example usage
draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range", "Histology Bool"], "_ALL_Splitted_Layer1", "percentage", None], True, True)
draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range", "Histology Bool"], "_ALL_Splitted_Layer1", "density", None], True, True)

draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range", "Histology Bool"], "_ALL_Merged_Layer1", "percentage", None], True, True)
draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range", "Histology Bool"], "_ALL_Merged_Layer1", "density", None], True, True)

draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range"], "_ALL_Merged_Layer1", "percentage", None], True, True)
draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range"], "_ALL_Merged_Layer1", "density", None], True, True)

draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla", "percentage", False], True, True)
draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_ILCs+THs_Layer1+Cortex+Medulla", "density", False], True, True)

draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_ILCs+THs_Cortex+Medulla", "percentage", False], True, True)
draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_ILCs+THs_Cortex+Medulla", "density", False], True, True)

draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_THs_Layer1+Cortex+Medulla", "percentage", False], True, True)
draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_THs_Layer1+Cortex+Medulla", "density", False], True, True)

draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_THs_Cortex+Medulla", "percentage", False], True, True)
draw(datasets_dict, [["ST"], ["Sex", "Age Range"], "_ST&TdT_THs_Cortex+Medulla", "density", False], True, True)
