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
import glob, re
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import importlib

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
plot_style = "white" # Plot style / Can be: "whitegrid", etc...
grid_color = "#777" # Grid color / Can be: "#ccc"...
stripplot_color = "#642222" # Stripplot color / Can be: "#ccc"...
stripplot_color_palette = ["#642222", "#226422"] # Stripplot color palette / Can be: "#ccc"...
axe_label_color = "#333"  # Axis label color / Can be: "#ccc"...
axe_color = "#777"  # Axis color / Can be: "#ccc"...
labelsize = 3  # Font size for labels / Format: x pt

# Define datasets and their structure
datasets_dict___AllData = {
    "multiplexMain": {
        "percentage": [
            "AA_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "AA_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "SG_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DA_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DA_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DC_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_HighPlex_Query_Total_Summary_Results_LymphomatousFollicle",
            "FL_HighPlex_Query_Total_Summary_Results_LF_Excluded",
            "SR_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_HighPlex_Query_Total_Summary_Results_WhitePulp",
            "SR_HighPlex_Query_Total_Summary_Results_RedPulp",
        ],
        "density": [
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_LymphomatousFollicle",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_LF_Excluded",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
        ],
    },
    "multiplexAlt#1.1": {
        "percentage": [
            "DC&CD3_ILC3-Only_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC&CD3_ILC3-Only_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC&CD3_ILC3-Only_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
        "density": [
            "DC&CD3_ILC3-Only_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC&CD3_ILC3-Only_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC&CD3_ILC3-Only_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
    },
    "multiplexAlt#1.2": {
        "percentage": [
            "DC&CD3_ILC1+ILC3_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC&CD3_ILC1+ILC3_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC&CD3_ILC1+ILC3_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
        "density": [
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
    },
    "multiplexAlt#2": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
    },
}
datasets_dict___Figures = {
    "AA+DA+DC_Layer1": {
        "percentage": [
            "AA_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SG+SR_Layer1": {
        "percentage": [
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "DC&CD3_ILC3-Only_Layer1": {
        "percentage": [
            "DC&CD3_ILC3-Only_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "DC&CD3_ILC3-Only_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "DC&CD3_ILC1+ILC3_Layer1": {
        "percentage": [
            "DC&CD3_ILC1+ILC3_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "ST&TdT_Layer1": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "ST&TdT_PooledTH_Layer1": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "ST&TdT_DetailedTH_Layer1": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "FL+AA+SG_Layer1": {
        "percentage": [
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "FL+AA+SG_NoT_Layer1": {
        "percentage": [
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SG+FL_Layer1": {
        "percentage": [
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SG+FL_NoT_Layer1": {
        "percentage": [
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SG_Layer1": {
        "percentage": [
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SG_NoT_Layer1": {
        "percentage": [
            "SG_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "FL_Layer1": {
        "percentage": [
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "FL_NoT_Layer1": {
        "percentage": [
            "FL_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
    "SR_Layer1": {
        "percentage": [
            "SR_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "density": [
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
    },
}
datasets_dict___Paper_1 = {
    "DA+DC": {
        "density": [
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
    },
    "SG+AA": {
        "density": [
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
    },
    "DC&CD3_ILC1+ILC3": {
        "density": [
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC&CD3_ILC1+ILC3_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
        ],
    },
    "ST&TdT_PooledTH": {
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
    },
    "ST&TdT_DetailedTH": {
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
    },
    "SR": {
        "density": [
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
        ],
    },
    "FL": {
        "density": [
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_LymphomatousFollicle",
            "FL_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_LF_Excluded",
        ],
    },
    "B+T+IL7R": {
        "density": {
            "multiplexAlt#2-BCells": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_B-Cells_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_B-Cells_Cortex",
            ],
            "multiplexAlt#2-TCells": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_T-Cells_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_T-Cells_Cortex",
            ],
            "multiplexMain": [
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            ],
        },
    },
    "Relative_IL7R": {
        "density": {
            "multiplexAlt#2": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            ],
            "multiplexMain": [
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            ],
        },
    },
}
datasets_dict___Figures_B_T_IL7R = {
    "multiplexMain": {
        "WholeTissue": {
            "percentage": [
                "SG_HighPlex_Query_Total_Summary_Results_Layer1",
                "AA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
        "Split": {
            "percentage": [
                "SG_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "SG_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "AA_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "AA_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DA_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DA_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DC_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DC_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "SR_HighPlex_Query_Total_Summary_Results_WhitePulp",
                "SR_HighPlex_Query_Total_Summary_Results_RedPulp",
            ],
            "density": [
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
            ],
        },
    },
    "multiplexAlt#2": {
        "WholeTissue": {
            "percentage": [
                "ST&TdT_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
        "Split": {
            "percentage": [
                "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
                "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            ],
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            ],
        },
    },
}
datasets_dict___Figures_TCR_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            "AA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "Split": [
            "AA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "AA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DA_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DC_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SG_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "SG_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SR_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
            "SR_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            "ST&TdT_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "Split": [
            "ST&TdT_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_PoolTCR+Density_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
    },
}
datasets_dict___Figures_IL7R_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "Split": [
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
            "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
            "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
        ],
        "Split": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
        ],
    },
}
datasets_dict___Figures_ST_DetailedTH_Split = {
    "ST&TdT_DetailedTH_Split_Cortex+Medulla": {
        "percentage": [
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
        "density": [
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Cortex",
            "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Medulla",
        ],
    },
}
datasets_dict___Figures_B_T_IL7R_ST_WithPanelCD20 = {
    "multiplexMain": {
        "WholeTissue": {
            "density": [
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_Layer1",
            ],
        },
        "Split": {
            "density": [
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "SG_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "AA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DA_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GerminalCenter",
                "DC_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_GC_Excluded",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_WhitePulp",
                "SR_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_RedPulp",
            ],
        },
    },
    "multiplexAlt#2-BCells": {
        "WholeTissue": {
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_B-Cells_Layer1",
            ],
        },
        "Split": {
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_B-Cells_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_B-Cells_Cortex",
            ],
        },
    },
    "multiplexAlt#2-TCells": {
        "WholeTissue": {
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_T-Cells_Layer1",
            ],
        },
        "Split": {
            "density": [
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_T-Cells_Medulla",
                "ST&TdT_PoolIL7R+Density_HighPlex_Query_Total_Summary_Results_With_B-Cells-FromComboCD20-ST_T-Cells_Cortex",
            ],
        },
    },
}

# Filter (± chunk) dataset columns
phenotypes_dict___AllData = {
    "multiplexMain": {
        "percentage": {
            "single_all": [
                [
                    "% CD20 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                    "% GATA Positive Cells",
                    "% ICOS Positive Cells",
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "single_chunked": [
                [
                    "% CD20 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                ],
                [
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                ],
                [
                    "% GATA Positive Cells",
                    "% ICOS Positive Cells",
                ],
                [
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "coloc": [
                [
                    "% B-Cells:CD20+TCR- Positive Cells", # B
                    "% T-Cells:CD20-TCR+ Positive Cells", # T
                    "% Pool-ILCs:CD20-TCR-IL7R+ Positive Cells", # IL7R Non B/T
                ],
                [
                    "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                ],
                [
                    "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                    "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                ],
                [
                    "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
                [
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC1s - Density/mm²", # ILC1
                    "TH1 - Density/mm²", # TH1
                ],
                [
                    "ILC2 - Density/mm²", # ILC2
                    "TH2 - Density/mm²", # TH2
                ],
                [
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 - Density/mm²", # TH17
                ],
            ],
        },
    },
    "multiplexAlt#1.1": {
        "percentage": {
            "single_all": [
                [
                    "% CD3 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "single_chunked": [
                [
                    "% CD3 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                ],
                [
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "coloc": [
                [
                    "% Mature T-Cells:CD3+TCR+ Positive Cells", # T
                    "% Pool-ILCs:CD3-TCR-IL7R+ Positive Cells", # IL7R Non T
                ],
                [
                    "% ILC3:ROR+AHR+CD3-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:CD3+TCR+ROR+AHR+ Positive Cells", # TH17
                    "% Resting T ILC3-like:ROR+AHR+CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC3
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "Mature T-Cells - Density/mm²", # T
                    "Pool IL7R (Non T) - Density/mm²", # IL7R Non T
                ],
                [
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 - Density/mm²", # TH17
                    "Resting T ILC3-like - Density/mm²", # icCD3+ ILC3
                ],
            ],
        },
    },
    "multiplexAlt#1.2": {
        "percentage": {
            "single_all": [
                [
                    "% CD3 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "single_chunked": [
                [
                    "% CD3 Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                ],
                [
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                ],
                [
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "coloc": [
                [
                    "% Mature T-Cells:CD3+TCR+ Positive Cells", # T
                    "% Pool-ILCs:CD3-TCR-IL7R+ Positive Cells", # IL7R Non T
                ],
                [
                    "% ILC1s:TBX+EOMES+/-CD3-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+CD3-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% T-Helper:CD3+TCR+TBX+EOMES+ Positive Cells", # TH1
                    "% Resting T ILC1s-like:TBX+EOMES+/-CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC1
                    "% Resting T NK/ILC1ie-like:TBX+EOMES+CD3+TCR-IL7R- Positive Cells", # icCD3+ NK/ILC1ie
                ],
                [
                    "% ILC3:ROR+AHR+CD3-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:CD3+TCR+ROR+AHR+ Positive Cells", # TH17
                    "% Resting T ILC3-like:ROR+AHR+CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC3
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "Mature T-Cells - Density/mm²", # T
                    "Pool IL7R (Non T) - Density/mm²", # IL7R Non T
                ],
                [
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC1s - Density/mm²", # ILC1
                    "TH1 - Density/mm²", # TH1
                    "Resting T NK/ILC1ie-like - Density/mm²", # icCD3+ NK/ILC1ie
                    "Resting T ILC1s-like - Density/mm²", # icCD3+ ILC1
                ],
                [
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 - Density/mm²", # TH17
                    "Resting T ILC3-like - Density/mm²", # icCD3+ ILC3
                ],
            ],
        },
    },
    "multiplexAlt#2": {
        "percentage": {
            "single_all": [
                [
                    "% TdT Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                    "% GATA Positive Cells",
                    "% ICOS Positive Cells",
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "single_chunked": [
                [
                    "% TdT Positive Cells",
                    "% TCR-Alpha-Delta Positive Cells",
                    "% IL7R Positive Cells",
                ],
                [
                    "% EOMES Positive Cells",
                    "% TBX Positive Cells",
                ],
                [
                    "% GATA Positive Cells",
                    "% ICOS Positive Cells",
                ],
                [
                    "% ROR-GammaT Positive Cells",
                    "% AHR Positive Cells",
                ],
            ],
            "coloc": [
                [
                    "% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                    "% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                    "% Pool-IL7R Non T:TdT-TCR-IL7R+ Positive Cells", # IL7R Non T
                    "% Pool-IL7R Early T:TdT+TCR-IL7R+ Positive Cells", # IL7R Immature T
                    "% Pool-IL7R Mature T:TdT-TCR+IL7R+ Positive Cells", # IL7R Mature T
                ],
                [
                    "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                    "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                    "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT–)
                    "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                    "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                ],
                [
                    "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                    "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT–)
                    "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                    "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                ],
                [
                    "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                    "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT–)
                    "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                    "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
                [
                    "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                    "ILC1s - Density/mm²", # ILC1
                    "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                    "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                    "Immature TH1 - Density/mm²", # Immature TH1
                ],
                [
                    "ILC2 - Density/mm²", # ILC2
                    "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                    "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                    "Immature TH2 - Density/mm²", # Immature TH2
                ],
                [
                    "ILC3 - Density/mm²", # ILC3
                    "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                    "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                    "Immature TH17 - Density/mm²", # Immature TH17
                ],
            ],
        },
    },
}
phenotypes_dict___Figures = {
    "AA+DA+DC_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG+SR_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "DC&CD3_ILC3-Only_Layer1": {
        "percentage": [
            [
                "% ILC3:ROR+AHR+CD3-TCR-IL7R+ Positive Cells", # ILC3
                "% Resting T ILC3-like:ROR+AHR+CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC3
            ],
            [
                "% Mature T-Cells:CD3+TCR+ Positive Cells", # T
                "% T-Helper:CD3+TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "ILC3 - Density/mm²", # ILC3
                "Resting T ILC3-like - Density/mm²", # icCD3+ ILC3
            ],
            [
                "Mature T-Cells - Density/mm²", # T
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "DC&CD3_ILC1+ILC3_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD3-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD3-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% Resting T ILC1s-like:TBX+EOMES+/-CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC1
                "% Resting T NK/ILC1ie-like:TBX+EOMES+CD3+TCR-IL7R- Positive Cells", # icCD3+ NK/ILC1ie
                "% ILC3:ROR+AHR+CD3-TCR-IL7R+ Positive Cells", # ILC3
                "% Resting T ILC3-like:ROR+AHR+CD3+TCR-IL7R+ Positive Cells", # icCD3+ ILC3
            ],
            [
                "% Mature T-Cells:CD3+TCR+ Positive Cells", # T
                "% T-Helper:CD3+TCR+TBX+EOMES+ Positive Cells", # TH1
                "% T-Helper:CD3+TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "Resting T NK/ILC1ie-like - Density/mm²", # icCD3+ NK/ILC1ie
                "Resting T ILC1s-like - Density/mm²", # icCD3+ ILC1
                "ILC3 - Density/mm²", # ILC3
                "Resting T ILC3-like - Density/mm²", # icCD3+ ILC3
            ],
            [
                "Mature T-Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "ST&TdT_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                "% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT–)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT–)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT–)
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
                "Early T-Cells - Density/mm²", # Immature T
                "Mature T-Cells - Density/mm²", # Mature T
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "ST&TdT_PooledTH_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+TdT-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                "% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT–)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT–)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT–)
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
                "Early T-Cells - Density/mm²", # Immature T
                "Mature T-Cells - Density/mm²", # Mature T
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "ST&TdT_DetailedTH_Layer1": {
        "percentage": [
            [
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT–)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT–)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT–)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "FL+AA+SG_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "FL+AA+SG_NoT_Layer1": {
        "percentage": [
            [
                "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG+FL_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG+FL_NoT_Layer1": {
        "percentage": [
            [
                "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG_NoT_Layer1": {
        "percentage": [
            [
                "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "FL_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "FL_NoT_Layer1": {
        "percentage": [
            [
                "% T-Helper:CD20-TCR+TBX+EOMES+ Positive Cells", # TH1
                "% T-Helper:CD20-TCR+GATA+ICOS+ Positive Cells", # TH2
                "% T-Helper:CD20-TCR+ROR+AHR+ Positive Cells", # TH17
            ],
        ],
        "density": [
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SR_Layer1": {
        "percentage": [
            [
                "% ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "% NK/ILC1ie:TBX+EOMES+CD20-TCR-IL7R- Positive Cells", # NK/ILC1ie
                "% ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "% ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
            [
                "% T-Cells:CD20-TCR+ Positive Cells", # T
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
                "T Cells - Density/mm²", # T
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
}
phenotypes_dict___Paper_1 = {
    "DA+DC": {
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "ILC2 - Density/mm²", # ILC2
                "ILC3 - Density/mm²", # ILC3
            ],
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "SG+AA": {
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "ILC2 - Density/mm²", # ILC2
                "ILC3 - Density/mm²", # ILC3
            ],
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "DC&CD3_ILC1+ILC3": {
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "Resting T NK/ILC1ie-like - Density/mm²", # icCD3+ NK/ILC1ie
                "Resting T ILC1s-like - Density/mm²", # icCD3+ ILC1
                "ILC3 - Density/mm²", # ILC3
                "Resting T ILC3-like - Density/mm²", # icCD3+ ILC3
            ],
            [
                "TH1 - Density/mm²", # TH1
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "ST&TdT_PooledTH": {
        "density": [
            [
                "ILC1s - Density/mm²", # ILC1
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC2 - Density/mm²", # ILC2
                "ILC3 - Density/mm²", # ILC3
            ],
            [
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "ST&TdT_DetailedTH": {
        "density": [
            [
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
            ],
            [
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
            ],
            [
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
    "SR": {
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "ILC2 - Density/mm²", # ILC2
                "ILC3 - Density/mm²", # ILC3
            ],
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "FL": {
        "density": [
            [
                "NK/ILC1ie - Density/mm²", # NK/ILC1ie
                "ILC1s - Density/mm²", # ILC1
                "ILC2 - Density/mm²", # ILC2
                "ILC3 - Density/mm²", # ILC3
            ],
            [
                "TH1 - Density/mm²", # TH1
                "TH2 - Density/mm²", # TH2
                "TH17 - Density/mm²", # TH17
            ],
        ],
    },
    "B+T+IL7R": {
        "density": {
            "multiplexAlt#2-BCells": [
                [
                    "B-Cells - Density/mm²", # B
                ],
            ],
            "multiplexAlt#2-TCells": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
            ],
            "multiplexMain": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
            ],
        },
    },
    "Relative_IL7R": {
        "density": {
            "multiplexAlt#2": [
                [
                    "%Pool-IL7R ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                    "%Pool-IL7R ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                    "%Pool-IL7R ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
                ],
            ],
            "multiplexMain": [
                [
                    "%Pool-IL7R ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                    "%Pool-IL7R ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                    "%Pool-IL7R ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
                ],
            ],
        },
    },
}
phenotypes_dict___Figures_B_T_IL7R = {
    "multiplexMain": {
        "WholeTissue": {
            "percentage": [
                [
                    "% B-Cells:CD20+TCR- Positive Cells", # B
                    "% T-Cells:CD20-TCR+ Positive Cells", # T
                    "% Pool-ILCs:CD20-TCR-IL7R+ Positive Cells", # IL7R Non B/T
                ],
            ],
            "density": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
            ],
        },
        "Split": {
            "percentage": [
                [
                    "% B-Cells:CD20+TCR- Positive Cells", # B
                    "% T-Cells:CD20-TCR+ Positive Cells", # T
                    "% Pool-ILCs:CD20-TCR-IL7R+ Positive Cells", # IL7R Non B/T
                ],
            ],
            "density": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
            ],
        },
    },
    "multiplexAlt#2": {
        "WholeTissue": {
            "percentage": [
                [
                    "% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                    "% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                    "% Pool-IL7R Non T:TdT-TCR-IL7R+ Positive Cells", # IL7R Non T
                    "% Pool-IL7R Early T:TdT+TCR-IL7R+ Positive Cells", # IL7R Immature T
                    "% Pool-IL7R Mature T:TdT-TCR+IL7R+ Positive Cells", # IL7R Mature T
                ],
            ],
            "density": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
            ],
        },
        "Split": {
            "percentage": [
                [
                    "% Early T-Cells:TdT+TCR- Positive Cells", # Immature T
                    "% Mature T-Cells:TdT-TCR+ Positive Cells", # Mature T
                    "% Pool-IL7R Non T:TdT-TCR-IL7R+ Positive Cells", # IL7R Non T
                    "% Pool-IL7R Early T:TdT+TCR-IL7R+ Positive Cells", # IL7R Immature T
                    "% Pool-IL7R Mature T:TdT-TCR+IL7R+ Positive Cells", # IL7R Mature T
                ],
            ],
            "density": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
            ],
        },
    },
}
phenotypes_dict___Figures_TCR_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "%Pool-TCR+ T-Helper:CD20-TCR+TBX+EOMES+ Cells", # TH1
                "%Pool-TCR+ T-Helper:CD20-TCR+GATA+ICOS+ Cells", # TH2
                "%Pool-TCR+ T-Helper:CD20-TCR+ROR+AHR+ Cells", # TH17
            ],
        ],
        "Split": [
            [
                "%Pool-TCR+ T-Helper:CD20-TCR+TBX+EOMES+ Cells", # TH1
                "%Pool-TCR+ T-Helper:CD20-TCR+GATA+ICOS+ Cells", # TH2
                "%Pool-TCR+ T-Helper:CD20-TCR+ROR+AHR+ Cells", # TH17
            ],
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            [
                "%Pool-TCR+ TH1 Cells", # TH1
                "%Pool-TCR+ TH2 Cells", # TH2
                "%Pool-TCR+ TH17 Cells", # TH17
            ],
        ],
        "Split": [
            [
                "%Pool-TCR+ TH1 Cells", # TH1
                "%Pool-TCR+ TH2 Cells", # TH2
                "%Pool-TCR+ TH17 Cells", # TH17
            ],
        ],
    },
}
phenotypes_dict___Figures_IL7R_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "%Pool-IL7R ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "%Pool-IL7R ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "%Pool-IL7R ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
        ],
        "Split": [
            [
                "%Pool-IL7R ILC1s:TBX+EOMES+/-CD20-TCR-IL7R+ Positive Cells", # ILC1
                "%Pool-IL7R ILC2:GATA+ICOS+CD20-TCR-IL7R+ Positive Cells", # ILC2
                "%Pool-IL7R ILC3:ROR+AHR+CD20-TCR-IL7R+ Positive Cells", # ILC3
            ],
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            [
                "%Pool-IL7R ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "%Pool-IL7R ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "%Pool-IL7R ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
            ],
        ],
        "Split": [
            [
                "%Pool-IL7R ILC1s:TBX+EOMES+/-TdT-TCR-IL7R+ Positive Cells", # ILC1
                "%Pool-IL7R ILC2:GATA+ICOS+TdT-TCR-IL7R+ Positive Cells", # ILC2
                "%Pool-IL7R ILC3:ROR+AHR+TdT-TCR-IL7R+ Positive Cells", # ILC3
            ],
        ],
    },
}
phenotypes_dict___Figures_ST_DetailedTH_Split = {
    "ST&TdT_DetailedTH_Split_Cortex+Medulla": {
        "percentage": [
            [
                "% T-Helper:TdT-TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT–)
                "% T-Helper:TdT+TCR+TBX+EOMES+ Positive Cells", # TH1 (TdT+)
                "% Immature T-Helper:TdT+TCR-TBX+EOMES+ Positive Cells", # Immature TH1
            ],
            [
                "% T-Helper:TdT-TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT–)
                "% T-Helper:TdT+TCR+GATA+ICOS+ Positive Cells", # TH2 (TdT+)
                "% Immature T-Helper:TdT+TCR-GATA+ICOS+ Positive Cells", # Immature TH2
            ],
            [
                "% T-Helper:TdT-TCR+ROR+AHR+ Positive Cells", # TH17 (TdT–)
                "% T-Helper:TdT+TCR+ROR+AHR+ Positive Cells", # TH17 (TdT+)
                "% Immature T-Helper:TdT+TCR-ROR+AHR+ Positive Cells", # Immature TH17
            ],
        ],
        "density": [
            [
                "TH1 TdT- - Density/mm²", # TH1 (TdT–)
                "TH1 TdT+ - Density/mm²", # TH1 (TdT+)
                "Immature TH1 - Density/mm²", # Immature TH1
            ],
            [
                "TH2 TdT- - Density/mm²", # TH2 (TdT–)
                "TH2 TdT+ - Density/mm²", # TH2 (TdT+)
                "Immature TH2 - Density/mm²", # Immature TH2
            ],
            [
                "TH17 TdT- - Density/mm²", # TH17 (TdT–)
                "TH17 TdT+ - Density/mm²", # TH17 (TdT+)
                "Immature TH17 - Density/mm²", # Immature TH17
            ],
        ],
    },
}
phenotypes_dict___Figures_B_T_IL7R_ST_WithPanelCD20 = {
    "multiplexMain": {
        "WholeTissue": {
            "density": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "B Cells - Density/mm²", # B
                    "T Cells - Density/mm²", # T
                    "Pool IL7R (Non B/T) - Density/mm²", # IL7R Non B/T
                ],
            ],
        },
    },
    "multiplexAlt#2-BCells": {
        "WholeTissue": {
            "density": [
                [
                    "B-Cells - Density/mm²", # B
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "B-Cells - Density/mm²", # B
                ],
            ],
        },
    },
    "multiplexAlt#2-TCells": {
        "WholeTissue": {
            "density": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "Early T-Cells - Density/mm²", # Immature T
                    "Mature T-Cells - Density/mm²", # Mature T
                    "Pool-IL7R Non T - Density/mm²", # IL7R Non T
                    "Pool-IL7R Early T - Density/mm²", # IL7R Immature T
                    "Pool-IL7R Mature T - Density/mm²", # IL7R Mature T
                ],
            ],
        },
    },
}

# Rename dataset columns
phenotypes_renamed_dict___AllData = {
    "multiplexMain": {
        "percentage": {
            "single_all": [
                [
                    "CD20+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                    "EOMES+ Cells",
                    "TBX+ Cells",
                    "GATA+ Cells",
                    "ICOS+ Cells",
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "single_chunked": [
                [
                    "CD20+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                ],
                [
                    "EOMES+ Cells",
                    "TBX+ Cells",
                ],
                [
                    "GATA+ Cells",
                    "ICOS+ Cells",
                ],
                [
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "coloc": [
                [
                    "B",
                    "T",
                    "IL7R Non B/T",
                ],
                [
                    "ILC1",
                    "NK/ILC1ie",
                    "TH1",
                ],
                [
                    "ILC2",
                    "TH2",
                ],
                [
                    "ILC3",
                    "TH17",
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "B",
                    "T",
                    "IL7R Non B/T",
                ],
                [
                    "NK/ILC1ie",
                    "ILC1",
                    "TH1",
                ],
                [
                    "ILC2",
                    "TH2",
                ],
                [
                    "ILC3",
                    "TH17",
                ],
            ],
        },
    },
    "multiplexAlt#1.1": {
        "percentage": {
            "single_all": [
                [
                    "CD3+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "single_chunked": [
                [
                    "CD3+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                ],
                [
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "coloc": [
                [
                    "T",
                    "IL7R Non T",
                ],
                [
                    "ILC3",
                    "TH17",
                    "icCD3+ ILC3",
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "T",
                    "IL7R Non T",
                ],
                [
                    "ILC3",
                    "TH17",
                    "icCD3+ ILC3",
                ],
            ],
        },
    },
    "multiplexAlt#1.2": {
        "percentage": {
            "single_all": [
                [
                    "CD3+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                    "EOMES+ Cells",
                    "TBX+ Cells",
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "single_chunked": [
                [
                    "CD3+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                ],
                [
                    "EOMES+ Cells",
                    "TBX+ Cells",
                ],
                [
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "coloc": [
                [
                    "T",
                    "IL7R Non T",
                ],
                [
                    "ILC1",
                    "NK/ILC1ie",
                    "TH1",
                    "icCD3+ ILC1",
                    "icCD3+ NK/ILC1ie",
                ],
                [
                    "ILC3",
                    "TH17",
                    "icCD3+ ILC3",
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "T",
                    "IL7R Non T",
                ],
                [
                    "NK/ILC1ie",
                    "ILC1",
                    "TH1",
                    "icCD3+ NK/ILC1ie",
                    "icCD3+ ILC1",
                ],
                [
                    "ILC3",
                    "TH17",
                    "icCD3+ ILC3",
                ],
            ],
        },
    },
    "multiplexAlt#2": {
        "percentage": {
            "single_all": [
                [
                    "TdT+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                    "EOMES+ Cells",
                    "TBX+ Cells",
                    "GATA+ Cells",
                    "ICOS+ Cells",
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "single_chunked": [
                [
                    "TdT+ Cells",
                    "TCRαδ+ Cells",
                    "IL7R+ Cells",
                ],
                [
                    "EOMES+ Cells",
                    "TBX+ Cells",
                ],
                [
                    "GATA+ Cells",
                    "ICOS+ Cells",
                ],
                [
                    "RORγT+ Cells",
                    "AHR+ Cells",
                ],
            ],
            "coloc": [
                [
                    "Immature T",
                    "Mature T",
                    "IL7R Non T",
                    "IL7R Immature T",
                    "IL7R Mature T",
                ],
                [
                    "ILC1",
                    "NK/ILC1ie",
                    "TH1 (TdT–)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                ],
                [
                    "ILC2",
                    "TH2 (TdT–)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                ],
                [
                    "ILC3",
                    "TH17 (TdT–)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
        },
        "density": {
            "coloc": [
                [
                    "Immature T",
                    "Mature T",
                    "IL7R Non T",
                    "IL7R Immature T",
                    "IL7R Mature T",
                ],
                [
                    "NK/ILC1ie",
                    "ILC1",
                    "TH1 (TdT–)",
                    "TH1 (TdT+)",
                    "Immature TH1",
                ],
                [
                    "ILC2",
                    "TH2 (TdT–)",
                    "TH2 (TdT+)",
                    "Immature TH2",
                ],
                [
                    "ILC3",
                    "TH17 (TdT–)",
                    "TH17 (TdT+)",
                    "Immature TH17",
                ],
            ],
        },
    },
}
phenotypes_renamed_dict___Figures = {
    "AA+DA+DC_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG+SR_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "DC&CD3_ILC3-Only_Layer1": {
        "percentage": [
            [
                "ILC3",
                "icCD3+ ILC3",
            ],
            [
                "T",
                "TH17",
            ],
        ],
        "density": [
            [
                "ILC3",
                "icCD3+ ILC3",
            ],
            [
                "T",
                "TH17",
            ],
        ],
    },
    "DC&CD3_ILC1+ILC3_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "icCD3+ ILC1",
                "icCD3+ NK/ILC1ie",
                "ILC3",
                "icCD3+ ILC3",
            ],
            [
                "T",
                "TH1",
                "TH17",
            ],
        ],
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "icCD3+ NK/ILC1ie",
                "icCD3+ ILC1",
                "ILC3",
                "icCD3+ ILC3",
            ],
            [
                "T",
                "TH1",
                "TH17",
            ],
        ],
    },
    "ST&TdT_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "Immature T",
                "Mature T",
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
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
                "Immature T",
                "Mature T",
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "ST&TdT_PooledTH_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "Immature T",
                "Mature T",
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
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
                "Immature T",
                "Mature T",
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "ST&TdT_DetailedTH_Layer1": {
        "percentage": [
            [
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
    "FL+AA+SG_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "FL+AA+SG_NoT_Layer1": {
        "percentage": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "density": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG+FL_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG+FL_NoT_Layer1": {
        "percentage": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "density": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG_NoT_Layer1": {
        "percentage": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "density": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "FL_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "FL_NoT_Layer1": {
        "percentage": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
        "density": [
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SR_Layer1": {
        "percentage": [
            [
                "ILC1",
                "NK/ILC1ie",
                "ILC2",
                "ILC3",
            ],
            [
                "T",
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
                "T",
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
}
phenotypes_renamed_dict___Paper_1 = {
    "DA+DC": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "SG+AA": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "DC&CD3_ILC1+ILC3": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "icCD3+ NK/ILC1ie",
                "icCD3+ ILC1",
                "ILC3",
                "icCD3+ ILC3",
            ],
            [
                "TH1",
                "TH17",
            ],
        ],
    },
    "ST&TdT_PooledTH": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                "TH1 TdT–",
                "TH1 TdT+",
                "Immature TH1",
                "TH2 TdT–",
                "TH2 TdT+",
                "Immature TH2",
                "TH17 TdT–",
                "TH17 TdT+",
                "Immature TH17",
            ],
        ],
    },
    "ST&TdT_DetailedTH": {
        "density": [
            [
                "TH1 TdT–",
                "TH1 TdT+",
                "Immature TH1",
            ],
            [
                "TH2 TdT–",
                "TH2 TdT+",
                "Immature TH2",
            ],
            [
                "TH17 TdT–",
                "TH17 TdT+",
                "Immature TH17",
            ],
        ],
    },
    "SR": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "FL": {
        "density": [
            [
                "NK/ILC1ie",
                "ILC1",
                "ILC2",
                "ILC3",
            ],
            [
                "TH1",
                "TH2",
                "TH17",
            ],
        ],
    },
    "B+T+IL7R": {
        "density": {
            "multiplexAlt#2-BCells": [
                [
                    "B", # B
                ],
            ],
            "multiplexAlt#2-TCells": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
            "multiplexMain": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
        },
    },
    "Relative_IL7R": {
        "density": {
            "multiplexAlt#2": [
                [
                    "ILC1", # ILC1
                    "ILC2", # ILC2
                    "ILC3", # ILC3
                ],
            ],
            "multiplexMain": [
                [
                    "ILC1", # ILC1
                    "ILC2", # ILC2
                    "ILC3", # ILC3
                ],
            ],
        },
    },
}
phenotypes_renamed_dict___Figures_B_T_IL7R = {
    "multiplexMain": {
        "WholeTissue": {
            "percentage": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
            "density": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
        },
        "Split": {
            "percentage": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
            "density": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
        },
    },
    "multiplexAlt#2": {
        "WholeTissue": {
            "percentage": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
            "density": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
        },
        "Split": {
            "percentage": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
            "density": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
        },
    },
}
phenotypes_renamed_dict___Figures_TCR_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "TH1", # TH1
                "TH2", # TH2
                "TH17", # TH17
            ],
        ],
        "Split": [
            [
                "TH1", # TH1
                "TH2", # TH2
                "TH17", # TH17
            ],
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            [
                "TH1", # TH1
                "TH2", # TH2
                "TH17", # TH17
            ],
        ],
        "Split": [
            [
                "TH1", # TH1
                "TH2", # TH2
                "TH17", # TH17
            ],
        ],
    },
}
phenotypes_renamed_dict___Figures_IL7R_Relative = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
        "Split": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
        "Split": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
    },
}
phenotypes_renamed_dict___Figures_ST_DetailedTH_Split = {
    "ST&TdT_DetailedTH_Split_Cortex+Medulla": {
        "percentage": [
            [
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
            ],
            [
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
            ],
            [
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
        "density": [
            [
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
            ],
            [
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
            ],
            [
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
}
phenotypes_renamed_dict___Figures_B_T_IL7R_ST_WithPanelCD20 = {
    "multiplexMain": {
        "WholeTissue": {
            "density": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "B", # B
                    "T", # T
                    "IL7R Non B/T", # IL7R Non B/T
                ],
            ],
        },
    },
    "multiplexAlt#2-BCells": {
        "WholeTissue": {
            "density": [
                [
                    "B", # B
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "B", # B
                ],
            ],
        },
    },
    "multiplexAlt#2-TCells": {
        "WholeTissue": {
            "density": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
        },
        "Split": {
            "density": [
                [
                    "Immature T", # Immature T
                    "Mature T", # Mature T
                    "IL7R Non T", # IL7R Non T
                    "IL7R Immature T", # IL7R Immature T
                    "IL7R Mature T", # IL7R Mature T
                ],
            ],
        },
    },
}

# Order dataset columns
phenotypes_ordered_dict___AllData = {
    "multiplexMain": {
        "single_all": [
            [
                "CD20+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
                "EOMES+ Cells",
                "TBX+ Cells",
                "GATA+ Cells",
                "ICOS+ Cells",
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "single_chunked": [
            [
                "CD20+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
            ],
            [
                "EOMES+ Cells",
                "TBX+ Cells",
            ],
            [
                "GATA+ Cells",
                "ICOS+ Cells",
            ],
            [
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "coloc": [
            [
                "B",
                "T",
                "IL7R Non B/T",
            ],
            [
                "NK/ILC1ie",
                "ILC1",
                "TH1",
            ],
            [
                "ILC2",
                "TH2",
            ],
            [
                "ILC3",
                "TH17",
            ],
        ],
    },
    "multiplexAlt#1.1": {
        "single_all": [
            [
                "CD3+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "single_chunked": [
            [
                "CD3+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
            ],
            [
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "coloc": [
            [
                "T",
                "IL7R Non T",
            ],
            [
                "ILC3",
                "TH17",
                "icCD3+ ILC3",
            ],
        ],
    },
    "multiplexAlt#1.2": {
        "single_all": [
            [
                "CD3+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
                "EOMES+ Cells",
                "TBX+ Cells",
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "single_chunked": [
            [
                "CD3+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
            ],
            [
                "EOMES+ Cells",
                "TBX+ Cells",
            ],
            [
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "coloc": [
            [
                "T",
                "IL7R Non T",
            ],
            [
                "NK/ILC1ie",
                "ILC1",
                "TH1",
                "icCD3+ NK/ILC1ie",
                "icCD3+ ILC1",
            ],
            [
                "ILC3",
                "TH17",
                "icCD3+ ILC3",
            ],
        ],
    },
    "multiplexAlt#2": {
        "single_all": [
            [
                "TdT+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
                "EOMES+ Cells",
                "TBX+ Cells",
                "GATA+ Cells",
                "ICOS+ Cells",
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "single_chunked": [
            [
                "TdT+ Cells",
                "TCRαδ+ Cells",
                "IL7R+ Cells",
            ],
            [
                "EOMES+ Cells",
                "TBX+ Cells",
            ],
            [
                "GATA+ Cells",
                "ICOS+ Cells",
            ],
            [
                "RORγT+ Cells",
                "AHR+ Cells",
            ],
        ],
        "coloc": [
            [
                "Immature T",
                "Mature T",
                "IL7R Non T",
                "IL7R Immature T",
                "IL7R Mature T",
            ],
            [
                "NK/ILC1ie",
                "ILC1",
                "TH1 (TdT–)",
                "TH1 (TdT+)",
                "Immature TH1",
            ],
            [
                "ILC2",
                "TH2 (TdT–)",
                "TH2 (TdT+)",
                "Immature TH2",
            ],
            [
                "ILC3",
                "TH17 (TdT–)",
                "TH17 (TdT+)",
                "Immature TH17",
            ],
        ],
    },
}
phenotypes_ordered_dict___Figures = {
    "AA+DA+DC_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG+SR_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "DC&CD3_ILC3-Only_Layer1": [
        [
            "ILC3",
            "icCD3+ ILC3",
        ],
        [
            "T",
            "TH17",
        ],
    ],
    "DC&CD3_ILC1+ILC3_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC3",
            "icCD3+ NK/ILC1ie",
            "icCD3+ ILC1",
            "icCD3+ ILC3",
        ],
        [
            "T",
            "TH1",
            "TH17",
        ],
    ],
    "ST&TdT_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "Immature T",
            "Mature T",
            "TH1 (TdT–)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT–)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT–)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "ST&TdT_PooledTH_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "Immature T",
            "Mature T",
            "TH1 (TdT–)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT–)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT–)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "ST&TdT_DetailedTH_Layer1": [
        [
            "TH1 (TdT–)",
            "TH1 (TdT+)",
            "Immature TH1",
            "TH2 (TdT–)",
            "TH2 (TdT+)",
            "Immature TH2",
            "TH17 (TdT–)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
    "FL+AA+SG_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "FL+AA+SG_NoT_Layer1": [
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG+FL_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG+FL_NoT_Layer1": [
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG_NoT_Layer1": [
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "FL_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "FL_NoT_Layer1": [
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SR_Layer1": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "T",
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
}
phenotypes_ordered_dict___Paper_1 = {
    "DA+DC": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "SG+AA": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "DC&CD3_ILC1+ILC3": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC3",
            "icCD3+ NK/ILC1ie",
            "icCD3+ ILC1",
            "icCD3+ ILC3",
        ],
        [
            "TH1",
            "TH17",
        ],
    ],
    "ST&TdT_PooledTH": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "TH1 TdT–",
            "TH1 TdT+",
            "Immature TH1",
            "TH2 TdT–",
            "TH2 TdT+",
            "Immature TH2",
            "TH17 TdT–",
            "TH17 TdT+",
            "Immature TH17",
        ],
    ],
    "ST&TdT_DetailedTH": [
        [
            "TH1 TdT–",
            "TH1 TdT+",
            "Immature TH1",
        ],
        [
            "TH2 TdT–",
            "TH2 TdT+",
            "Immature TH2",
        ],
        [
            "TH17 TdT–",
            "TH17 TdT+",
            "Immature TH17",
        ],
    ],
    "SR": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "FL": [
        [
            "NK/ILC1ie",
            "ILC1",
            "ILC2",
            "ILC3",
        ],
        [
            "TH1",
            "TH2",
            "TH17",
        ],
    ],
    "B+T+IL7R": {
        "multiplexAlt#2-BCells": [
            [
                "B", # B
            ],
        ],
        "multiplexAlt#2-TCells": [
            [
                "Immature T", # Immature T
                "Mature T", # Mature T
                "IL7R Non T", # IL7R Non T
                "IL7R Immature T", # IL7R Immature T
                "IL7R Mature T", # IL7R Mature T
            ],
        ],
        "multiplexMain": [
            [
                "B", # B
                "T", # T
                "IL7R Non B/T", # IL7R Non B/T
            ],
        ],
    },
    "Relative_IL7R": {
        "multiplexAlt#2": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
        "multiplexMain": [
            [
                "ILC1", # ILC1
                "ILC2", # ILC2
                "ILC3", # ILC3
            ],
        ],
    },
}
phenotypes_ordered_dict___Figures_B_T_IL7R = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "B", # B
                "T", # T
                "IL7R Non B/T", # IL7R Non B/T
            ],
        ],
        "Split": [
            [
                "B", # B
                "T", # T
                "IL7R Non B/T", # IL7R Non B/T
            ],
        ],
    },
    "multiplexAlt#2": {
        "WholeTissue": [
            [
                "Immature T", # Immature T
                "Mature T", # Mature T
                "IL7R Non T", # IL7R Non T
                "IL7R Immature T", # IL7R Immature T
                "IL7R Mature T", # IL7R Mature T
            ],
        ],
        "Split": [
            [
                "Immature T", # Immature T
                "Mature T", # Mature T
                "IL7R Non T", # IL7R Non T
                "IL7R Immature T", # IL7R Immature T
                "IL7R Mature T", # IL7R Mature T
            ],
        ],
    },
}
phenotypes_ordered_dict___Figures_TCR_Relative = {
    "WholeTissue": [
        [
            "TH1", # TH1
            "TH2", # TH2
            "TH17", # TH17
        ],
    ],
    "Split": [
        [
            "TH1", # TH1
            "TH2", # TH2
            "TH17", # TH17
        ],
    ],
}
phenotypes_ordered_dict___Figures_IL7R_Relative = {
    "WholeTissue": [
        [
            "ILC1", # ILC1
            "ILC2", # ILC2
            "ILC3", # ILC3
        ],
    ],
    "Split": [
        [
            "ILC1", # ILC1
            "ILC2", # ILC2
            "ILC3", # ILC3
        ],
    ],
}
phenotypes_ordered_dict___Figures_ST_DetailedTH_Split = {
    "ST&TdT_DetailedTH_Split_Cortex+Medulla": [
        [
            "TH1 (TdT–)",
            "TH1 (TdT+)",
            "Immature TH1",
        ],
        [
            "TH2 (TdT–)",
            "TH2 (TdT+)",
            "Immature TH2",
        ],
        [
            "TH17 (TdT–)",
            "TH17 (TdT+)",
            "Immature TH17",
        ],
    ],
}
phenotypes_ordered_dict___Figures_B_T_IL7R_ST_WithPanelCD20 = {
    "multiplexMain": {
        "WholeTissue": [
            [
                "B", # B
                "T", # T
                "IL7R Non B/T", # IL7R Non B/T
            ],
        ],
        "Split": [
            [
                "B", # B
                "T", # T
                "IL7R Non B/T", # IL7R Non B/T
            ],
        ],
    },
    "multiplexAlt#2-BCells": {
        "WholeTissue": [
            [
                "B", # B
            ],
        ],
        "Split": [
            [
                "B", # B
            ],
        ],
    },
    "multiplexAlt#2-TCells": {
        "WholeTissue": [
            [
                "Immature T", # Immature T
                "Mature T", # Mature T
                "IL7R Non T", # IL7R Non T
                "IL7R Immature T", # IL7R Immature T
                "IL7R Mature T", # IL7R Mature T
            ],
        ],
        "Split": [
            [
                "Immature T", # Immature T
                "Mature T", # Mature T
                "IL7R Non T", # IL7R Non T
                "IL7R Immature T", # IL7R Immature T
                "IL7R Mature T", # IL7R Mature T
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
    df.columns = ["Image Tag"] + ["Analysis Region"] + renamed_cols
    return df

def create_phenotype_dataframes___AllData(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    output_type = dataset_keys[2]
    multiplex_type = dataset_keys[0]

    # Set "value_name"
    if data_type == "percentage": value_name = "% Positive Cells"
    elif data_type == "density": value_name = "# Cells / mm²"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___AllData[multiplex_type][data_type][output_type]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___AllData[multiplex_type][data_type][output_type]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___AllData[multiplex_type][output_type]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)
        
        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Set "value_name"
    if data_type == "percentage": value_name = "% Positive Cells"
    elif data_type == "density": value_name = "# Cells / mm²"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures[tissue_selection][data_type]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures[tissue_selection][data_type]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures[tissue_selection]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        if tissue_selection == "ST&TdT_PooledTH_Layer1":
            if phenotypes_ordered == ["Immature T", "Mature T", "TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                df["TH1"] = df[["TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1"]].mean(axis=1).round(8)
                df["TH2"] = df[["TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2"]].mean(axis=1).round(8)
                df["TH17"] = df[["TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]].mean(axis=1).round(8)

                df = df.drop(columns=["TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"])
                df = df.reindex(columns=["Image Tag", "Analysis Region", "Immature T", "Mature T", "TH1", "TH2", "TH17"])

                # Reset "phenotypes_ordered"
                phenotypes_ordered = ["Immature T", "Mature T", "TH1", "TH2", "TH17"]                

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Paper_1(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Set "value_name"
    if data_type == "density": value_name = "# Cells / mm²"

    if (tissue_selection == "B+T+IL7R"):
        multiplex_type = dataset_keys[2]

        # Data depends on dataset and phenotypes
        phenotypes_tuple = phenotypes_dict___Paper_1[tissue_selection][data_type][multiplex_type]
        phenotypes_renamed_tuple = phenotypes_renamed_dict___Paper_1[tissue_selection][data_type][multiplex_type]
        phenotypes_ordered_tuple = phenotypes_ordered_dict___Paper_1[tissue_selection][multiplex_type]

        for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
            # Filter dataset columns
            reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

            # Define dataset columns type
            dtypes_col = {"Image Tag": str, "Analysis Region": str}
            for pheno in phenotypes:
                dtypes_col[pheno] = np.float64

            # Read dataset and set indexes with panda
            df = None # !important reset dataframe before reading csv
            df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

            # Rename "Image Tag" data (ST/ST&TdT matching) # Not needed...
            #df['Image Tag'] = df['Image Tag'].str.split("_").str[3]

            # Rename columns
            df = rename_dataframe_cols(df, phenotypes_renamed)

            # Print dataframe for safety...
            #print(df)

            # Print dataframe for safety...
            #print(df.head())

            # Format dataframe for violin/box plots
            df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

            # Return dataframe
            yield value_name, df, phenotypes_ordered
    else:
        if (tissue_selection == "Relative_IL7R"):
            multiplex_type = dataset_keys[2]
            
            # Data depends on dataset and phenotypes
            phenotypes_tuple = phenotypes_dict___Paper_1[tissue_selection][data_type][multiplex_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict___Paper_1[tissue_selection][data_type][multiplex_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict___Paper_1[tissue_selection][multiplex_type]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                # Filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Rename "Image Tag" data (ST/ST&TdT matching) # Not needed...
                #df['Image Tag'] = df['Image Tag'].str.split("_").str[3]

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)

                # Print dataframe for safety...
                #print(df)

                # Print dataframe for safety...
                #print(df.head())

                # Format dataframe for violin/box plots
                df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

                # Return dataframe
                yield value_name, df, phenotypes_ordered
        else:
            # Data depends on dataset and phenotypes
            phenotypes_tuple = phenotypes_dict___Paper_1[tissue_selection][data_type]
            phenotypes_renamed_tuple = phenotypes_renamed_dict___Paper_1[tissue_selection][data_type]
            phenotypes_ordered_tuple = phenotypes_ordered_dict___Paper_1[tissue_selection]

            for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
                # Filter dataset columns
                reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

                # Define dataset columns type
                dtypes_col = {"Image Tag": str, "Analysis Region": str}
                for pheno in phenotypes:
                    dtypes_col[pheno] = np.float64

                # Read dataset and set indexes with panda
                df = None # !important reset dataframe before reading csv
                df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

                # Rename columns
                df = rename_dataframe_cols(df, phenotypes_renamed)

                # Print dataframe for safety...
                #print(df)

                if tissue_selection == "ST&TdT_PooledTH":
                    if phenotypes_ordered == ["TH1 TdT–", "TH1 TdT+", "Immature TH1", "TH2 TdT–", "TH2 TdT+", "Immature TH2", "TH17 TdT–", "TH17 TdT+", "Immature TH17"]:
                        df["TH1"] = df[["TH1 TdT–", "TH1 TdT+", "Immature TH1"]].mean(axis=1).round(8)
                        df["TH2"] = df[["TH2 TdT–", "TH2 TdT+", "Immature TH2"]].mean(axis=1).round(8)
                        df["TH17"] = df[["TH17 TdT–", "TH17 TdT+", "Immature TH17"]].mean(axis=1).round(8)

                        df = df.drop(columns=["TH1 TdT–", "TH1 TdT+", "Immature TH1", "TH2 TdT–", "TH2 TdT+", "Immature TH2", "TH17 TdT–", "TH17 TdT+", "Immature TH17"])
                        df = df.reindex(columns=["Image Tag", "Analysis Region", "TH1", "TH2", "TH17"])

                        # Reset "phenotypes_ordered"
                        phenotypes_ordered = ["TH1", "TH2", "TH17"]                

                # Print dataframe for safety...
                #print(df.head())

                # Need to reorder the dataframe columns here since we display split data (in/out interface)...
                # => Hue order in seaborn reflects the interface type...
                df = df[["Image Tag", "Analysis Region"] + phenotypes_ordered]

                # Print dataframe for safety...
                #print(df.head())

                # Format dataframe for violin/box plots
                df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

                # Return dataframe
                yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures_B_T_IL7R(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    multiplex_type = dataset_keys[2]
    tissue_structure = dataset_keys[0]

    # Set "value_name"
    if data_type == "percentage": value_name = "% Positive Cells"
    elif data_type == "density": value_name = "# Cells / mm²"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures_B_T_IL7R[multiplex_type][tissue_structure][data_type]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures_B_T_IL7R[multiplex_type][tissue_structure][data_type]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures_B_T_IL7R[multiplex_type][tissue_structure]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures_TCR_Relative(dataset, dataset_keys, input_file):
    # Dataset key values
    multiplex_type = dataset_keys[1]
    tissue_structure = dataset_keys[0]

    # Set "value_name"
    value_name = "% Positive Cells (Pool-TCR Relative)"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures_TCR_Relative[multiplex_type][tissue_structure]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures_TCR_Relative[multiplex_type][tissue_structure]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures_TCR_Relative[tissue_structure]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures_IL7R_Relative(dataset, dataset_keys, input_file):
    # Dataset key values
    multiplex_type = dataset_keys[1]
    tissue_structure = dataset_keys[0]

    # Set "value_name"
    value_name = "% Positive Cells (Pool IL7R Relative)"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures_IL7R_Relative[multiplex_type][tissue_structure]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures_IL7R_Relative[multiplex_type][tissue_structure]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures_IL7R_Relative[tissue_structure]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures_ST_DetailedTH_Split(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Set "value_name"
    if data_type == "percentage": value_name = "% Positive Cells"
    elif data_type == "density": value_name = "# Cells / mm²"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures_ST_DetailedTH_Split[tissue_selection][data_type]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures_ST_DetailedTH_Split[tissue_selection][data_type]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures_ST_DetailedTH_Split[tissue_selection]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def create_phenotype_dataframes___Figures_B_T_IL7R_ST_WithPanelCD20(dataset, dataset_keys, input_file):
    # Dataset key values
    data_type = dataset_keys[1]
    multiplex_type = dataset_keys[2]
    tissue_structure = dataset_keys[0]

    # Set "value_name"
    if data_type == "density": value_name = "# Cells / mm²"

    # Data depends on dataset and phenotypes
    phenotypes_tuple = phenotypes_dict___Figures_B_T_IL7R_ST_WithPanelCD20[multiplex_type][tissue_structure][data_type]
    phenotypes_renamed_tuple = phenotypes_renamed_dict___Figures_B_T_IL7R_ST_WithPanelCD20[multiplex_type][tissue_structure][data_type]
    phenotypes_ordered_tuple = phenotypes_ordered_dict___Figures_B_T_IL7R_ST_WithPanelCD20[multiplex_type][tissue_structure]

    for phenotypes, phenotypes_renamed, phenotypes_ordered in zip(phenotypes_tuple, phenotypes_renamed_tuple, phenotypes_ordered_tuple):
        # Filter dataset columns
        reader_columns = ["Image Tag", "Analysis Region"] + phenotypes

        # Define dataset columns type
        dtypes_col = {"Image Tag": str, "Analysis Region": str}
        for pheno in phenotypes:
            dtypes_col[pheno] = np.float64

        # Read dataset and set indexes with panda
        df = None # !important reset dataframe before reading csv
        df = pd.read_csv(input_file, sep=",", usecols=reader_columns, dtype=dtypes_col)

        # Rename "Image Tag" data (ST/ST&TdT matching) # Not needed...
        #df['Image Tag'] = df['Image Tag'].str.split("_").str[3]

        # Rename columns
        df = rename_dataframe_cols(df, phenotypes_renamed)

        # Print dataframe for safety...
        #print(df)

        # Print dataframe for safety...
        #print(df.head())

        # Format dataframe for violin/box plots
        df = df.melt(id_vars=["Image Tag", "Analysis Region"], var_name="Phenotypes", value_name=value_name)

        # Return dataframe
        yield value_name, df, phenotypes_ordered

def plot_dataframe_to_file(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_halo_tints_icCD3 = ["#7891b1", "#fabb93", "#bad1ed", "#e266e1", "#eaece5", "#d0d0aa"] # Tints #4 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        data_type = dataset_keys[1]
        tissue_selection = dataset_keys[0]

        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if (tissue_selection == "AA+DA+DC_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SG+SR_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "DC&CD3_ILC3-Only_Layer1"):
                if _order == ["ILC3", "icCD3+ ILC3"]:
                    customPalette = {
                        "ILC3": sns_pal_hex_halo_tints[3],
                        "icCD3+ ILC3": sns_pal_hex_halo_tints_icCD3[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "DC&CD3_ILC1+ILC3_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC3", "icCD3+ NK/ILC1ie", "icCD3+ ILC1", "icCD3+ ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC3": sns_pal_hex_halo_tints[3],
                        "icCD3+ NK/ILC1ie": sns_pal_hex_halo_tints_icCD3[0],
                        "icCD3+ ILC1": sns_pal_hex_halo_tints_icCD3[1],
                        "icCD3+ ILC3": sns_pal_hex_halo_tints_icCD3[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "ST&TdT_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["Immature T", "Mature T", "TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                    customPalette = {
                        "Immature T": sns_pal_hex_t_cells_tints[2],
                        "Mature T": sns_pal_hex_t_cells_tints[3],
                        "TH1 (TdT–)": sns_pal_hex_th1_cells_tints[2],
                        "TH1 (TdT+)": sns_pal_hex_th1_cells_tints[3],
                        "Immature TH1": sns_pal_hex_th1_cells_tints[4],
                        "TH2 (TdT–)": sns_pal_hex_th2_cells_tints[2],
                        "TH2 (TdT+)": sns_pal_hex_th2_cells_tints[3],
                        "Immature TH2": sns_pal_hex_th2_cells_tints[4],
                        "TH17 (TdT–)": sns_pal_hex_th17_cells_tints[2],
                        "TH17 (TdT+)": sns_pal_hex_th17_cells_tints[3],
                        "Immature TH17": sns_pal_hex_th17_cells_tints[4]
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "ST&TdT_PooledTH_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["Immature T", "Mature T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "Immature T": sns_pal_hex_t_cells_tints[2],
                        "Mature T": sns_pal_hex_t_cells_tints[3],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "ST&TdT_DetailedTH_Layer1"):
                if  _order == ["TH1 (TdT–)", "TH1 (TdT+)", "TH2 (TdT–)", "TH2 (TdT+)", "TH17 (TdT–)", "TH17 (TdT+)"]:
                    customPalette = {
                        "TH1 (TdT–)": sns_pal_hex_th1_cells_tints[2],
                        "TH1 (TdT+)": sns_pal_hex_th1_cells_tints[3],
                        "TH2 (TdT–)": sns_pal_hex_th2_cells_tints[2],
                        "TH2 (TdT+)": sns_pal_hex_th2_cells_tints[3],
                        "TH17 (TdT–)": sns_pal_hex_th17_cells_tints[2],
                        "TH17 (TdT+)": sns_pal_hex_th17_cells_tints[3],
                    }
                    customPalettes.append(customPalette)
                if  _order == ["TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                    customPalette = {
                        "TH1 (TdT–)": sns_pal_hex_th1_cells_tints[2],
                        "TH1 (TdT+)": sns_pal_hex_th1_cells_tints[3],
                        "Immature TH1": sns_pal_hex_th1_cells_tints[4],
                        "TH2 (TdT–)": sns_pal_hex_th2_cells_tints[2],
                        "TH2 (TdT+)": sns_pal_hex_th2_cells_tints[3],
                        "Immature TH2": sns_pal_hex_th2_cells_tints[4],
                        "TH17 (TdT–)": sns_pal_hex_th17_cells_tints[2],
                        "TH17 (TdT+)": sns_pal_hex_th17_cells_tints[3],
                        "Immature TH17": sns_pal_hex_th17_cells_tints[4]
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "FL+AA+SG_NoT_Layer1"):
                if _order == ["TH1", "TH2", "TH17"]:
                    customPalette = {
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SG+FL_NoT_Layer1"):
                if _order == ["TH1", "TH2", "TH17"]:
                    customPalette = {
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SG_NoT_Layer1"):
                if _order == ["TH1", "TH2", "TH17"]:
                    customPalette = {
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "FL_NoT_Layer1"):
                if _order == ["TH1", "TH2", "TH17"]:
                    customPalette = {
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "FL+AA+SG_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SG+FL_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SG_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "FL_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
            elif (tissue_selection == "SR_Layer1"):
                if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                        "ILC1": sns_pal_hex_halo_tints[1],
                        "ILC2": sns_pal_hex_halo_tints[2],
                        "ILC3": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)
                elif _order == ["T", "TH1", "TH2", "TH17"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "TH1": sns_pal_hex_halo_tints[1],
                        "TH2": sns_pal_hex_halo_tints[2],
                        "TH17": sns_pal_hex_halo_tints[3],
                    }
                    customPalettes.append(customPalette)

        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        if (tissue_selection == "AA+DA+DC_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (9*cm, 6*cm), [1,1,1], [1,1]
        elif (tissue_selection == "SG+SR_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (6*cm, 6*cm), [1,1], [1,1]
        elif (tissue_selection == "DC&CD3_ILC3-Only_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (3*cm, 6*cm), [1], [1,1]
        elif (tissue_selection == "DC&CD3_ILC1+ILC3_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (9*cm, 6*cm), [1], [1,1]
        elif (tissue_selection == "ST&TdT_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (9*cm, 3.75*cm), [1,3], [1]
        elif (tissue_selection == "ST&TdT_PooledTH_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 3.75*cm), [1,1.2], [1]
        elif (tissue_selection == "ST&TdT_DetailedTH_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (9*cm, 3.75*cm), [1], [1]
        elif (tissue_selection == "FL+AA+SG_NoT_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (9*cm, 2*cm), [1,1,1], [1]
        elif (tissue_selection == "SG+FL_NoT_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 2*cm), [1,1], [1]
        elif (tissue_selection == "SG_NoT_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (3*cm, 2*cm), [1], [1]
        elif (tissue_selection == "FL_NoT_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (3*cm, 2*cm), [1], [1]
        elif (tissue_selection == "FL+AA+SG_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (15*cm, 8*cm), [1.5,1,1], [1,1]
        elif (tissue_selection == "SG+FL_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (6*cm, 6*cm), [1,1], [1,1]
        elif (tissue_selection == "SG_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 3*cm), [1,1], [1]
        elif (tissue_selection == "FL_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 3*cm), [1,1], [1]
        elif (tissue_selection == "SR_Layer1"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 3*cm), [1,1], [1]

        # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style x/y axis bottom/left ticks
            if "_NoT_" in tissue_selection: # For THs without T Cells...
                plt.rcParams["xtick.bottom"] = True
                plt.rcParams["xtick.major.size"] = 1
                plt.rcParams["xtick.major.width"] = .5
                plt.rcParams["xtick.color"] = axe_color

                # Draw the full plot            
                (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios, layout="constrained")
                #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

                df_join = []
                if row > 1: # Arrange plots in rows
                    for r in range(row):
                        if (r % 2) == 0: df_even = df[0::2]
                        else: df_odd = df[1::2]

                    df_join = df_even + df_odd
                else:
                    df_join = df

                df_index = 0
                if row == 1 and col == 1:
                    sns_plot = sns.violinplot(data=df_join[0], ax=axes, x=value_name, y="Phenotypes", bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[0], palette=customPalettes[0], linewidth=.15, fill=True, split=False, order=order[0], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                    sns.stripplot(data=df_join[0], ax=axes, x=value_name, y="Phenotypes", color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    xlabel_rotation = 90 # Horizontal y labels
                    
                    # Set "ylabel_rotation" (in degrees => can be: "90", etc...)
                    ylabel_rotation = -90 # Horizontal y labels

                    axes.tick_params(axis="x", labelsize=labelsize, labelcolor=axe_label_color, pad=1, rotation=xlabel_rotation)
                    axes.tick_params(axis="y", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=ylabel_rotation)
                    #axes.set_ylabel(value_name, size=4, weight="bold")
                    axes.set_ylabel("")
                    axes.set_xlabel("")

                    axes.spines["left"].set_linewidth(.5)
                    axes.spines["left"].set_color(axe_color)

                    axes.spines["bottom"].set_linewidth(.5)
                    axes.spines["bottom"].set_color(axe_color)

                    axes.set_xlim(fixed_limits[0])
                elif row == 1 and col > 1:
                    for c in range(col):
                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x=value_name, y="Phenotypes", bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[0], palette=customPalettes[0], linewidth=.15, fill=True, split=False, order=order[0], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x=value_name, y="Phenotypes", color=stripplot_color, size=0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 90 # Horizontal y labels
                        
                        # Set "ylabel_rotation" (in degrees => can be: "90", etc...)
                        ylabel_rotation = -90 # Horizontal y labels

                        axes[c].tick_params(axis="x", labelsize=labelsize, labelcolor=axe_label_color, pad=1, rotation=xlabel_rotation)
                        axes[c].tick_params(axis="y", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=ylabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        axes[c].set_xlim(fixed_limits[0])

                        df_index += 1

                # Despine and save plot
                sns.despine(offset={"bottom": 2.5}, top=True, right=True)
                sns_plot.figure.savefig(outpath, dpi=300, format=extension)
            else:
                plt.rcParams["ytick.left"] = True
                plt.rcParams["ytick.major.size"] = 1
                plt.rcParams["ytick.major.width"] = .5
                plt.rcParams["ytick.color"] = axe_color

                # Draw the full plot            
                (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios, layout="constrained")
                #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...
                                
                df_join = []
                if row > 1: # Arrange plots in rows
                    for r in range(row):
                        if (r % 2) == 0: df_even = df[0::2]
                        else: df_odd = df[1::2]

                    df_join = df_even + df_odd
                else:
                    df_join = df

                df_index = 0
                if row == 1 and col == 1:
                    split_by_tdt_status = False
                    if split_by_tdt_status: # Change boolean above...
                        if (tissue_selection == "ST&TdT_DetailedTH_Layer1"):
                            for _order in order:
                                if  _order == ["TH1 (TdT–)", "TH1 (TdT+)", "TH2 (TdT–)", "TH2 (TdT+)", "TH17 (TdT–)", "TH17 (TdT+)"]:
                                    df_join[0]["TdT+ Status"] = np.where(df_join[0]["Phenotypes"].apply(lambda s: "TdT+" in s), True, False)
                                    df_join[0]["Phenotypes Renamed"] = df_join[0]["Phenotypes"]

                                    for str_order in _order:
                                        str_order_replace = str_order.split(" ")[0]
                                        df_join[0]['Phenotypes Renamed'].mask(df_join[0]['Phenotypes Renamed'] == str_order, str_order_replace, inplace=True)

                                    # Print dataframe for safety...
                                    #print(df_join[0].to_string()) # Get whole dataframe
                                if  _order == ["TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                                    df_join[0]["TdT+ Status"] = np.where(df_join[0]["Phenotypes"].apply(lambda s: "TdT+" in s), True, False)
                                    df_join[0]["Phenotypes Renamed"] = df_join[0]["Phenotypes"]

                                    for str_order in _order:
                                        if "TdT" in str_order: str_order_replace = str_order.split(" ")[0]
                                        else: str_order_replace = str_order
                                        
                                        df_join[0]['Phenotypes Renamed'].mask(df_join[0]['Phenotypes Renamed'] == str_order, str_order_replace, inplace=True)

                                    # Print dataframe for safety...
                                    #print(df_join[0].to_string()) # Get whole dataframe
                    
                    if (split_by_tdt_status): # Change boolean above...
                        sns_plot = sns.violinplot(data=df_join[0], ax=axes, x="Phenotypes Renamed", y=value_name, bw_adjust=.55, cut=2, hue="TdT+ Status", hue_order=[False, True], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                        sns.stripplot(data=df_join[0], ax=axes, x="Phenotypes Renamed", y=value_name, hue="TdT+ Status", palette=stripplot_color_palette, size=0.75, legend=False)
                    else:
                        sns_plot = sns.violinplot(data=df_join[0], ax=axes, x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[0], palette=customPalettes[0], linewidth=.15, fill=True, split=False, order=order[0], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[0], ax=axes, x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    if tissue_selection == "AA+DA+DC_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "SG+SR_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "DC&CD3_ILC3-Only_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": xlabel_rotation = 90 # Horizontal x labels
                    elif tissue_selection == "ST&TdT_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "ST&TdT_PooledTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                    elif tissue_selection == "ST&TdT_DetailedTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                    elif tissue_selection == "FL+AA+SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "SG+FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                    elif tissue_selection == "SR_Layer1": xlabel_rotation = 0 # Horizontal x labels

                    axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes.tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    #axes.set_ylabel(value_name, size=4, weight="bold")
                    axes.set_ylabel("")
                    axes.set_xlabel("")

                    axes.spines["left"].set_linewidth(.5)
                    axes.spines["left"].set_color(axe_color)

                    axes.spines["bottom"].set_linewidth(.5)
                    axes.spines["bottom"].set_color(axe_color)

                    axes.set_ylim(fixed_limits[0])
                elif row > 1 and col > 1:
                    for r in range(row):
                        for c in range(col):
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[r], palette=customPalettes[r], linewidth=.15, fill=True, split=False, order=order[r], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                            sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                            # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                            if tissue_selection == "AA+DA+DC_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "SG+SR_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "DC&CD3_ILC3-Only_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": xlabel_rotation = 90 # Horizontal x labels
                            elif tissue_selection == "ST&TdT_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "ST&TdT_PooledTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                            elif tissue_selection == "ST&TdT_DetailedTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                            elif tissue_selection == "FL+AA+SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "SG+FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                            elif tissue_selection == "SR_Layer1": xlabel_rotation = 0 # Horizontal x labels

                            axes[r][c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                            axes[r][c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                            #axes[r][c].set_ylabel(value_name, size=4, weight="bold")
                            axes[r][c].set_ylabel("")
                            axes[r][c].set_xlabel("")

                            axes[r][c].spines["left"].set_linewidth(.5)
                            axes[r][c].spines["left"].set_color(axe_color)

                            axes[r][c].spines["bottom"].set_linewidth(.5)
                            axes[r][c].spines["bottom"].set_color(axe_color)

                            if r == 0: axes[r][c].set_ylim(fixed_limits[0])
                            elif r == 1: axes[r][c].set_ylim(fixed_limits[1])

                            df_index += 1
                elif row == 1:
                    for c in range(col):
                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[c], palette=customPalettes[c], linewidth=.15, fill=True, split=False, order=order[c], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        if tissue_selection == "AA+DA+DC_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG+SR_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "DC&CD3_ILC3-Only_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_PooledTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_DetailedTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "FL+AA+SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG+FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SR_Layer1": xlabel_rotation = 0 # Horizontal x labels

                        axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        if c == 0: axes[c].set_ylim(fixed_limits[0])
                        elif c == 1: axes[c].set_ylim(fixed_limits[1])

                        df_index += 1
                elif col == 1:
                    for r in range(row):
                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[r], palette=customPalettes[r], linewidth=.15, fill=True, split=False, order=order[r], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[r], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        if tissue_selection == "AA+DA+DC_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG+SR_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "DC&CD3_ILC3-Only_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_PooledTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "ST&TdT_DetailedTH_Layer1": xlabel_rotation = 90 # Horizontal x labels
                        elif tissue_selection == "FL+AA+SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG+FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SG_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "FL_Layer1": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_selection == "SR_Layer1": xlabel_rotation = 0 # Horizontal x labels

                        axes[r].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[r].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[r].set_ylabel(value_name, size=4, weight="bold")
                        axes[r].set_ylabel("")
                        axes[r].set_xlabel("")

                        axes[r].spines["left"].set_linewidth(.5)
                        axes[r].spines["left"].set_color(axe_color)

                        axes[r].spines["bottom"].set_linewidth(.5)
                        axes[r].spines["bottom"].set_color(axe_color)

                        if r == 0: axes[r].set_ylim(fixed_limits[0])
                        elif r == 1: axes[r].set_ylim(fixed_limits[1])

                        df_index += 1

                # Despine and save plot
                sns.despine(offset={"bottom": 2.5}, top=True, right=True)
                sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")
    else:
        # Dataset key values
        data_type = dataset_keys[1]
        output_type = dataset_keys[2]
        multiplex_type = dataset_keys[0]

        # ColorPalettes
        customPalette = {}
        if "single_all" in output_type:
            if multiplex_type == "multiplexMain":
                customPalette = {
                    "CD20+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "GATA+ Cells": sns_pal_hex[5],
                    "ICOS+ Cells": sns_pal_hex[6],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#1.1":
                customPalette = {
                    "CD3+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#1.2":
                customPalette = {
                    "CD3+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#2":
                customPalette = {
                    "TdT+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "GATA+ Cells": sns_pal_hex[5],
                    "ICOS+ Cells": sns_pal_hex[6],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
        elif "single_chunk" in output_type:
            if multiplex_type == "multiplexMain":
                customPalette = {
                    "CD20+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "GATA+ Cells": sns_pal_hex[5],
                    "ICOS+ Cells": sns_pal_hex[6],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#1.1":
                customPalette = {
                    "CD3+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#1.2":
                customPalette = {
                    "CD3+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
            elif multiplex_type == "multiplexAlt#2":
                customPalette = {
                    "TdT+ Cells": sns_pal_hex[0],
                    "TCRαδ+ Cells": sns_pal_hex[1],
                    "IL7R+ Cells": sns_pal_hex[2],
                    "EOMES+ Cells": sns_pal_hex[3],
                    "TBX+ Cells": sns_pal_hex[4],
                    "GATA+ Cells": sns_pal_hex[5],
                    "ICOS+ Cells": sns_pal_hex[6],
                    "RORγT+ Cells": sns_pal_hex[7],
                    "AHR+ Cells": sns_pal_hex[8]
                }
        elif "coloc" in output_type:
            if multiplex_type == "multiplexMain":
                if order == ["B", "T", "IL7R Non B/T"]:
                    customPalette = {
                        "B": sns_pal_hex[0],
                        "T": sns_pal_hex[1],
                        "IL7R Non B/T": sns_pal_hex_il7r_cells_tints[0],
                    }
                elif order == ["NK/ILC1ie", "ILC1", "TH1"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex[2],
                        "ILC1": sns_pal_hex[3],
                        "TH1": sns_pal_hex[4],
                    }
                elif order == ["ILC2", "TH2"]:
                    customPalette = {
                        "ILC2": sns_pal_hex[5],
                        "TH2": sns_pal_hex[6],
                    }
                elif order == ["ILC3", "TH17"]:
                    customPalette = {
                        "ILC3": sns_pal_hex[7],
                        "TH17": sns_pal_hex[8],
                    }
            elif multiplex_type == "multiplexAlt#1.1":
                if order == ["T", "IL7R Non T"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                    }
                elif order == ["ILC3", "TH17", "icCD3+ ILC3"]:
                    customPalette = {
                        "ILC3": sns_pal_hex[7],
                        "TH17": sns_pal_hex[8],
                        "icCD3+ ILC3": sns_pal_hex[9],
                    }
            elif multiplex_type == "multiplexAlt#1.2":
                if order == ["T", "IL7R Non T"]:
                    customPalette = {
                        "T": sns_pal_hex_halo_tints[5],
                        "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                    }
                elif order == ["NK/ILC1ie", "ILC1", "TH1", "icCD3+ NK/ILC1ie", "icCD3+ ILC1"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex[2],
                        "ILC1": sns_pal_hex[3],
                        "TH1": sns_pal_hex[4],
                        "icCD3+ NK/ILC1ie": sns_pal_hex[5],
                        "icCD3+ ILC1": sns_pal_hex[6],
                    }
                elif order == ["ILC3", "TH17", "icCD3+ ILC3"]:
                    customPalette = {
                        "ILC3": sns_pal_hex[7],
                        "TH17": sns_pal_hex[8],
                        "icCD3+ ILC3": sns_pal_hex[9],
                    }
            elif multiplex_type == "multiplexAlt#2":
                if order == ["Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]:
                    customPalette = {
                        "Immature T": sns_pal_hex_t_cells_tints[2],
                        "Mature T": sns_pal_hex_t_cells_tints[3],
                        "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                        "IL7R Immature T": sns_pal_hex_il7r_cells_tints[2],
                        "IL7R Mature T": sns_pal_hex_il7r_cells_tints[3],
                    }
                if order == ["NK/ILC1ie", "ILC1", "TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1"]:
                    customPalette = {
                        "NK/ILC1ie": sns_pal_hex[2],
                        "ILC1": sns_pal_hex[3],
                        "TH1 (TdT–)": sns_pal_hex_th1_cells_tints[2],
                        "TH1 (TdT+)": sns_pal_hex_th1_cells_tints[3],
                        "Immature TH1": sns_pal_hex_th1_cells_tints[4],
                    }
                if order == ["ILC2", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2"]:
                    customPalette = {
                        "ILC2": sns_pal_hex[5],
                        "TH2 (TdT–)": sns_pal_hex_th2_cells_tints[2],
                        "TH2 (TdT+)": sns_pal_hex_th2_cells_tints[3],
                        "Immature TH2": sns_pal_hex_th2_cells_tints[4],
                    }
                if order == ["ILC3", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                    customPalette = {
                        "ILC3": sns_pal_hex[7],
                        "TH17 (TdT–)": sns_pal_hex_th17_cells_tints[2],
                        "TH17 (TdT+)": sns_pal_hex_th17_cells_tints[3],
                        "Immature TH17": sns_pal_hex_th17_cells_tints[4]
                    }

        # Set hue order
        hue_order = order # Default...
        if multiplex_type == "multiplexMain":
            if order == ["NK/ILC1ie", "ILC1", "TH1"]:
                hue_order = ["ILC1", "TH1", "NK/ILC1ie"]

        # Create single plot
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot
            cm = 1/2.54  # Centimeters in inches (for "figsize")
            if len(order) <= 3: (figure, axes) = plt.subplots(row, col, figsize=(1*cm*len(order), 3*cm), layout="constrained")
            elif len(order) > 3 and len(order) <= 5: (figure, axes) = plt.subplots(row, col, figsize=(1*cm*len(order), 3*cm), layout="constrained")
            elif len(order) > 5 and len(order) <= 9: (figure, axes) = plt.subplots(row, col, figsize=(1*cm*len(order), 3*cm), layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            sns_plot = sns.violinplot(data=df, x="Phenotypes", y=value_name, bw_adjust=.55, cut=1.45, hue="Phenotypes", hue_order=hue_order, palette=customPalette, linewidth=.15, fill=True, split=False, order=order, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))

            # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
            if len(order) <= 5: xlabel_rotation = 0 # Horizontal x labels
            if len(order) > 5 and len(order) <= 9: xlabel_rotation = 90 # Vertical x labels

            sns_plot.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
            sns_plot.tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
            #sns_plot.set_ylabel(value_name, size=4, weight="bold")
            sns_plot.set_ylabel("")
            sns_plot.set_xlabel("")

            sns_plot.spines["left"].set_linewidth(.5)
            sns_plot.spines["left"].set_color(axe_color)

            sns_plot.spines["bottom"].set_linewidth(.5)
            sns_plot.spines["bottom"].set_color(axe_color)

            if "single_all" in output_type:
                if data_type == "percentage":
                    if multiplex_type == "multiplexMain": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#1.1": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#1.2": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#2": sns_plot.set_ylim(-15.9, 120)
            elif "single_chunk" in output_type:
                if data_type == "percentage":
                    if multiplex_type == "multiplexMain": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#1.1": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#1.2": sns_plot.set_ylim(-15.9, 120)
                    elif multiplex_type == "multiplexAlt#2": sns_plot.set_ylim(-15.9, 120)
            elif "coloc" in output_type:
                if data_type == "percentage":
                    if multiplex_type == "multiplexMain":
                        if order == ["B", "T", "IL7R Non B/T"]: sns_plot.set_ylim(-9.9, 80)
                        elif order == ["NK/ILC1ie", "ILC1", "TH1"]: sns_plot.set_ylim(-5.9, 50)
                        elif order == ["ILC2", "TH2"]: sns_plot.set_ylim(-5.9, 50)
                        elif order == ["ILC3", "TH17"]: sns_plot.set_ylim(-5.9, 50)
                    elif multiplex_type == "multiplexAlt#1.1":
                        if order == ["T", "IL7R Non T"]: sns_plot.set_ylim(-9.9, 80)
                        elif order == ["ILC3", "TH17", "icCD3+ ILC3"]: sns_plot.set_ylim(-5.9, 50)
                    elif multiplex_type == "multiplexAlt#1.2":
                        if order == ["T", "IL7R Non T"]: sns_plot.set_ylim(-9.9, 80)
                        elif order == ["NK/ILC1ie", "ILC1", "TH1", "icCD3+ NK/ILC1ie", "icCD3+ ILC1"]: sns_plot.set_ylim(-5.9, 50)
                        elif order == ["ILC3", "TH17", "icCD3+ ILC3"]: sns_plot.set_ylim(-5.9, 50)
                    elif multiplex_type == "multiplexAlt#2":
                        if order == ["Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]: sns_plot.set_ylim(-9.9, 80)
                        if order == ["NK/ILC1ie", "ILC1", "TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1"]: sns_plot.set_ylim(-3.9, 20)
                        if order == ["ILC2", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2"]: sns_plot.set_ylim(-3.9, 20)
                        if order == ["ILC3", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]: sns_plot.set_ylim(-3.9, 20)
                elif data_type == "density":  
                    if multiplex_type == "multiplexMain":
                        if order == ["B", "T", "IL7R Non B/T"]: sns_plot.set_ylim(-999, 10000)
                        elif order == ["NK/ILC1ie", "ILC1", "TH1"]: sns_plot.set_ylim(-99, 700)
                        elif order == ["ILC2", "TH2"]: sns_plot.set_ylim(-799, 5000)
                        elif order == ["ILC3", "TH17"]: sns_plot.set_ylim(-99, 700)
                    elif multiplex_type == "multiplexAlt#1.1":
                        if order == ["T", "IL7R Non T"]: sns_plot.set_ylim(-999, 10000)
                        elif order == ["ILC3", "TH17", "icCD3+ ILC3"]: sns_plot.set_ylim(-99, 700)
                    elif multiplex_type == "multiplexAlt#1.2":
                        if order == ["T", "IL7R Non T"]: sns_plot.set_ylim(-999, 10000)
                        elif order == ["NK/ILC1ie", "ILC1", "TH1", "icCD3+ NK/ILC1ie", "icCD3+ ILC1"]: sns_plot.set_ylim(-99, 700)
                        elif order == ["ILC3", "TH17", "icCD3+ ILC3"]: sns_plot.set_ylim(-99, 700)
                    elif multiplex_type == "multiplexAlt#2":
                        if order == ["Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]: sns_plot.set_ylim(-999, 10000)
                        if order == ["NK/ILC1ie", "ILC1", "TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1"]: sns_plot.set_ylim(-29, 200)
                        if order == ["ILC2", "TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2"]: sns_plot.set_ylim(-399, 3000)
                        if order == ["ILC3", "TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]: sns_plot.set_ylim(-399, 3000)

            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_paper_1(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_halo_tints_icCD3 = ["#7891b1", "#fabb93", "#bad1ed", "#e266e1", "#eaece5", "#d0d0aa"] # Tints #4 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    if (tissue_selection == "B+T+IL7R"):
        #print(df)
        #print(len(df))
        #print(order)

        # Concatenate ST/ST&TdT data
        dfs_1 = [] # Init an empty array
        dfs_2 = [] # Init an empty array
        dfs_3 = [] # Init an empty array
        df_ST_Medulla = [] # Init an empty array
        df_ST_Cortex = [] # Init an empty array

        for idx, dataframe in enumerate(df):
            if idx > 3: dfs_1.append(dataframe) # SG+AA+DA+DC+SR

            if idx == 0: df_ST_Medulla.append(dataframe) # ST Medulla
            if idx == 2: df_ST_Medulla.append(dataframe) # ST Medulla

            if idx == 1: df_ST_Cortex.append(dataframe) # ST Cortex
            if idx == 3: df_ST_Cortex.append(dataframe) # ST Cortex

        dfs_2.append(pd.concat(df_ST_Medulla, ignore_index=True))
        dfs_3.append(pd.concat(df_ST_Cortex, ignore_index=True))
        df = dfs_2 + dfs_3 + dfs_1

        #print(df)
        #print(len(df))
        
        # Concatenate ST/ST&TdT orders
        orders_1 = [] # Init an empty array
        orders_2 = [] # Init an empty array

        for idx, _order in enumerate(order):
            if idx == 2: orders_1 = _order
            if idx < 2: orders_2.append(_order)

        orders_2 = sum(orders_2, [])
        order = [orders_1, orders_2]
        
        #print(order)

        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if _order == ["B", "T", "IL7R Non B/T"]:
                customPalette = {
                    "B": sns_pal_hex_halo[4],
                    "T": sns_pal_hex_halo_tints[5],
                    "IL7R Non B/T": sns_pal_hex_il7r_cells_tints[0],
                }
                customPalettes.append(customPalette)
            if _order == ["B", "Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]:
                customPalette = {
                    "B": sns_pal_hex_halo[4],
                    "Immature T": sns_pal_hex_t_cells_tints[2],
                    "Mature T": sns_pal_hex_t_cells_tints[3],
                    "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                    "IL7R Immature T": sns_pal_hex_il7r_cells_tints[2],
                    "IL7R Mature T": sns_pal_hex_il7r_cells_tints[3],
                }
                customPalettes.append(customPalette)
        
        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (18*cm, 4*cm), [2.25, 1, 1, 1, 1, 1], [1]
    else:
        if (tissue_selection == "Relative_IL7R"):
            # ColorPalettes
            customPalette = {
                "ILC1": sns_pal_hex_halo_tints[1],
                "ILC2": sns_pal_hex_halo_tints[2],
                "ILC3": sns_pal_hex_halo_tints[3],
            }

            # Figure axes share + size + ratios
            cm = 1/2.54  # Centimeters in inches (for "figsize")
            sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (7.75*cm, 9*cm), [1, 1], [1, 1, 1]
        else:
            # ColorPalettes
            customPalette = {}
            customPalettes = []
            for _order in order:
                if (tissue_selection == "SR"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC2": sns_pal_hex_halo_tints[2],
                            "ILC3": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH2", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH2": sns_pal_hex_halo_tints[2],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "ST&TdT_PooledTH"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC2": sns_pal_hex_halo_tints[2],
                            "ILC3": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH2", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH2": sns_pal_hex_halo_tints[2],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "ST&TdT_DetailedTH"):
                    if  _order == ["TH1 TdT–", "TH1 TdT+", "Immature TH1"]:
                        customPalette = {
                            "TH1 TdT–": sns_pal_hex_th1_cells_tints[2],
                            "TH1 TdT+": sns_pal_hex_th1_cells_tints[3],
                            "Immature TH1": sns_pal_hex_th1_cells_tints[4],
                        }
                        customPalettes.append(customPalette)
                    if  _order == ["TH2 TdT–", "TH2 TdT+", "Immature TH2"]:
                        customPalette = {
                            "TH2 TdT–": sns_pal_hex_th2_cells_tints[2],
                            "TH2 TdT+": sns_pal_hex_th2_cells_tints[3],
                            "Immature TH2": sns_pal_hex_th2_cells_tints[4],
                        }
                        customPalettes.append(customPalette)
                    if  _order == ["TH17 TdT–", "TH17 TdT+", "Immature TH17"]:
                        customPalette = {
                            "TH17 TdT–": sns_pal_hex_th17_cells_tints[2],
                            "TH17 TdT+": sns_pal_hex_th17_cells_tints[3],
                            "Immature TH17": sns_pal_hex_th17_cells_tints[4]
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "DC&CD3_ILC1+ILC3"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC3", "icCD3+ NK/ILC1ie", "icCD3+ ILC1", "icCD3+ ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC3": sns_pal_hex_halo_tints[3],
                            "icCD3+ NK/ILC1ie": sns_pal_hex_halo_tints_icCD3[0],
                            "icCD3+ ILC1": sns_pal_hex_halo_tints_icCD3[1],
                            "icCD3+ ILC3": sns_pal_hex_halo_tints_icCD3[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "DA+DC"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC2": sns_pal_hex_halo_tints[2],
                            "ILC3": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH2", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH2": sns_pal_hex_halo_tints[2],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "SG+AA"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC2": sns_pal_hex_halo_tints[2],
                            "ILC3": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH2", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH2": sns_pal_hex_halo_tints[2],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                elif (tissue_selection == "FL"):
                    if _order == ["NK/ILC1ie", "ILC1", "ILC2", "ILC3"]:
                        customPalette = {
                            "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                            "ILC1": sns_pal_hex_halo_tints[1],
                            "ILC2": sns_pal_hex_halo_tints[2],
                            "ILC3": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)
                    elif _order == ["TH1", "TH2", "TH17"]:
                        customPalette = {
                            "TH1": sns_pal_hex_halo_tints[1],
                            "TH2": sns_pal_hex_halo_tints[2],
                            "TH17": sns_pal_hex_halo_tints[3],
                        }
                        customPalettes.append(customPalette)

            # Figure axes share + size + ratios
            cm = 1/2.54  # Centimeters in inches (for "figsize")
            if (tissue_selection == "SR"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (10*cm, 4*cm), [1.5, 1], [1]
            elif (tissue_selection == "ST&TdT_PooledTH"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (6*cm, 3.75*cm), [1.5, 1], [1]
            elif (tissue_selection == "ST&TdT_DetailedTH"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (10.5*cm, 3.75*cm), [1, 1, 1], [1]
            elif (tissue_selection == "DC&CD3_ILC1+ILC3"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (9*cm, 3.75*cm), [2, 1], [1]
            elif (tissue_selection == "DA+DC"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (7.25*cm, 6*cm), [1, 1], [1, 1]
            elif (tissue_selection == "SG+AA"): sharey, sharex, figsize, width_ratios, height_ratios = "row", "none", (7.25*cm, 6*cm), [1, 1], [1, 1]
            elif (tissue_selection == "FL"): sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 7.25*cm), [1], [1, 1]

    # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
    try:
        # Set aesthetic parameters
        sns.set_theme(style=plot_style, font="Helvetica")
        sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
        sns.set_context("paper")

        # Get fonts as font in postscript filetypes
        plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica"]

        # Style x/y axis bottom/left ticks
        plt.rcParams["ytick.left"] = True
        plt.rcParams["ytick.major.size"] = 1
        plt.rcParams["ytick.major.width"] = .5
        plt.rcParams["ytick.color"] = axe_color

        # Draw the full plot            
        (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios, layout="constrained")
        #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

        if (tissue_selection == "B+T+IL7R"):
            # Print dataframe...
            #print(df)

            df_join = []
            N = 2 # Make couples
            for i in range(0, len(df), N):
                df_join.append(pd.concat(df[i:i + N], ignore_index=True))

            # Print joined dataframes...
            #print(df_join)

            df_index = 0
            for c in range(col):
                _customPalette = customPalettes[0]
                if c == 0: _customPalette = customPalettes[1]

                if df_index >= 2: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"]
                elif df_index == 1: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP", "RED PULP"]
                elif df_index == 0: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA", "CORTEX"]

                sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                # Custom colors + legend
                # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                handles = []
                colors = list(_customPalette.values())
                for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                    rgb = to_rgb(colors[ind // 2])
                    if ind % 2 != 0:
                        rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                    
                    violin.set_facecolor(rgb)
                    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                
                # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                xlabel_rotation = 90 # Horizontal x labels

                axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                #axes[c].set_ylabel(value_name, size=4, weight="bold")
                axes[c].set_ylabel("")
                axes[c].set_xlabel("")

                axes[c].spines["left"].set_linewidth(.5)
                axes[c].spines["left"].set_color(axe_color)

                axes[c].spines["bottom"].set_linewidth(.5)
                axes[c].spines["bottom"].set_color(axe_color)

                axes[c].set_ylim(fixed_limits[0])

                df_index += 1

            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        else:
            if (tissue_selection == "Relative_IL7R"):
                # Print dataframe...
                #print(df)

                df_join = []
                N = 2 # Make couples
                for i in range(0, len(df), N):
                    df_join.append(pd.concat(df[i:i + N], ignore_index=True))

                # Print joined dataframes...
                #print(df_join)

                df_index = 0
                for r in range(row):
                    for c in range(col):
                        if df_index >= 2: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"]
                        elif df_index == 1: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP", "RED PULP"]
                        elif df_index == 0: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA", "CORTEX"]

                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                        sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                        # Custom colors + legend
                        # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                        handles = []
                        colors = list(customPalette.values())
                        for ind, violin in enumerate(axes[r,c].findobj(PolyCollection)):
                            rgb = to_rgb(colors[ind // 2])
                            if ind % 2 != 0:
                                rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                            
                            violin.set_facecolor(rgb)
                            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                        
                        # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                        axes[r,c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 0 # Horizontal x labels

                        axes[r,c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[r,c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[r,c].set_ylabel(value_name, size=4, weight="bold")
                        axes[r,c].set_ylabel("")
                        axes[r,c].set_xlabel("")

                        axes[r,c].spines["left"].set_linewidth(.5)
                        axes[r,c].spines["left"].set_color(axe_color)

                        axes[r,c].spines["bottom"].set_linewidth(.5)
                        axes[r,c].spines["bottom"].set_color(axe_color)

                        axes[r,c].set_ylim(fixed_limits[0])

                        df_index += 1

                # Despine and save plot
                sns.despine(offset={"bottom": 2.5}, top=True, right=True)
                sns_plot.figure.savefig(outpath, dpi=300, format=extension)
            else:
                # Print dataframe...
                #print(df)

                df_join = []
                if (tissue_selection == "SR"):
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))
                elif (tissue_selection == "ST&TdT_PooledTH"):
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))
                elif (tissue_selection == "ST&TdT_DetailedTH"):
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))
                elif (tissue_selection == "DC&CD3_ILC1+ILC3"):
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))
                elif (tissue_selection == "DA+DC"):
                    for r in range(row):
                        N = 2 # Make couples
                        for i in range(r, len(df), N * 2):
                            df_join.append(pd.concat([df[i], df[i + N]], ignore_index=True))
                elif (tissue_selection == "SG+AA"):
                    for r in range(row):
                        N = 2 # Make couples
                        for i in range(r, len(df), N * 2):
                            df_join.append(pd.concat([df[i], df[i + N]], ignore_index=True))
                elif (tissue_selection == "FL"):
                    for r in range(row):
                        df_join.append(pd.concat(df[r::len(range(row))], ignore_index=True))

                # Print joined dataframes...
                #print(df_join)
                
                df_index = 0
                if (tissue_selection == "SR"):
                    for c in range(col):
                        split_by_analysis_region = True

                        if split_by_analysis_region: # Change boolean above...
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["White pulp", "Red pulp"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalettes[c].values())
                            for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["WHITE PULP", "RED PULP"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 0 # Horizontal x labels

                        axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        axes[c].set_ylim(fixed_limits[0])

                        df_index += 1
                elif (tissue_selection == "ST&TdT_PooledTH"):
                    for c in range(col):
                        split_by_analysis_region = True

                        if split_by_analysis_region: # Change boolean above...
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["Medulla", "Cortex"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalettes[c].values())
                            for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["MEDULLA", "CORTEX"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 0 # Horizontal x labels

                        axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        axes[c].set_ylim(fixed_limits[0])

                        df_index += 1
                elif (tissue_selection == "ST&TdT_DetailedTH"):
                    for c in range(col):
                        split_by_analysis_region = True

                        if split_by_analysis_region: # Change boolean above...
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["Medulla", "Cortex"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalettes[c].values())
                            for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["MEDULLA", "CORTEX"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 90 # Horizontal x labels

                        axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        axes[c].set_ylim(fixed_limits[0])

                        df_index += 1
                elif (tissue_selection == "DC&CD3_ILC1+ILC3"):
                    for c in range(col):
                        split_by_analysis_region = True

                        if split_by_analysis_region: # Change boolean above...
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["Germinal center", "Extra-follicular zone"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalettes[c].values())
                            for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 90 # Horizontal x labels

                        axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[c].set_ylabel(value_name, size=4, weight="bold")
                        axes[c].set_ylabel("")
                        axes[c].set_xlabel("")

                        axes[c].spines["left"].set_linewidth(.5)
                        axes[c].spines["left"].set_color(axe_color)

                        axes[c].spines["bottom"].set_linewidth(.5)
                        axes[c].spines["bottom"].set_color(axe_color)

                        if c == 0: axes[c].set_ylim(fixed_limits[0])
                        elif c == 1: axes[c].set_ylim(fixed_limits[1])

                        df_index += 1
                elif (tissue_selection == "DA+DC"):
                    for r in range(row):
                        for c in range(col):
                            split_by_analysis_region = True

                            if split_by_analysis_region: # Change boolean above...
                                hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"]

                                sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                                sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                                # Custom colors + legend
                                # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                                handles = []
                                colors = list(customPalettes[r].values())
                                for ind, violin in enumerate(axes[r,c].findobj(PolyCollection)):
                                    rgb = to_rgb(colors[ind // 2])
                                    if ind % 2 != 0:
                                        rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                    
                                    violin.set_facecolor(rgb)
                                    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                                
                                # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                                axes[r,c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                            # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                            xlabel_rotation = 0 # Horizontal x labels

                            axes[r][c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                            axes[r][c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                            #axes[r][c].set_ylabel(value_name, size=4, weight="bold")
                            axes[r][c].set_ylabel("")
                            axes[r][c].set_xlabel("")

                            axes[r][c].spines["left"].set_linewidth(.5)
                            axes[r][c].spines["left"].set_color(axe_color)

                            axes[r][c].spines["bottom"].set_linewidth(.5)
                            axes[r][c].spines["bottom"].set_color(axe_color)

                            if r == 0: axes[r][c].set_ylim(fixed_limits[0])
                            elif r == 1: axes[r][c].set_ylim(fixed_limits[1])

                            df_index += 1
                elif (tissue_selection == "SG+AA"):
                    for r in range(row):
                        for c in range(col):
                            split_by_analysis_region = True

                            if split_by_analysis_region: # Change boolean above...
                                hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"]

                                sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                                sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                                # Custom colors + legend
                                # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                                handles = []
                                colors = list(customPalettes[r].values())
                                for ind, violin in enumerate(axes[r,c].findobj(PolyCollection)):
                                    # There is no violin object for NK/ILC1ie - SG - Germinal center since the the density is null for all slides...
                                    # Need to tweak this to get the right colors...
                                    
                                    if (r == 0 and c == 0): # SG ILCs plot...
                                        if ind == 0: rgb = to_rgb(colors[0]) # Set "rgb" manually...
                                        if ind == 1: rgb = to_rgb(colors[1]) # Set "rgb" manually...
                                        if ind == 2: rgb = to_rgb(colors[1]) # Set "rgb" manually...
                                        if ind == 3: rgb = to_rgb(colors[2]) # Set "rgb" manually...
                                        if ind == 4: rgb = to_rgb(colors[2]) # Set "rgb" manually...
                                        if ind == 5: rgb = to_rgb(colors[3]) # Set "rgb" manually...
                                        if ind == 6: rgb = to_rgb(colors[3]) # Set "rgb" manually...

                                        if ind % 2 == 0:
                                            rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                        
                                        violin.set_facecolor(rgb)
                                        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                                    else:
                                        rgb = to_rgb(colors[ind // 2])
                                        if ind % 2 != 0:
                                            rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                        
                                        violin.set_facecolor(rgb)
                                        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                                
                                # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                                axes[r,c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                            # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                            xlabel_rotation = 0 # Horizontal x labels

                            axes[r][c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                            axes[r][c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                            #axes[r][c].set_ylabel(value_name, size=4, weight="bold")
                            axes[r][c].set_ylabel("")
                            axes[r][c].set_xlabel("")

                            axes[r][c].spines["left"].set_linewidth(.5)
                            axes[r][c].spines["left"].set_color(axe_color)

                            axes[r][c].spines["bottom"].set_linewidth(.5)
                            axes[r][c].spines["bottom"].set_color(axe_color)

                            if r == 0: axes[r][c].set_ylim(fixed_limits[0])
                            elif r == 1: axes[r][c].set_ylim(fixed_limits[1])

                            df_index += 1
                elif (tissue_selection == "FL"):
                    for r in range(row):
                        split_by_analysis_region = True

                        if split_by_analysis_region: # Change boolean above...
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["LymphomatousFollicle", "LF Excluded"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[r], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalettes[r].values())
                            for ind, violin in enumerate(axes[r].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[r].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["FOLLICULAR ZONE", "EXTRA-FOLLICULAR ZONE"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        xlabel_rotation = 0 # Horizontal x labels

                        axes[r].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[r].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[r].set_ylabel(value_name, size=4, weight="bold")
                        axes[r].set_ylabel("")
                        axes[r].set_xlabel("")

                        axes[r].spines["left"].set_linewidth(.5)
                        axes[r].spines["left"].set_color(axe_color)

                        axes[r].spines["bottom"].set_linewidth(.5)
                        axes[r].spines["bottom"].set_color(axe_color)

                        if r == 0: axes[r].set_ylim(fixed_limits[0])
                        elif r == 1: axes[r].set_ylim(fixed_limits[1])

                        df_index += 1

                # Despine and save plot
                sns.despine(offset={"bottom": 2.5}, top=True, right=True)
                sns_plot.figure.savefig(outpath, dpi=300, format=extension)
    except:
         print("Exceptions occurred...")

def plot_dataframe_to_file_figures_B_T_IL7R(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        data_type = dataset_keys[1]
        tissue_structure = dataset_keys[0]

        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if _order == ["B", "T", "IL7R Non B/T"]:
                customPalette = {
                    "B": sns_pal_hex_halo[4],
                    "T": sns_pal_hex_halo_tints[5],
                    "IL7R Non B/T": sns_pal_hex_il7r_cells_tints[0],
                }
                customPalettes.append(customPalette)
            if _order == ["Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]:
                customPalette = {
                    "Immature T": sns_pal_hex_t_cells_tints[2],
                    "Mature T": sns_pal_hex_t_cells_tints[3],
                    "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                    "IL7R Immature T": sns_pal_hex_il7r_cells_tints[2],
                    "IL7R Mature T": sns_pal_hex_il7r_cells_tints[3],
                }
                customPalettes.append(customPalette)

        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        sharey, sharex, figsize, width_ratios = "row", "none", (18*cm, 4*cm), [1,1,1,1,1,2]

        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            df_join = []
            if (tissue_structure == "Split"): # Arrange plots in cols
                N = 2 # Make couples
                if row == 1 and col > 1:
                    for i in range(0, len(df), N):
                        df_join.append(pd.concat(df[i:i + N], ignore_index=True))
            else:
                if col > 1:
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))

            df_index = 0
            if row == 1 and col == 6:
                for c in range(col):
                    if (tissue_structure == "Split"):
                        _customPalette = customPalettes[0]
                        if c == 5: _customPalette = customPalettes[1]

                        if df_index <= 3: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["GERMINAL CENTER | LEFT", "EXTRA-FOLLICULAR ZONE | RIGHT"]
                        elif df_index == 4: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP | LEFT", "RED PULP | RIGHT"]
                        elif df_index == 5: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA | LEFT", "CORTEX | RIGHT"]

                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                        # Custom colors + legend
                        # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                        handles = []
                        colors = list(_customPalette.values())
                        for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                            rgb = to_rgb(colors[ind // 2])
                            if ind % 2 != 0:
                                rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                            
                            violin.set_facecolor(rgb)
                            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                        
                        # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                        axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)
                    else:
                        _order = order[0]
                        if c == 5: _order = order[1]
                        _customPalette = customPalettes[0]
                        if c == 5: _customPalette = customPalettes[1]

                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=_order, palette=_customPalette, linewidth=.15, fill=True, split=False, order=_order, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    if tissue_structure == "WholeTissue": xlabel_rotation = 90 # Horizontal x labels
                    elif tissue_structure == "Split": xlabel_rotation = 90 # Horizontal x labels

                    axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    #axes[c].set_ylabel(value_name, size=4, weight="bold")
                    axes[c].set_ylabel("")
                    axes[c].set_xlabel("")

                    axes[c].spines["left"].set_linewidth(.5)
                    axes[c].spines["left"].set_color(axe_color)

                    axes[c].spines["bottom"].set_linewidth(.5)
                    axes[c].spines["bottom"].set_color(axe_color)

                    axes[c].set_ylim(fixed_limits[0])

                    df_index += 1

            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_figures_TCR_Relative(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        tissue_structure = dataset_keys[0]
       
        # ColorPalettes
        customPalette = {
            "TH1": sns_pal_hex_halo_tints[1],
            "TH2": sns_pal_hex_halo_tints[2],
            "TH17": sns_pal_hex_halo_tints[3],
        }
        
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        if (tissue_structure == "WholeTissue"): sharey, sharex, figsize, width_ratios = "row", "none", (9*cm, 6*cm), [1, 1, 1]
        elif (tissue_structure == "Split"): sharey, sharex, figsize, width_ratios = "row", "none", (9*cm, 6*cm), [1, 1, 1]

        # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            df_join = []
            if (tissue_structure == "Split"): # Arrange plots in cols
                N = 2 # Make couples
                if row > 1 and col > 1:
                    for i in range(0, len(df), N):
                        df_join.append(pd.concat(df[i:i + N], ignore_index=True))
            else:
                df_join = df
            
            df_index = 0
            if row > 1 and col > 1:
                for r in range(row):
                    for c in range(col):
                        if (tissue_structure == "Split"):
                            if df_index <= 3: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["GERMINAL CENTER | LEFT", "EXTRA-FOLLICULAR ZONE | RIGHT"]
                            elif df_index == 4: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP | LEFT", "RED PULP | RIGHT"]
                            elif df_index == 5: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA | LEFT", "CORTEX | RIGHT"]

                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalette.values())
                            for ind, violin in enumerate(axes[r,c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                            
                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[r,c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)
                        else:
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[0], palette=customPalette, linewidth=.15, fill=True, split=False, order=order[0], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                            sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        if tissue_structure == "WholeTissue": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_structure == "Split": xlabel_rotation = 0 # Horizontal x labels

                        axes[r][c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[r][c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[r][c].set_ylabel(value_name, size=4, weight="bold")
                        axes[r][c].set_ylabel("")
                        axes[r][c].set_xlabel("")

                        axes[r][c].spines["left"].set_linewidth(.5)
                        axes[r][c].spines["left"].set_color(axe_color)

                        axes[r][c].spines["bottom"].set_linewidth(.5)
                        axes[r][c].spines["bottom"].set_color(axe_color)

                        axes[r][c].set_ylim(fixed_limits[0])

                        df_index += 1
            
            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_figures_IL7R_Relative(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        tissue_structure = dataset_keys[0]
       
        # ColorPalettes
        customPalette = {
            "ILC1": sns_pal_hex_halo_tints[1],
            "ILC2": sns_pal_hex_halo_tints[2],
            "ILC3": sns_pal_hex_halo_tints[3],
        }
        
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        if (tissue_structure == "WholeTissue"): sharey, sharex, figsize, width_ratios = "row", "none", (9*cm, 6*cm), [1, 1, 1]
        elif (tissue_structure == "Split"): sharey, sharex, figsize, width_ratios = "row", "none", (9*cm, 6*cm), [1, 1, 1]

        # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            df_join = []
            if (tissue_structure == "Split"): # Arrange plots in cols
                N = 2 # Make couples
                if row > 1 and col > 1:
                    for i in range(0, len(df), N):
                        df_join.append(pd.concat(df[i:i + N], ignore_index=True))
            else:
                df_join = df
            
            df_index = 0
            if row > 1 and col > 1:
                for r in range(row):
                    for c in range(col):
                        if (tissue_structure == "Split"):
                            if df_index <= 3: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["GERMINAL CENTER | LEFT", "EXTRA-FOLLICULAR ZONE | RIGHT"]
                            elif df_index == 4: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP | LEFT", "RED PULP | RIGHT"]
                            elif df_index == 5: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA | LEFT", "CORTEX | RIGHT"]

                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                            sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                            # Custom colors + legend
                            # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                            handles = []
                            colors = list(customPalette.values())
                            for ind, violin in enumerate(axes[r,c].findobj(PolyCollection)):
                                rgb = to_rgb(colors[ind // 2])
                                if ind % 2 != 0:
                                    rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                                
                                violin.set_facecolor(rgb)
                                handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                            
                            # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                            axes[r,c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)
                        else:
                            sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[0], palette=customPalette, linewidth=.15, fill=True, split=False, order=order[0], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                            sns.stripplot(data=df_join[df_index], ax=axes[r,c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                        # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                        if tissue_structure == "WholeTissue": xlabel_rotation = 0 # Horizontal x labels
                        elif tissue_structure == "Split": xlabel_rotation = 0 # Horizontal x labels

                        axes[r][c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                        axes[r][c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                        #axes[r][c].set_ylabel(value_name, size=4, weight="bold")
                        axes[r][c].set_ylabel("")
                        axes[r][c].set_xlabel("")

                        axes[r][c].spines["left"].set_linewidth(.5)
                        axes[r][c].spines["left"].set_color(axe_color)

                        axes[r][c].spines["bottom"].set_linewidth(.5)
                        axes[r][c].spines["bottom"].set_color(axe_color)

                        axes[r][c].set_ylim(fixed_limits[0])

                        df_index += 1
            
            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_figures_ST_DetailedTH_Split(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        data_type = dataset_keys[1]
        tissue_selection = dataset_keys[0]

        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if (tissue_selection == "ST&TdT_DetailedTH_Split_Cortex+Medulla"):
                if  _order == ["TH1 (TdT–)", "TH1 (TdT+)", "Immature TH1"]:
                    customPalette = {
                        "TH1 (TdT–)": sns_pal_hex_th1_cells_tints[2],
                        "TH1 (TdT+)": sns_pal_hex_th1_cells_tints[3],
                        "Immature TH1": sns_pal_hex_th1_cells_tints[4],
                    }
                    customPalettes.append(customPalette)
                if  _order == ["TH2 (TdT–)", "TH2 (TdT+)", "Immature TH2"]:
                    customPalette = {
                        "TH2 (TdT–)": sns_pal_hex_th2_cells_tints[2],
                        "TH2 (TdT+)": sns_pal_hex_th2_cells_tints[3],
                        "Immature TH2": sns_pal_hex_th2_cells_tints[4],
                    }
                    customPalettes.append(customPalette)
                if  _order == ["TH17 (TdT–)", "TH17 (TdT+)", "Immature TH17"]:
                    customPalette = {
                        "TH17 (TdT–)": sns_pal_hex_th17_cells_tints[2],
                        "TH17 (TdT+)": sns_pal_hex_th17_cells_tints[3],
                        "Immature TH17": sns_pal_hex_th17_cells_tints[4]
                    }
                    customPalettes.append(customPalette)

        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        if (tissue_selection == "ST&TdT_DetailedTH_Split_Cortex+Medulla"): sharey, sharex, figsize, width_ratios = "none", "none", (10.5*cm, 3.75*cm), [1, 1, 1]

        # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...
            
            df_join = []
            if col > 1: # Arrange plots in cols
                for c in range(col):
                    df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))

            df_index = 0
            if row == 1 and col == 3:
                for c in range(col):
                    split_by_analysis_region = True

                    if split_by_analysis_region: # Change boolean above...
                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=["Medulla", "Cortex"], palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                        # Custom colors + legend
                        # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                        handles = []
                        colors = list(customPalettes[c].values())
                        for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                            rgb = to_rgb(colors[ind // 2])
                            if ind % 2 != 0:
                                rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                            
                            violin.set_facecolor(rgb)
                            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))

                        # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                        axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=1, labels=["MEDULLA | LEFT", "CORTEX | RIGHT"], title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)
                    else:
                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[c], palette=customPalettes[c], linewidth=.15, fill=True, split=False, order=order[c], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    if tissue_selection == "ST&TdT_DetailedTH_Split_Cortex+Medulla": xlabel_rotation = 90 # Horizontal x labels

                    axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    #axes[c].set_ylabel(value_name, size=4, weight="bold")
                    axes[c].set_ylabel("")
                    axes[c].set_xlabel("")

                    axes[c].spines["left"].set_linewidth(.5)
                    axes[c].spines["left"].set_color(axe_color)

                    axes[c].spines["bottom"].set_linewidth(.5)
                    axes[c].spines["bottom"].set_color(axe_color)

                    if c == 0: axes[c].set_ylim(fixed_limits[0])
                    elif c == 1: axes[c].set_ylim(fixed_limits[1])
                    elif c == 2: axes[c].set_ylim(fixed_limits[2])

                    df_index += 1

                # Despine and save plot
                sns.despine(offset={"bottom": 2.5}, top=True, right=True)
                sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_figures_B_T_IL7R_ST_WithPanelCD20(value_name, dataset_keys, df, order, outpath, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray and order_isArray:
        # Dataset key values
        data_type = dataset_keys[1]
        tissue_structure = dataset_keys[0]

        #print(df)
        #print(len(df))
        #print(order)

        # Concatenate ST/ST&TdT data
        if tissue_structure == "WholeTissue":
            dfs_1 = [] # Init an empty array
            dfs_2 = [] # Init an empty array
            df_ST_WholeTissue = [] # Init an empty array

            for idx, dataframe in enumerate(df):
                if idx < 5: dfs_1.append(dataframe) # SG+AA+DA+DC+SR
                if idx >= 5: df_ST_WholeTissue.append(dataframe) # ST Whole Tissue

            dfs_2.append(pd.concat(df_ST_WholeTissue, ignore_index=True))
            df = dfs_1 + dfs_2
        elif tissue_structure == "Split":
            dfs_1 = [] # Init an empty array
            dfs_2 = [] # Init an empty array
            dfs_3 = [] # Init an empty array
            df_ST_Medulla = [] # Init an empty array
            df_ST_Cortex = [] # Init an empty array

            for idx, dataframe in enumerate(df):
                if idx < 10: dfs_1.append(dataframe) # SG+AA+DA+DC+SR

                if idx == 10: df_ST_Medulla.append(dataframe) # ST Medulla
                if idx == 12: df_ST_Medulla.append(dataframe) # ST Medulla

                if idx == 11: df_ST_Cortex.append(dataframe) # ST Cortex
                if idx == 13: df_ST_Cortex.append(dataframe) # ST Cortex

            dfs_2.append(pd.concat(df_ST_Medulla, ignore_index=True))
            dfs_3.append(pd.concat(df_ST_Cortex, ignore_index=True))
            df = dfs_1 + dfs_2 + dfs_3
        
        #print(df)
        #print(len(df))
        
        # Concatenate ST/ST&TdT orders
        orders_1 = [] # Init an empty array
        orders_2 = [] # Init an empty array

        for idx, _order in enumerate(order):
            if idx < 1: orders_1 = _order
            if idx >= 1: orders_2.append(_order)

        orders_2 = sum(orders_2, [])
        order = [orders_1, orders_2]
        
        #print(order)

        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if _order == ["B", "T", "IL7R Non B/T"]:
                customPalette = {
                    "B": sns_pal_hex_halo[4],
                    "T": sns_pal_hex_halo_tints[5],
                    "IL7R Non B/T": sns_pal_hex_il7r_cells_tints[0],
                }
                customPalettes.append(customPalette)
            if _order == ["B","Immature T", "Mature T", "IL7R Non T", "IL7R Immature T", "IL7R Mature T"]:
                customPalette = {
                    "B": sns_pal_hex_halo[4],
                    "Immature T": sns_pal_hex_t_cells_tints[2],
                    "Mature T": sns_pal_hex_t_cells_tints[3],
                    "IL7R Non T": sns_pal_hex_il7r_cells_tints[1],
                    "IL7R Immature T": sns_pal_hex_il7r_cells_tints[2],
                    "IL7R Mature T": sns_pal_hex_il7r_cells_tints[3],
                }
                customPalettes.append(customPalette)

        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        sharey, sharex, figsize, width_ratios = "row", "none", (18*cm, 4*cm), [1,1,1,1,1,2.25]

        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            # Style y axis left ticks
            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            df_join = []
            if (tissue_structure == "Split"): # Arrange plots in cols
                N = 2 # Make couples
                if row == 1 and col > 1:
                    for i in range(0, len(df), N):
                        df_join.append(pd.concat(df[i:i + N], ignore_index=True))
            else:
                if col > 1:
                    for c in range(col):
                        df_join.append(pd.concat(df[c::len(range(col))], ignore_index=True))

            df_index = 0
            if row == 1 and col == 6:
                for c in range(col):
                    if (tissue_structure == "Split"):
                        _customPalette = customPalettes[0]
                        if c == 5: _customPalette = customPalettes[1]

                        if df_index <= 3: hue_order, labels = ["Germinal center", "Extra-follicular zone"], ["GERMINAL CENTER | LEFT", "EXTRA-FOLLICULAR ZONE | RIGHT"]
                        elif df_index == 4: hue_order, labels = ["White pulp", "Red pulp"], ["WHITE PULP | LEFT", "RED PULP | RIGHT"]
                        elif df_index == 5: hue_order, labels = ["Medulla", "Cortex"], ["MEDULLA | LEFT", "CORTEX | RIGHT"]

                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Analysis Region", hue_order=hue_order, palette=["0", ".5"], linewidth=.15, fill=True, split=True, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"), legend=False)
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, hue="Analysis Region", palette=stripplot_color_palette, size=0.75, legend=False)

                        # Custom colors + legend
                        # => https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

                        handles = []
                        colors = list(_customPalette.values())
                        for ind, violin in enumerate(axes[c].findobj(PolyCollection)):
                            rgb = to_rgb(colors[ind // 2])
                            if ind % 2 != 0:
                                rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
                            
                            violin.set_facecolor(rgb)
                            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor=axe_color, linewidth=.15))
                        
                        # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
                        axes[c].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], loc=2, labels=labels, title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, shadow=None, frameon=False, facecolor="white", edgecolor=axe_color, framealpha=0.5, fontsize=labelsize-0.75)
                    else:
                        _order = order[0]
                        if c == 5: _order = order[1]
                        _customPalette = customPalettes[0]
                        if c == 5: _customPalette = customPalettes[1]

                        sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=_order, palette=_customPalette, linewidth=.15, fill=True, split=False, order=_order, density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                        sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y=value_name, color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    if tissue_structure == "WholeTissue": xlabel_rotation = 90 # Horizontal x labels
                    elif tissue_structure == "Split": xlabel_rotation = 90 # Horizontal x labels

                    axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    #axes[c].set_ylabel(value_name, size=4, weight="bold")
                    axes[c].set_ylabel("")
                    axes[c].set_xlabel("")

                    axes[c].spines["left"].set_linewidth(.5)
                    axes[c].spines["left"].set_color(axe_color)

                    axes[c].spines["bottom"].set_linewidth(.5)
                    axes[c].spines["bottom"].set_color(axe_color)

                    axes[c].set_ylim(fixed_limits[0])

                    df_index += 1

            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")

def plot_dataframe_to_file_figures_FlowCytometry(y_axis_legends, df, order, outpath, keyword, row=1, col=1, fixed_limits=False):
    # See: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    # Custom color palettes
    sns_pal = sns.color_palette("pastel") # Pick a seaborn color palette...
    sns_pal_hex = sns_pal.as_hex() # 10 colors "sns_pal_hex" (0-based index) => ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]

    sns_pal_hex_halo = ["#1f497d", "#f78f4c", "#8db3e2", "#d000ce", "#dde0d5", "#b2b172"] # Halo-based color palette (hex colors)
    sns_pal_hex_halo_tints = ["#4b6d97", "#f8a56f", "#a3c2e7", "#d932d7", "#e3e6dd", "#c1c08e"] # Tints #2 (0-based index) from "sns_pal_hex_halo" hex colors... => @ https://www.color-hex.com/
    sns_pal_hex_th17_cells_tints = ["#d000ce", "#d932d7", "#de4cdc", "#e266e1", "#e77fe6"]
    sns_pal_hex_th2_cells_tints = ["#8db3e2", "#a3c2e7", "#afc9ea", "#bad1ed", "#c6d9f0"]
    sns_pal_hex_th1_cells_tints = ["#f78f4c", "#f8a56f", "#f9b081", "#fabb93", "#fbc7a5"]
    sns_pal_hex_t_cells_tints = ["#b2b172", "#c1c08e", "#c9c89c", "#d0d0aa", "#d8d8b8"]
    sns_pal_hex_il7r_cells_tints = ["#84565c", "#9c777c", "#a8888c", "#b5999d"]

    df_isArray = isinstance(df, list)
    order_isArray = isinstance(order, list)

    if df_isArray:
        # ColorPalettes
        customPalette = {}
        customPalettes = []
        for _order in order:
            if _order == ["ILC1", "ILC2", "ILC3"]:
                customPalette = {
                    "ILC1": sns_pal_hex_halo_tints[1],
                    "ILC2": sns_pal_hex_halo_tints[2],
                    "ILC3": sns_pal_hex_halo_tints[3],
                }
                customPalettes.append(customPalette)
            elif _order == ["NK/ILC1ie"]:
                customPalette = {
                    "NK/ILC1ie": sns_pal_hex_halo_tints[0],
                }
                customPalettes.append(customPalette)

        # Figure axes share + size + ratios
        cm = 1/2.54  # Centimeters in inches (for "figsize")
        if keyword == "ILCs+NK": sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (6*cm, 3*cm), [3,1], [1]
        else: sharey, sharex, figsize, width_ratios, height_ratios = "none", "none", (3*cm, 3*cm), [1], [1]

        # Create multiple plots (https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)
        try:
            # Set aesthetic parameters
            sns.set_theme(style=plot_style, font="Helvetica")
            sns.set_style(plot_style, {"grid.color": grid_color, "grid.linestyle": "dotted"})
            sns.set_context("paper")

            # Get fonts as font in postscript filetypes
            plt.rcParams["ps.useafm"] = True # Get fonts as font => https://matplotlib.org/stable/users/explain/text/fonts.html
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica"]

            plt.rcParams["ytick.left"] = True
            plt.rcParams["ytick.major.size"] = 1
            plt.rcParams["ytick.major.width"] = .5
            plt.rcParams["ytick.color"] = axe_color

            # Draw the full plot            
            (figure, axes) = plt.subplots(row, col, sharey=sharey, sharex=sharex, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios, layout="constrained")
            #plt.setp(sns_plot.collections, alpha=.85) # Add some transparancy to the violin plot colors...

            df_join = df

            df_index = 0
            if keyword == "ILCs+NK":
                for c in range(col):
                    sns_plot = sns.violinplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y="Value", bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[c], palette=customPalettes[c], linewidth=.15, fill=True, split=False, order=order[c], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                    sns.stripplot(data=df_join[df_index], ax=axes[c], x="Phenotypes", y="Value", color=stripplot_color, size=0.75)

                    # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                    xlabel_rotation = 0 # Horizontal x labels

                    axes[c].tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                    axes[c].tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                    #axes[c].set_ylabel(value_name, size=4, weight="bold")
                    axes[c].set_ylabel("")
                    axes[c].set_xlabel("")

                    axes[c].spines["left"].set_linewidth(.5)
                    axes[c].spines["left"].set_color(axe_color)

                    axes[c].spines["bottom"].set_linewidth(.5)
                    axes[c].spines["bottom"].set_color(axe_color)

                    if c == 0: axes[c].set_ylim(fixed_limits[0])
                    elif c == 1: axes[c].set_ylim(fixed_limits[1])

                    df_index += 1
            else:
                if keyword == "ILCs": df_index = 0
                elif keyword == "NK": df_index = 1

                sns_plot = sns.violinplot(data=df_join[df_index], ax=axes, x="Phenotypes", y="Value", bw_adjust=.55, cut=2, hue="Phenotypes", hue_order=order[df_index], palette=customPalettes[df_index], linewidth=.15, fill=True, split=False, order=order[df_index], density_norm="count", inner_kws=dict(marker=".", box_width=4, whis_width=.75, color=".55"))
                sns.stripplot(data=df_join[df_index], ax=axes, x="Phenotypes", y="Value", color=stripplot_color, size=0.75)

                # Set "xlabel_rotation" (in degrees => can be: "90", etc...)
                xlabel_rotation = 0 # Horizontal x labels

                axes.tick_params(axis="y", labelsize=labelsize, labelcolor=axe_label_color, pad=1)
                axes.tick_params(axis="x", labelsize=labelsize+0.5, labelcolor=axe_label_color, pad=-3, rotation=xlabel_rotation)
                #axes.set_ylabel(value_name, size=4, weight="bold")
                axes.set_ylabel("")
                axes.set_xlabel("")

                axes.spines["left"].set_linewidth(.5)
                axes.spines["left"].set_color(axe_color)

                axes.spines["bottom"].set_linewidth(.5)
                axes.spines["bottom"].set_color(axe_color)

                axes.set_ylim(fixed_limits[0])

            # Despine and save plot
            sns.despine(offset={"bottom": 2.5}, top=True, right=True)
            sns_plot.figure.savefig(outpath, dpi=300, format=extension)
        except:
            print("Exceptions occurred...")
          
# DRAW FUNCTIONS
def draw_allData(dataset_dict, dataset_keys, row=1, col=1, fixed_limits=False):
    # Dataset key values
    data_type = dataset_keys[1]
    output_type = dataset_keys[2]
    multiplex_type = dataset_keys[0]

    # Draw...
    for dataset in dataset_dict[multiplex_type][data_type]:
        # Local dataset path
        file_path = dirname + "/DATASETS/" + dataset + ".csv"
        
        for idx, phenotype_dataframes in enumerate(create_phenotype_dataframes___AllData(dataset, dataset_keys, file_path)):
            value_name, df, order = phenotype_dataframes

            # Output directory
            output_dir = dirname + "/OUTPUT"
            if output_type == "single_all": output_filename = "violinPlot_" + multiplex_type + "_" + output_type + "_" + dataset.replace("_HighPlex_Query_Total_Summary_Results_", "_") + "_" + data_type
            else: output_filename = "violinPlot_" + multiplex_type + "_" + output_type + "_{}".format(idx + 1) + "_" + dataset.replace("_HighPlex_Query_Total_Summary_Results_", "_") + "_" + data_type
            outpath = os.path.join(output_dir, output_filename + "." + extension)

            plot_dataframe_to_file(value_name, dataset_keys, df, order, outpath, row, col, fixed_limits)

def draw_figures(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_" + tissue_selection + "_" + data_type
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for dataset in dataset_dict[tissue_selection][data_type]:
        # Local dataset path
        file_path = dirname + "/DATASETS/" + dataset + ".csv"
        
        for phenotype_dataframes in create_phenotype_dataframes___Figures(dataset, dataset_keys, file_path):
            value_name, df, order = phenotype_dataframes
            if not order in orders: orders.append(order)
            dfs.append(df)

    plot_dataframe_to_file(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_paper_1(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_Paper_1_" + tissue_selection + "_" + data_type
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Output directory (csv)
    source_data = 'sourceData_'
    csv_output_filename = f"{source_data}" + output_filename
    csv_outpath = os.path.join(output_dir, csv_output_filename + ".csv")

    # Draw...
    if (tissue_selection == "B+T+IL7R"):
        dfs = [] # Init an empty array
        orders = [] # Init an empty array
        for multiplex_type in ["multiplexAlt#2-BCells", "multiplexAlt#2-TCells", "multiplexMain"]:
            dataset_keys.append(multiplex_type)

            for dataset in dataset_dict[tissue_selection][data_type][multiplex_type]:
                # Local dataset path
                file_path = dirname + "/DATASETS/" + dataset + ".csv"

                for phenotype_dataframes in create_phenotype_dataframes___Paper_1(dataset, dataset_keys, file_path):
                    value_name, df, order = phenotype_dataframes
                    if not order in orders: orders.append(order)
                    dfs.append(df)

            dataset_keys.remove(multiplex_type)
    else:
        if (tissue_selection == "Relative_IL7R"):
            dfs = [] # Init an empty array
            orders = [] # Init an empty array
            for multiplex_type in ["multiplexAlt#2", "multiplexMain"]:
                dataset_keys.append(multiplex_type)

                for dataset in dataset_dict[tissue_selection][data_type][multiplex_type]:
                    # Local dataset path
                    file_path = dirname + "/DATASETS/" + dataset + ".csv"

                    for phenotype_dataframes in create_phenotype_dataframes___Paper_1(dataset, dataset_keys, file_path):
                        value_name, df, order = phenotype_dataframes
                        if not order in orders: orders.append(order)
                        dfs.append(df)

                dataset_keys.remove(multiplex_type)
        else:
            dfs = [] # Init an empty array
            orders = [] # Init an empty array
            for dataset in dataset_dict[tissue_selection][data_type]:
                # Local dataset path
                file_path = dirname + "/DATASETS/" + dataset + ".csv"
                
                for phenotype_dataframes in create_phenotype_dataframes___Paper_1(dataset, dataset_keys, file_path):
                    value_name, df, order = phenotype_dataframes
                    if not order in orders: orders.append(order)
                    dfs.append(df)

    # Get data source...
    if _get_source_data:
        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat = df_concat.dropna(subset=["# Cells / mm²"])
        
        # Simplify the "Image Tag" column
        split_string_1 = df_concat["Image Tag"].str.split("_")
        split_string_1_1 = split_string_1.str[2].str.split(".")

        part_1 = split_string_1.str[0]
        part_3 = split_string_1_1.str[1]
        part_2 = split_string_1_1.str[2].str.replace("Run", "")
        part_2 = part_2.str.replace("FL", "FOLLICULAR-LYMPHOMA")
        part_2 = part_2.str.replace("SG", "LYMPH-NODE")
        part_2 = part_2.str.replace("DA", "APPENDIX")
        part_2 = part_2.str.replace("AA", "TONSIL")
        part_2 = part_2.str.replace("SR", "SPLEEN")
        part_2 = part_2.str.replace("ST", "THYMUS")
        part_2 = part_2.str.replace("DC", "ILEUM")

        transform_string = part_1 + "_" + part_2 + "_" + part_3
        df_concat["Image Tag"] = transform_string
        
        #print(df_concat)

        df_concat.to_csv(csv_outpath, sep=",", encoding="utf-8", index=False, header=True) # Use "na_rep='NaN'" to keep NaN values in the CSV file 
    # --------------------------------------------------------------------------------------------------------------------------------------------

    plot_dataframe_to_file_paper_1(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_B_T_IL7R(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_structure = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_AA+DA+DC+SG+SR+ST&TdT_" + tissue_structure + "_" + data_type
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for multiplex_type in ["multiplexMain", "multiplexAlt#2"]:
        dataset_keys.append(multiplex_type)

        for dataset in dataset_dict[multiplex_type][tissue_structure][data_type]:
            # Local dataset path
            file_path = dirname + "/DATASETS/" + dataset + ".csv"

            for phenotype_dataframes in create_phenotype_dataframes___Figures_B_T_IL7R(dataset, dataset_keys, file_path):
                value_name, df, order = phenotype_dataframes
                if not order in orders: orders.append(order)
                dfs.append(df)

        dataset_keys.remove(multiplex_type)

    plot_dataframe_to_file_figures_B_T_IL7R(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_TCR_Relative(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    tissue_structure = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_AA+DA+DC+SG+SR+ST&TdT_" + tissue_structure + "_relative_%TCR"
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for multiplex_type in ["multiplexMain", "multiplexAlt#2"]:
        dataset_keys.append(multiplex_type)

        for dataset in dataset_dict[multiplex_type][tissue_structure]:
            # Local dataset path
            file_path = dirname + "/DATASETS/" + dataset + ".csv"

            for phenotype_dataframes in create_phenotype_dataframes___Figures_TCR_Relative(dataset, dataset_keys, file_path):
                value_name, df, order = phenotype_dataframes
                if not order in orders: orders.append(order)
                dfs.append(df)

        dataset_keys.remove(multiplex_type)

    plot_dataframe_to_file_figures_TCR_Relative(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_IL7R_Relative(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    tissue_structure = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_AA+DA+DC+SG+SR+ST&TdT_" + tissue_structure + "_relative_%IL7R"
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for multiplex_type in ["multiplexMain", "multiplexAlt#2"]:
        dataset_keys.append(multiplex_type)

        for dataset in dataset_dict[multiplex_type][tissue_structure]:
            # Local dataset path
            file_path = dirname + "/DATASETS/" + dataset + ".csv"

            for phenotype_dataframes in create_phenotype_dataframes___Figures_IL7R_Relative(dataset, dataset_keys, file_path):
                value_name, df, order = phenotype_dataframes
                if not order in orders: orders.append(order)
                dfs.append(df)

        dataset_keys.remove(multiplex_type)

    plot_dataframe_to_file_figures_IL7R_Relative(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_ST_DetailedTH_Split(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_selection = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_" + tissue_selection + "_" + data_type
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for dataset in dataset_dict[tissue_selection][data_type]:
        # Local dataset path
        file_path = dirname + "/DATASETS/" + dataset + ".csv"

        for phenotype_dataframes in create_phenotype_dataframes___Figures_ST_DetailedTH_Split(dataset, dataset_keys, file_path):
            value_name, df, order = phenotype_dataframes
            if not order in orders: orders.append(order)
            dfs.append(df)

    plot_dataframe_to_file_figures_ST_DetailedTH_Split(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_B_T_IL7R_ST_WithPanelCD20(dataset_dict, dataset_keys, row, col, fixed_limits):
    # Dataset key values
    data_type = dataset_keys[1]
    tissue_structure = dataset_keys[0]

    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_AA+DA+DC+SG+SR+ST&TdT(+BCells_Using_ST_WithPanelCD20)_" + tissue_structure + "_" + data_type
    outpath = os.path.join(output_dir, output_filename + "." + extension)

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array
    for multiplex_type in ["multiplexMain", "multiplexAlt#2-BCells", "multiplexAlt#2-TCells"]:
        dataset_keys.append(multiplex_type)

        for dataset in dataset_dict[multiplex_type][tissue_structure][data_type]:
            # Local dataset path
            file_path = dirname + "/DATASETS/" + dataset + ".csv"

            for phenotype_dataframes in create_phenotype_dataframes___Figures_B_T_IL7R_ST_WithPanelCD20(dataset, dataset_keys, file_path):
                value_name, df, order = phenotype_dataframes
                if not order in orders: orders.append(order)
                dfs.append(df)

        dataset_keys.remove(multiplex_type)

    plot_dataframe_to_file_figures_B_T_IL7R_ST_WithPanelCD20(value_name, dataset_keys, dfs, orders, outpath, row, col, fixed_limits)

def draw_figures_FlowCytometry(dataset, keyword, row, col, fixed_limits):
    # Output directory
    output_dir = dirname + "/OUTPUT"
    output_filename = "violinPlot_figures_FlowCytometry_"
    outpath = os.path.join(output_dir, output_filename + keyword + "." + extension)

    # Output directory (csv)
    source_data = 'sourceData_'
    csv_output_filename = f"{source_data}" + output_filename
    csv_outpath = os.path.join(output_dir, csv_output_filename + keyword + ".csv")

    # Draw...
    dfs = [] # Init an empty array
    orders = [] # Init an empty array

    # Axis legends / input_file
    y_axis_legends = ["% Of CD45+LIN- Cells", "% Of CD3-LIN-CD45+CD127+ Cells"]

    # Local dataset path
    input_file = dirname + "/DATASETS/" + dataset + ".csv"

    # Filter dataset columns
    reader_columns_1 = ["ID", "ILC1", "ILC2", "ILC3"]
    reader_columns_2 = ["ID", "NK/ILC1ie"]

    # Define dataset columns type
    dtypes_col_1 = {}
    dtypes_col_2 = {}
    for column in reader_columns_1:
        dtypes_col_1[column] = np.float64
    for column in reader_columns_2:
        dtypes_col_2[column] = np.float64

    # Read dataset and set indexes with panda
    df_1 = None # !important reset dataframe before reading csv
    df_2 = None # !important reset dataframe before reading csv
    df_1 = pd.read_csv(input_file, sep=",", usecols=reader_columns_1, dtype=dtypes_col_1)
    df_2 = pd.read_csv(input_file, sep=",", usecols=reader_columns_2, dtype=dtypes_col_2)

    # Print dataframe for safety...
    #print(df_1)
    #print(df_2)

    # Format dataframe for violin/box plots
    df_1 = df_1.melt(id_vars=["ID"], var_name="Phenotypes", value_name="Value")
    df_2 = df_2.melt(id_vars=["ID"], var_name="Phenotypes", value_name="Value")

    # Print dataframe for safety...
    #print(df_1)
    #print(df_2)

    dfs.append(df_1)
    dfs.append(df_2)

    orders.append(["ILC1", "ILC2", "ILC3"])
    orders.append(["NK/ILC1ie"])

    # Get data source...
    if _get_source_data:
        if keyword == "ILCs+NK":
            df_csv = dfs
            df_concat = pd.concat(df_csv, ignore_index=True)
        elif keyword == "ILCs":
            df_concat = df_1
        elif keyword == "NK":
            df_concat = df_2

        df_concat = df_concat.dropna(subset=["Value"])
               
        #print(df_concat)

        df_concat.to_csv(csv_outpath, sep=",", encoding="utf-8", index=False, header=True) # Use "na_rep='NaN'" to keep NaN values in the CSV file 
    # --------------------------------------------------------------------------------------------------------------------------------------------

    plot_dataframe_to_file_figures_FlowCytometry(y_axis_legends, dfs, orders, outpath, keyword, row, col, fixed_limits)

#-------
# DRAW !
#-------

# Global vars
_get_source_data = True # Set to "True" to get the source data (for figures paper and flow cytometry only)...

#-------------
# BATCH DRAW !
#-------------
_multiplex_type = [
    "multiplexMain",
    "multiplexAlt#1.1",
    "multiplexAlt#1.2",
    "multiplexAlt#2",
]
_tissue_selection = [
    "AA+DA+DC_Layer1",
    "SG+SR_Layer1",
    "DC&CD3_ILC3-Only_Layer1",
    "DC&CD3_ILC1+ILC3_Layer1",
    "ST&TdT_Layer1",
    "ST&TdT_PooledTH_Layer1",
    "ST&TdT_DetailedTH_Layer1",
    "FL+AA+SG_Layer1",
    "FL+AA+SG_NoT_Layer1",
    "SG+FL_Layer1",
    "SG+FL_NoT_Layer1",
    "SG_Layer1",
    "SG_NoT_Layer1",
    "FL_Layer1",
    "FL_NoT_Layer1",
    "SR_Layer1",
]

def batch_draw_allData():
    for multiplex_type in _multiplex_type:
        for data_type in ["percentage"]:
            for output_type in ["single_all", "single_chunked", "coloc"]:
                draw_allData(datasets_dict___AllData, [multiplex_type, data_type, output_type])
        for data_type in ["density"]:
            for output_type in ["coloc"]:
                draw_allData(datasets_dict___AllData, [multiplex_type, data_type, output_type])

def batch_draw_figures(): 
    for tissue_selection in _tissue_selection:
        if tissue_selection == "AA+DA+DC_Layer1": row, col = 2, 3
        elif tissue_selection == "SG+SR_Layer1": row, col = 2, 2
        elif tissue_selection == "DC&CD3_ILC3-Only_Layer1": row, col = 2, 1
        elif tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": row, col = 2, 1
        elif tissue_selection == "ST&TdT_Layer1": row, col = 1, 2
        elif tissue_selection == "ST&TdT_PooledTH_Layer1": row, col = 1, 2
        elif tissue_selection == "ST&TdT_DetailedTH_Layer1": row, col = 1, 1
        elif tissue_selection == "FL+AA+SG_Layer1": row, col = 2, 3
        elif tissue_selection == "FL+AA+SG_NoT_Layer1": row, col = 1, 3
        elif tissue_selection == "SG+FL_Layer1": row, col = 2, 2
        elif tissue_selection == "SG+FL_NoT_Layer1": row, col = 1, 2
        elif tissue_selection == "SG_Layer1": row, col = 1, 2
        elif tissue_selection == "SG_NoT_Layer1": row, col = 1, 1
        elif tissue_selection == "FL_Layer1": row, col = 1, 2
        elif tissue_selection == "FL_NoT_Layer1": row, col = 1, 1
        elif tissue_selection == "SR_Layer1": row, col = 1, 2

        for data_type in ["percentage", "density"]:
            if data_type == "percentage":
                if tissue_selection == "AA+DA+DC_Layer1": fixed_limits = [(-0.69, 6), (-5.9, 50)]
                if tissue_selection == "SG+SR_Layer1": fixed_limits = [(-1.09, 6), (-5.9, 50)]
                if tissue_selection == "DC&CD3_ILC3-Only_Layer1": fixed_limits = [(-0.79, 6), (-5.9, 50)]
                if tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": fixed_limits = [(-0.79, 6), (-5.9, 50)]
                if tissue_selection == "ST&TdT_Layer1": fixed_limits = [(-0.49, 4), (-6.9, 60)]
                if tissue_selection == "ST&TdT_PooledTH_Layer1": fixed_limits = [(-0.49, 4), (-6.9, 60)]
                if tissue_selection == "ST&TdT_DetailedTH_Layer1": fixed_limits = [(-2.9, 20)]
                if tissue_selection == "FL+AA+SG_Layer1": fixed_limits = [(-1.09, 6), (-5.9, 50)]
                if tissue_selection == "FL+AA+SG_NoT_Layer1": fixed_limits = [(-6.9, 40)]
                if tissue_selection == "SG+FL_Layer1": fixed_limits = [(-1.09, 6), (-5.9, 50)]
                if tissue_selection == "SG+FL_NoT_Layer1": fixed_limits = [(-3.9, 20)]
                if tissue_selection == "SG_Layer1": fixed_limits = [(-0.9, 6), (-5.9, 60)]
                if tissue_selection == "SG_NoT_Layer1": fixed_limits = [(-1.9, 20)]
                if tissue_selection == "FL_Layer1": fixed_limits = [(-1.09, 6), (-5.9, 50)]
                if tissue_selection == "FL_NoT_Layer1": fixed_limits = [(-3.9, 20)]
                if tissue_selection == "SR_Layer1": fixed_limits = [(-0.49, 3), (-3.9, 30)]
            elif data_type == "density":
                if tissue_selection == "AA+DA+DC_Layer1": fixed_limits = [(-79, 700), (-699, 6000)]
                if tissue_selection == "SG+SR_Layer1": fixed_limits = [(-129, 700), (-899, 8000)]
                if tissue_selection == "DC&CD3_ILC3-Only_Layer1": fixed_limits = [(-79, 700), (-699, 6000)]
                if tissue_selection == "DC&CD3_ILC1+ILC3_Layer1": fixed_limits = [(-79, 700), (-699, 6000)]
                if tissue_selection == "ST&TdT_Layer1": fixed_limits = [(-39, 300), (-899, 8000)]
                if tissue_selection == "ST&TdT_PooledTH_Layer1": fixed_limits = [(-39, 300), (-899, 8000)]
                if tissue_selection == "ST&TdT_DetailedTH_Layer1": fixed_limits = [(-399, 2000)]
                if tissue_selection == "FL+AA+SG_Layer1": fixed_limits = [(-139, 800), (-899, 8000)]
                if tissue_selection == "FL+AA+SG_NoT_Layer1": fixed_limits = [(-899, 4000)]
                if tissue_selection == "SG+FL_Layer1": fixed_limits = [(-129, 700), (-899, 8000)]
                if tissue_selection == "SG+FL_NoT_Layer1": fixed_limits = [(-499, 2000)]
                if tissue_selection == "SG_Layer1": fixed_limits = [(-139, 800), (-899, 8000)]
                if tissue_selection == "SG_NoT_Layer1": fixed_limits = [(-199, 2000)]
                if tissue_selection == "FL_Layer1": fixed_limits = [(-139, 800), (-699, 6000)]
                if tissue_selection == "FL_NoT_Layer1": fixed_limits = [(-499, 2000)]
                if tissue_selection == "SR_Layer1": fixed_limits = [(-49, 300), (-399, 3000)]

            draw_figures(datasets_dict___Figures, [tissue_selection, data_type], row, col, fixed_limits)

batch_draw_allData()
batch_draw_figures()

#---------------------
# DRAW FIGURES PAPER !
#---------------------
draw_paper_1(datasets_dict___Paper_1, ["SR", "density"], 1, 2, [(-49, 350)])
draw_paper_1(datasets_dict___Paper_1, ["ST&TdT_PooledTH", "density"], 1, 2, [(-199, 1000)])
draw_paper_1(datasets_dict___Paper_1, ["ST&TdT_DetailedTH", "density"], 1, 3, [(-499, 2500)])
draw_paper_1(datasets_dict___Paper_1, ["DC&CD3_ILC1+ILC3", "density"], 1, 2, [(-79, 700), (-79, 500)])
draw_paper_1(datasets_dict___Paper_1, ["DA+DC", "density"], 2, 2, [(-59, 500), (-299, 2000)])
draw_paper_1(datasets_dict___Paper_1, ["SG+AA", "density"], 2, 2, [(-139, 1000), (-799, 5000)])
draw_paper_1(datasets_dict___Paper_1, ["FL", "density"], 2, 1, [(-139, 800), (-599, 3000)])
draw_paper_1(datasets_dict___Paper_1, ["B+T+IL7R", "density"], 1, 6, [(-1299, 12000)])
draw_paper_1(datasets_dict___Paper_1, ["Relative_IL7R", "density"], 3, 2, [(-12.9, 80)])

#------------------------------
# DRAW FIGURES FLOW CYTOMETRY !
#------------------------------
draw_figures_FlowCytometry("_flow_cytometry_dataz", "ILCs+NK", 1, 2, [(-10, 100), (-3, 30)]) # ILCs + NK/ILC1ie...
draw_figures_FlowCytometry("_flow_cytometry_dataz", "ILCs", 1, 1, [(-10, 100)]) # ILCs only...
draw_figures_FlowCytometry("_flow_cytometry_dataz", "NK", 1, 1, [(-3, 30)]) # NK only...

#-----------------
# FIGURES OTHERS !
#-----------------
draw_figures_B_T_IL7R(datasets_dict___Figures_B_T_IL7R, ["WholeTissue", "percentage"], 1, 6, [(-6.9, 60)])
draw_figures_B_T_IL7R(datasets_dict___Figures_B_T_IL7R, ["WholeTissue", "density"], 1, 6, [(-899, 8000)])
draw_figures_B_T_IL7R(datasets_dict___Figures_B_T_IL7R, ["Split", "percentage"], 1, 6, [(-9.9, 80)])
draw_figures_B_T_IL7R(datasets_dict___Figures_B_T_IL7R, ["Split", "density"], 1, 6, [(-1299, 12000)])

#-----------------
# FIGURES OTHERS !
#-----------------
draw_figures_TCR_Relative(datasets_dict___Figures_TCR_Relative, ["WholeTissue"], 2, 3, [(-12.9, 80)])
draw_figures_TCR_Relative(datasets_dict___Figures_TCR_Relative, ["Split"], 2, 3, [(-12.9, 100)])

#-----------------
# FIGURES OTHERS !
#-----------------
draw_figures_IL7R_Relative(datasets_dict___Figures_IL7R_Relative, ["WholeTissue"], 2, 3, [(-6.9, 40)])
draw_figures_IL7R_Relative(datasets_dict___Figures_IL7R_Relative, ["Split"], 2, 3, [(-12.9, 80)])

#-----------------
# FIGURES OTHERS !
#-----------------
draw_figures_ST_DetailedTH_Split(datasets_dict___Figures_ST_DetailedTH_Split, ["ST&TdT_DetailedTH_Split_Cortex+Medulla", "percentage"], 1, 3, [(-0.07, 0.5), (-2.9, 20), (-2.9, 20)])
draw_figures_ST_DetailedTH_Split(datasets_dict___Figures_ST_DetailedTH_Split, ["ST&TdT_DetailedTH_Split_Cortex+Medulla", "density"], 1, 3, [(-9.9, 50), (-499, 2500), (-499, 2500)])

#-----------------
# FIGURES OTHERS !
#-----------------
draw_figures_B_T_IL7R_ST_WithPanelCD20(datasets_dict___Figures_B_T_IL7R_ST_WithPanelCD20, ["WholeTissue", "density"], 1, 6, [(-899, 8000)])
draw_figures_B_T_IL7R_ST_WithPanelCD20(datasets_dict___Figures_B_T_IL7R_ST_WithPanelCD20, ["Split", "density"], 1, 6, [(-1299, 12000)])
