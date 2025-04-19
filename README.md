# Data Visualization by Imag'IN Labs
**Tools repository for data visualization using python packages (Matplotlib, Panda, Seaborn, etc...).**  
Datasets, visualization and analysis tools used in the paper *"Spatial mapping of Innate Lymphoid Cells in human lymphoid tissues and lymphoma at single-cell resolution"* authored by *Nathalie Van Acker <sup>\*,1</sup>, François-Xavier Frenois <sup>\*,1</sup>, Pauline Gravelle <sup>1,2</sup>, Charlotte Syrykh <sup>1</sup>, Camille Laurent <sup>1,2</sup> and Pierre Brousset <sup>#,1,2</sup>*.

<sup>1</sup>	Department of Pathology, CHU, Imag’IN Platform, IUCT-Oncopole, Toulouse, France.  
<sup>2</sup>	Cancer Research Center of Toulouse, Team 9 NoLymIT and Labex TOUCAN, Toulouse, France.  
<sup>#</sup>	Correspondence: brousset.p@chu-toulouse.fr | +33 5 31 156 141.  
<sup>*</sup>	Authors equally contributed (NVA & FXF).

**Imag'IN Platform :** https://www.imagin-labs.net

## Clustering and clustermaps
**To run the provided Python script, follow these steps:**

1. **Install Required Dependencies**   
    
    Ensure you have the necessary Python libraries installed. You can install them using ```pip```: ```pip3 install pandas seaborn matplotlib numpy scikit-learn```

2. **Set Up the Directory Structure**

    - Place the script in a directory.    
    - Ensure the following subdirectories and files exist:
        - ```DATASETS/```: Contains the input CSV files referenced in the script.
        - ```OUTPUT/```: The directory where output files (e.g., clustermaps) will be saved.
        - ```MODULES/utilities.py```: A custom module used in the script.

3. **Configure the Script**

    - Set the global vars:
        - ```describe_data```: set to ```True``` to describe continuous and categorical data
        - ```pca_analysis_show_plots```: set to ```True``` to show PCA analysis plots interactively
        - ```clustering_analysis_show_plot```: set to ```True``` to show clustering analysis plots interactively
        - ```clustering_analysis_print_output```: set to ```True``` to print clustering analysis output interactively
    - Modify the ```draw()``` function calls at the bottom of the script to specify the dataset and parameters you want to process.    
    - Example:

        ```draw(datasets_dict, [["AA", "DA", "DC", "SG", "SR", "ST"], ["Sex", "Age Range"], "_ALL_Merged_Layer1", "density", None], True, True)```

        > - Dataset Keys: Adjust the dataset keys to match your input data.
        > - Data Type: Choose between ```"percentage"``` or ```"density"```.
        > - Save Option: Set ```save=True``` to save the output files.

4. **Run the Script**

    Open a terminal, navigate to the directory containing the script, and execute: ```python3 innate-lymphoid-cells_clustermap.py```

5. **View the Output**

    - The script will generate clustermaps and save them in the ```OUTPUT/``` directory.    
    - If ```save=False```, the plots will be displayed interactively using ```matplotlib```.

6. **Debugging**

    If you encounter errors, ensure:

    - The input CSV files in ```DATASETS/``` match the expected structure.
    - The ```utilities.py``` module is present and functional.
    - All required libraries are installed.

## Violin Plots
**To run the provided Python script, follow these steps:**

1. **Install Required Dependencies**   
    
    Ensure you have the necessary Python libraries installed. You can install them using ```pip```: ```pip3 install pandas seaborn matplotlib numpy importlib```

2. **Run the Script**

    Open a terminal, navigate to the directory containing the script, and execute: ```python3 innate-lymphoid-cells_violinplot.py```

3. **View the Output**

    - The script will generate violinplots and save them in the ```OUTPUT/``` directory.
    - The script will batch draw single plots and plots organized in "figures", plus all the plots used in the figures of the paper referenced above.

4. **Debugging**

    If you encounter errors, ensure:

    - The input CSV files in ```DATASETS/``` match the expected structure.
    - The ```utilities.py``` module is present and functional.
    - All required libraries are installed.

# Author / License
Author: François-Xavier Frenois, PhD (ouatataz@free.fr).
Licensed under the MIT (http://www.opensource.org/licenses/mit-license.php).
