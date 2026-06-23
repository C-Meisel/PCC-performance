# PCC-performance

Cell-test data and Python scripts for the Gaussian Process (GP) and Random Forest Regressor (RFR) models of proton-conducting ceramic fuel cell (PCC) performance.

I developed these models during my PhD at the Colorado School of Mines to identify which fabrication and testing parameters most strongly influence cell performance, using a hand-curated dataset of cells fabricated and tested in-house over roughly five years.

## What this repository supports

This code and data accompany Data-driven insights into protonic-ceramic fuel cell and electrolysis performance. Published in Journal of Materials Chemistry A in 2025. https://doi.org/10.1039/D4TA08326A

Meisel, C. et al. Data-driven insights into protonic-ceramic fuel cell and electrolysis performance. Journal of Materials Chemistry A 13, 10863–10880 (2025).

Machine learning models show that lowering the electrolyte thickness-to-grain-size ratio, using smaller NiO particles, and removing organics before sintering boosts performance. The positrode is key for fuel cells, while the electrolyte drives electrolysis.

## Contents

- `FuelCell_RandomForest_pub.py`: Random Forest Regressor model of fuel-cell peak power density. Includes k-fold cross-validation, permutation-importance analysis computed across folds, partial dependence plots, and a goodness-of-fit panel (measured versus predicted, residuals, residual distribution, and learning curve).
- `RandomForest_Electrolysis_pub.py`: the same Random Forest workflow applied to electrolysis performance.
- `FuelCell_GaussianProcess_pub.py` and `GaussianProcess_Electrolysis_pub.py`: Gaussian Process regression models for the fuel-cell and electrolysis cases.
- `Random_Forest_HypTune_FC.py` and `Random_Forest_HypTune_EC_pub.py`: hyperparameter-tuning scripts for the fuel-cell and electrolysis Random Forest models.
- `Performance_data_pub.xlsx`: the full cell-test dataset used in the paper, plus additional data. Key sheets are described below.


## Performance_data_pub.xlsx contains all the data used in the paper and more <br> 
### Data sheet overview: <br> 
All_data - All data from all sucessful cell tests. <br> 
Mines_4411 - All data for all sucessful tests of BCZYYb4411 cells fabricated and tested at mines. <br> 
LowColl - Mines 4411 cell data with the initial 29 columns dropped. <br> 
LC_LC_FC_n9 - LowColl data for the fuel cell models (Cell 9 has been taken out of the data). <br> 
LC_FC_n9_sel - Model as a result of column selection. <br> 
**LC_FC_n9_sel_s - Model as a result of column selection without columns that had no effect on the model (This is the final fuel cell data model).**<br> 
LC_EC - Mines 4411 cell data with the initial 29 columns dropped. With all rows that contain electrolysis performance data.<br> 
LC_EC_n64 - LowColl data for the electrolysis models (Cell 64 has been taken out of the data).<br> 
LC_EC_n64_sel - Model as a result of column selection.<br> 
**LC_EC_n64_sel_s - Model as a result of column selection without columns that had no effect on the model (This is the final electrolysis data model).**<br> 
Resistance - Resistance data for all 4411 cells, used for Figures 7 and 8.<br> 

## Requirements

Python 3, with pandas, numpy, scikit-learn, matplotlib, and seaborn.

## How to run

Each script is a self-contained analysis configured at the top of the file. Before running, set `data_path` to the location of `Performance_data_pub.xlsx` and set the figure-save paths if you want to export figures. The boolean flags near the top (for example `plot_feature_importance`, `plot_pdp`, and `all_gof`) toggle which analyses and figures are produced.

## Note

This is research code written to support a publication rather than production software. It is shared for transparency and reproducibility.

## License

BSD-3-Clause.


