# PCC-performance
Cell data and python scripts for Gaussian Process (GP) and Random Forest Regressor (RFR) models

Four of the python scripts are for implementing the GP and RFR models for fuel cell and electrolysis performance
Two of the scripts are for tuning the hyper parameters for the RFR models

Performance_data_pub.xlsx contains all the data used in the paper and more
Data sheet overview:
All_data - All data from all sucessful cell tests.
Mines_4411 - All data for all sucessful tests of BCZYYb4411 cells fabricated and tested at mines.
LowColl - Mines 4411 cell data with the initial 29 columns dropped.
LC_LC_FC_n9 - LowColl data for the fuel cell models (Cell 9 has been taken out of the data).
LC_FC_n9_sel - Model as a result of column selection.
**LC_FC_n9_sel_s - Model as a result of column selection without columns that had no effect on the model (This is the final fuel cell data model).**
LC_EC - Mines 4411 cell data with the initial 29 columns dropped. With all rows that contain electrolysis performance data
LC_EC_n64 - LowColl data for the electrolysis models (Cell 64 has been taken out of the data).
LC_EC_n64_sel - Model as a result of column selection.
**LC_EC_n64_sel_s - Model as a result of column selection without columns that had no effect on the model (This is the final electrolysis data model).**
Resistance - Resistance data for all 4411 cells, used for Figures 7 and 8.


