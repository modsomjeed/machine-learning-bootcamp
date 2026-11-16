# Data assumptions

The customer table is `data/fresh_and_good_customers.csv` in this folder. It has 2,400 rows and no customer id. `make prepare` assigns `FG0000` through `FG2399` from file order, trains on the June snapshot columns, and writes the feature table, the drift tables, the MLflow alias, and the Feast online store in this folder.

`churned` is the label. July spend is not a feature. The fit is one 80/20 split with seed 42. The alias moves only when the held-out ROC-AUC is at least 0.70. A monitor column is flagged when PSI is at least 0.2.

Who signs checkpoint 1 is the TA or the instructor. TA availability is not settled.
