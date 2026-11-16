# Data assumptions

The only customer table is `data/fresh_and_good_customers.csv` in this folder. The file has 2,400 rows and no customer id. Ids `FG0000` through `FG2399` follow the file's row order.

This pipeline trains on the June snapshot columns only. July labels stay out of the model matrix. `churned` is the target.

`data/live.parquet` starts as a copy of the training reference. A facilitator may replace the live file during the drift drill. Until that happens, the two files match.

The metric gate is a held-out ROC-AUC of 0.70. The alias moves only when the score is at least that high.

The plain runner is the path this day verifies on every laptop. Airflow in Docker is optional and was not assumed to be installed on every machine. One working Docker install per team is enough for the UI. The stock compose file does not mount this repo's `lib` folder and does not ship the scientific packages. See `airflow/README.md`.

This folder does not need a TA or instructor signature. That signing role is still unsettled.
