# P7_API_V2

Within DS/P7 project, repo dedicated to API testing and release (FastAPI)

One single branch (main)

# Description

In this Repo, you will find:

### api_V2.py ###

Python script to launch a FastAPI API including several path in order to get:

- list of credit ids
- single client details
- 'pd.describe' of the dataset
- prediction (crédit accepté ou refusé) + predict_proba (acceptance threshold = 0.30)
- shap value (feature local importance)

### src ###

Several pkl files used within API:
best_lgbm2.pkl => best model trained
data_smpl.pkl => an extract of 12 000 credit
+
samplerV2.pkl, transformerV2. pkl and est_LGBM which are a decompisition of best_lgbm2.
=> Required to get Shap values

### test_api.py ###

Unit testing cases: 1 per fonction defined in API_V2.py

## Github/workflow/test.yml

 -> allow automatic unit testings on push + automatic deployment in render



# Installation

Through conda / .venv environment launch pip install -r requirements.txt

## To launch API:

- Locally:
In terminal, run uvicorn api_V2:api --reload, API is displayed on http://127.0.0.1:8000

- Production: https://xavier-convert.onrender.com/

# TESTINGS:

For local testings, at root level, run pytest in terminal

While pushing to Github a new version, testings are performed automatically. If passed then auto deployment on https://dashboard.render.com/web/srv-cmn90v0cmk4c73e6pdeg


