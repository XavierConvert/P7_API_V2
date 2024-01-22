# P7_API_V2

Within DS/P7 project, repo dedicated to API (FastAPI)

#### Description ####

In this Repo, you will find:

## api.py ##

Python script to launch a FastAPI API including several path in order to get:

- id list
- client details
- 'pd.describe' of the dataset
- prediction (crédit accepté ou refusé) + predict_proba (acceptance threshold = 0.35)
- shap value

## src ##

Several pkl files used within API 

## test_api.py ""

Several test cases to be run with pytest


#### Installation ####

Through conda / .venv environment launch pip install -f requirement.txt

For API:

In terminal, run uvicorn api:app --reload, API is displayed on http://127.0.0.1:8000

For tests:

At root level, run pytest in terminal


