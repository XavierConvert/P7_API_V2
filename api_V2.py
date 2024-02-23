#####################################
########### Script pour API #########
#####################################

### 1. Import des librairies requises

### 2. Création de l'objet API

### 3. Chargement des fichiers .pkl requis + préparation du fichier shap_values:
## - Modèle ML
## - Data
## - Décomposition du modèle pour le calcul des shap values (imputer / transformer / estimator)

### 4. Défintion des différentes routes / fonctions:
## 1. message d'accueil
## 2. list des ids (12299 dans la dernière version chargée)
## 3. Informations clients => par appel du client id
## 4. Synthèse du dataset => describe()
## 5. Prédiction du modèle par appel d'un client (seuil prodict proba = 0.7)
## 6. Appel shap values => par appel du client id


# 1. Import libraries 
import uvicorn
from fastapi import FastAPI, Response #, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import numpy as np
import json
from pydantic import BaseModel
 #from sklearn.model_selection import train_test_split
import shap


    
# 2. Create the api object
api = FastAPI()

# 3. Imports of required pkl files:

pkl_1= open("src/best_lgbm2.pkl","rb")
classifier=pd.read_pickle(pkl_1)

pkl_2= open("src/data2pkl_V2.pkl","rb")
data=pd.read_pickle(pkl_2)

pkl_3 = open("src/samplerV2.pkl","rb")
sampler=pd.read_pickle(pkl_3)

pkl_4 = open("src/transformer2_V2.pkl","rb")
transformer2=pd.read_pickle(pkl_4)

pkl_5 = open("src/est_LGBM.pkl","rb")
estimator2=pd.read_pickle(pkl_5)


# Préparation du dataframe svv (shap_value.values) 'global' qui sera appelé par client id dans la fonction getshap_value(cid:int):

filtered_data=data.drop(['AMT_GOODS_PRICE'], axis =1)
#print(f"nb de NA sur EXT_SRC_1 avant imputation= {filtered_data['EXT_SOURCE_1'].isna().sum()}")
transf_data = pd.DataFrame(sampler.transform(filtered_data),columns = filtered_data.columns)
#print(f"nb de NA sur EXT_SRC_1 après imputation= {transf_data['EXT_SOURCE_1'].isna().sum()}")
#print(f'transf_data shape = {transf_data.shape}') 
data_for_shap=transf_data.drop(['SK_ID_CURR'],axis=1)
data_for_shap_tr=pd.DataFrame(transformer2.transform(data_for_shap),columns=data_for_shap.columns)
explainer=shap.Explainer(estimator2,data_for_shap_tr)
shap_values=explainer(data_for_shap_tr,check_additivity=False)
#bv=shap_values.base_values
svv=pd.DataFrame(shap_values.values, columns = data_for_shap.columns).round(2)
svv['SK_ID_CURR']=transf_data['SK_ID_CURR']  
#print(f'svv shape = {svv.shape}')


## 4. Routes

# 4.1 Index route, opens automatically on http://127.0.0.1:8000
@api.get('/')
def index():
    return {'message': 'Hello. API is running'}   

#4.2 Id list (taken from to X_test in ML notebook)
@api.get("/ids/") 
def get_ids() -> dict:
    ids=data['SK_ID_CURR'].to_dict()
    return ids
    
#4.3 Client detail calling an Id
@api.get("/client_details/{cid}")
def get_client_detail(cid:int):
    # NaN in EXT_SOURCE 1,2 and 3 are replaced by 'NA'
    no_nan_data=data.fillna('NA')
    filtered_data=no_nan_data.loc[no_nan_data['SK_ID_CURR']==cid,:].T   
    return filtered_data.iloc[:,0] 
    
    
# 4.4 Display describe() to get an overview of the dataset (to compare a single Id to the whole dataset)
@api.get('/data/')
def show_data():
    return data.describe().round(2)


# 4.5 Has the credit been accepted or refused (depending of the optimal threshold)
@api.get("/prediction/{cid}")
def get_predictions(cid:int):
    filtered_data=data.loc[data['SK_ID_CURR']==cid,:]
    #print(f'filtered_data shape avant drop = {filtered_data.shape}') #=> 25 ok
    
    # Comme pour l'entrainement du modèle dans le notebook ML, je n'utilise ni SK_ID_CURR ni AMNT GOOD PRICE:
    filtered_data=filtered_data.drop(['SK_ID_CURR','AMT_GOODS_PRICE'], axis =1)
    #print(f'filtered_data après drop shape = {filtered_data.shape}') #=> 23
    prediction = classifier.predict_proba(filtered_data).tolist()
    
    # Seuil determiné dans notebook ML = 0.7
    
    if(prediction[0][1]>0.7):
        avis="Crédit refusé"
    else:
        avis="Crédit accepté"
    return {'prediction': avis,
            'proba_rembour':round(prediction[0][0],2)}
    
    # On pourrait ajouter le gain ou la perte estimée
   
# 4.6 Get the shap values (features that have the more influence on decision) that will be used in dashboard to display graph      
@api.get("/shap_val/{cid}")
def shap_value(cid:int):
    ### Cette version fonctionne mais pose 1 pb:
    ### 1. fait appel à des fichiers .pkl qui constituent le modèle global:
    ###        - sampler + imputer:  pour imputation des NaN uniquement (le sampler ne s'applique pas)
    ###        - transformer (standardScaler)
    ###        - estimateur ()
    ###  => requiert à chaque deploiement d'un nouveau modèle de déployer également les elements constitutifs de ce modèle => risque d'oubli
    ### Notes: - pls essais ont été effectués afin de décomposer l'objet classifier en classifier[-3], classifier[-2] et classifier[-1] 
    ###             => KO pour la majorité des SK_ID_CURR
    ###             => IndexError: single positional indexer is out-of-bounds
    ###        - même si je ne conserve que la partie estimateur ca plante 'IndexError: single positional indexer is out-of-bounds'   

    # 4 lignes suivantes sont OK:
    svv2=svv.loc[svv['SK_ID_CURR']==cid]
    svv2=svv2.drop(['SK_ID_CURR'],axis=1)
    svv2=svv2.T
    return svv2.iloc[:,0]

    #return JSONResponse(content=explainer)
     
# pour tester en local, depuis le terminal:
# uvicorn api_V2:api --reload  


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    #uvicorn.run(api, host='127.0.0.1', port=8000)
    # version KO du 18/02/24: uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run(api, host='127.0.0.1', port=8000)
