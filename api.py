
# 1. Import libraries 
import uvicorn
from fastapi import FastAPI #, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
#from credit import Credit #, Client
import pickle
import pandas as pd
import numpy as np
import json
from pydantic import BaseModel
#from sklearn.model_selection import train_test_split
import shap
#shap.initjs()

    
# 2. Create the app object
app = FastAPI()

pickle_in = open("src/classifier.pkl","rb")
#classifier=pd.read_pickle(pickle_in)
classifier=pickle.load(pickle_in)

pickle_in_2= open("src/data2pkl.pkl","rb")
data=pd.read_pickle(pickle_in_2)
#data=pickle.load(pickle_in_2)

pickle_in_3 = open("src/sampler.pkl","rb")
sampler=pd.read_pickle(pickle_in_3)
#sampler=pickle.load(pickle_in_3)

pickle_in_4 = open("src/transformer2.pkl","rb")
transformer2=pd.read_pickle(pickle_in_4)
#transformer2=pickle.load(pickle_in_4)

pickle_in_5 = open("src/logreg2.pkl","rb")
estimator2=pd.read_pickle(pickle_in_5)
#estimator2=pickle.load(pickle_in_5)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello. API is running'}   

@app.get("/ids/") 
def get_ids() -> dict:
    ids=data['SK_ID_CURR'].to_dict()
    return ids


@app.get("/client_details/{cid}")
def get_client_detail(cid:int):
    no_nan_data=data.fillna('NA')
    filtered_data=no_nan_data.loc[no_nan_data['SK_ID_CURR']==cid,:].T#.to_dict()    
    return filtered_data.iloc[:,0] 
    
@app.get('/data/')
def show_data():
    return data.describe().round(2)


@app.get("/prediction/{cid}")
def get_predictions(cid:int):
    filtered_data=data.loc[data['SK_ID_CURR']==cid,:]
    filtered_data=filtered_data.drop(['SK_ID_CURR','AMT_GOODS_PRICE'], axis =1)
    prediction = classifier.predict_proba(filtered_data).tolist()
    
    if(prediction[0][1]>0.65):
        avis="Crédit refusé"
    else:
        avis="Crédit accepté"
    return {'prediction': avis,
            'proba_rembour':round(prediction[0][0],2)}
    #return prediction[0][0]

# Prévoir une route qui fait appel à la fonction gain

# Prévoir une route qui donné les prédictions pour un nouveau client (don cpas de SK_ID mais formulaire avec les différentes données attendues)  
        
@app.get("/shap_val/{cid}")
def shap_value(cid:int):
    filtered_data=data.drop(['AMT_GOODS_PRICE'], axis =1)
    
    transf_data = pd.DataFrame(sampler.transform(filtered_data),columns = filtered_data.columns) 
    data_for_shap=transf_data.drop(['SK_ID_CURR'],axis=1)
    data_for_shap_tr=pd.DataFrame(transformer2.transform(data_for_shap),columns=data_for_shap.columns)
    explainer=shap.LinearExplainer(estimator2,data_for_shap_tr)
    shap_values=explainer(data_for_shap_tr)
    
    svv=pd.DataFrame(shap_values.values, columns = data_for_shap.columns).round(2)
    svv['SK_ID_CURR']=transf_data['SK_ID_CURR']  
    svv=svv.loc[svv['SK_ID_CURR']==cid]
    svv=svv.drop(['SK_ID_CURR'],axis=1)
    svv=svv.T
    return svv.iloc[:,0]
    
    
# uvicorn api:app --reload   

  



# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted redemption of the credit with the confidence

#@app.post('/predict')
#def predict_credit(data:Credit):
#    data = data.model_dump()
#    #SK_ID_CURR=data['SK_ID_CURR']
#    FLAG_OWN_CAR=data['FLAG_OWN_CAR']
#    FLAG_OWN_REALTY=data['FLAG_OWN_REALTY']
#    AMT_INCOME_TOTAL=data['AMT_INCOME_TOTAL']
#    AMT_CREDIT=data['AMT_CREDIT']
#    AMT_ANNUITY=data['AMT_ANNUITY']
#    #AMT_GOODS_PRICE=data['AMT_GOODS_PRICE']
#    CNT_FAM_MEMBERS=data['CNT_FAM_MEMBERS']
#    EXT_SOURCE_1=data['EXT_SOURCE_1']
#    EXT_SOURCE_2=data['EXT_SOURCE_2']
#    EXT_SOURCE_3=data['EXT_SOURCE_3']
#    client_age=data['client_age']
#    client_prof_exp=data['client_prof_exp']
#    Cash_loans=data['Cash_loans']
#    GENDER_FEMALE=data['GENDER_FEMALE']
#    active_client=data['active_client']
#    relationship=data['relationship']
        
#    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
#    prediction = classifier.predict([[FLAG_OWN_CAR,FLAG_OWN_REALTY,AMT_INCOME_TOTAL,AMT_CREDIT,AMT_CREDIT,
#                                        AMT_ANNUITY,CNT_FAM_MEMBERS,EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3,
#                                        client_age,client_prof_exp,Cash_loans,GENDER_FEMALE,active_client,relationship]])
#        # SK_ID_CURR,AMT_GOODS_PRICE
        
#        if(prediction[0]>0.65):
#            prediction="Crédit refusé"
#        else:
#            prediction="Crédit accepté"
#        return {
#            'prediction': prediction
#        }
    


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #uvicorn.run(app, host="0.0.0.0", port=80)#, reload=False)
    
