from ..api_V2 import get_ids #..src.api
from ..api_V2 import show_data #..src.api
from ..api_V2 import get_client_detail
from ..api_V2 import get_predictions
#import pandas as pd
#import logging

#logging.basicConfig(filename='test.log', level=logging.DEBUG) #, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S' )

def test_get_ids():
    ids = get_ids()
    print(f'le dataset contient {len(ids.values())} ids')
    #logging.debug(msg='1er essai log')
    # test qui doit passer
    assert len(ids.values()) >= 1
    
    # test qui ne doit être KO (test CI/GitHub Actions):
    #assert len(ids.values()) ==1
    
def test_client_details(cid=101400):
    data = get_client_detail(cid)
    print(f'id test = {cid}')
    print(data['AMT_CREDIT'])
    assert data['AMT_CREDIT']==370629

def test_show_data():
    df=show_data()
    print(df.columns.to_list())
    assert len(df.columns.to_list())==25
    #['SK_ID_CURR','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS','EXT_SOURCE_1',
    #                              'EXT_SOURCE_2','EXT_SOURCE_3','Prev_contract_nb','Prev_AMT_CREDIT','Refused_rate','default_payment','INCOME_CREDIT_PERC','ANNUITY_INCOME_PERC',
    #                              'PAYMENT_RATE','client_age','INCOME_PER_PERSON','client_prof_exp','Cash_loans','GENDER_FEMALE','active_client','relationship']

def test_get_prediction(cid=101400):
    pred=get_predictions(cid)
    #print (type(pred))
    print (pred["prediction"])
    print (pred['proba_rembour'])
    assert pred["prediction"]=='Crédit accepté'
    assert pred['proba_rembour']==0.61
    
