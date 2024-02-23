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
    
def test_client_details(cid=101420):
    data = get_client_detail(cid)
    print(f'id test = {cid}')
    print(data['AMT_CREDIT'])
    assert data['AMT_CREDIT']==972000

def test_show_data():
    df=show_data()
    print(df.columns.to_list())
    assert len(df.columns.to_list())==25
    
def test_get_prediction(cid=101420):
    pred=get_predictions(cid)
    #print (type(pred))
    print (pred["prediction"])
    print (pred['proba_rembour'])
    if pred["prediction"]=='Crédit accepté':
        assert pred['proba_rembour']<=0.7
    else:
        assert pred['proba_rembour']>0.7
    
