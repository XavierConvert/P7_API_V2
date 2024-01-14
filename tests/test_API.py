from ..src.api import get_ids

#from api import show_data
#from api import get_predictions
import pandas as pd

def test_get_ids():
    ids = get_ids()
    assert len(ids.values) >= 1
    

#def test_show_data()
#    df=show_data()
#    assert df.keys= ['SK_ID_CURR','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS','EXT_SOURCE_1','EXT_SOURCE_2',
#                     'EXT_SOURCE_3','client_age','client_prof_exp','Cash_loans','GENDER_FEMALE','active_client','relationship']

#def test_get_prediction()
#    pred=get_predictions(id)
#    assert pred