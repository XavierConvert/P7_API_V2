from api import get_ids
#from api import show_data
#from api import get_predictions
import pandas as pd

def test_get_ids():
    ids = get_ids()
    assert len(ids.values) >= 1
    

#def test_show_data()
#    df=show_data()
#    assert df.keys= #liste des colonnes à récupérer

#def test_get_prediction()
#    pred=get_predictions(id)
#    assert pred