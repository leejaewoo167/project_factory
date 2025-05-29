from models.LIGHTGBM import LightGBM
from models.RANDOMFOREST import Randomforest
from models.XGBOOST import Xgboost
from models.VOTING import Voting

def SelectModel(model_name: str, 
                params: dict):
    
    # Every models are implement using the scikit-learn API
    model = {'lightgbm': LightGBM, 
             'randomforest': Randomforest, 
             'xgboost': Xgboost,
             'voting': Voting}

    return model[model_name](params)