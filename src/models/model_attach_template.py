# step 1: load the required libraries 
from sklearn.ensemble import RandomForestClassifier
from utils import logger

def modelName(params):
    
    # candidate parameters
    param1 = params['depth']
    param_set = {
        "param1": [p for p in range(param1[0], param1[1]+1)]
    }
    
    # define model
    classifier = RandomForestClassifier()
    
    logger.info(f'load random forest model')
    return {"parameters": param_set, 
            "estimator": classifier}
    
    
# after make this file, plz attach to select_model.py