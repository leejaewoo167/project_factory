# step 1: load the required libraries 
from sklearn.ensemble import RandomForestClassifier
from utils import logger, set_parameters

def modelName(params):
    
    # candidate parameters
    param1 = params['depth']
    param_set = {
        "param1": set_parameters(param=param1)
    }
    
    # define model
    classifier = RandomForestClassifier()
    
    logger.info(f'load random forest model')
    return {"parameters": param_set, 
            "estimator": classifier}
    
    
# after make this file, plz attach to select_model.py