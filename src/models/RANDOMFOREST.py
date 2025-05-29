from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils import logger

def Randomforest(params):
    
    # candidate parameters
    depths = params['depth']
    param_set = {
        "max_depth": [d for d in range(depths[0], depths[1]+1)]
    }
    classifier = RandomForestClassifier()
    
    logger.info(f'load random forest model')
    return {"parameters": param_set, 
            "estimator": classifier}