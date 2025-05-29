from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from utils import logger, set_parameters

def Voting(params):
    
    cand_model = ['rf','xgb', 'lgbm']
    cand_params = [['n_estimators', 'max_depth'],
                   ['n_estimators', 'max_depth', 'learning_rate'],
                   ['n_estimators', 'max_depth', 'learning_rate']]
    
    param_set = {
        f'{mod}__{p}': set_parameters(params[mod][p])\
            for mod, param in zip(cand_model,cand_params)\
                for p in param
    }
    
    bagged_lgbm = BaggingClassifier(
        estimator=LGBMClassifier(),
        bootstrap=True,
        n_jobs=5) 
    
    classifier = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('lgbm', bagged_lgbm)],
        voting='soft'
        )
    
    logger.info(f'load Voting model')
    return  {"parameters": param_set, 
            "estimator": classifier}
