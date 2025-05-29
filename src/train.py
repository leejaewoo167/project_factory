from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self, 
                 model: dict,
                 dataset: dict,
                 params:dict):
        
        self.params = params 
        self.seed = params['seed']
        
        self.train_x = dataset['train_X']
        self.train_y = dataset['train_y']
        
        self.test_x = dataset['test_X']
        self.test_y = dataset['test_y']
        
        self.preprocessor = dataset['processor'] 
        
        self.param_set = model['parameters']
        self.model = model['estimator']
    
    def model_train(self):
        
        # define strategy of parameter search
        pr_search = GridSearchCV(estimator=self.model,
                                 param_grid=self.param_set,
                                 scoring='roc_auc', # metric
                                 cv = 5, # # of cross validation 
                                 n_jobs=5) # # of CPU core
        
        # define training pipeline
        train_pipeline = ImbPipeline(steps = [('processor', self.preprocessor),
                                              ('smote', SMOTE(random_state = self.seed)),
                                              ('classifier', pr_search)])
        
        # fitting with train feature and train targets
        train_pipeline.fit(self.train_x, 
                           self.train_y)
        
        return train_pipeline