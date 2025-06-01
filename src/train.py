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
        """_summary_

        Args:
            model (dict): loaded model
            dataset (dict): loaded dataset
            params (dict): parameters
        """
        
        self.train_x = dataset['train_X']
        self.train_y = dataset['train_y']
        
        self.test_x = dataset['test_X']
        self.test_y = dataset['test_y']
        
        self.preprocessor = dataset['processor'] 
        
        self.param_set = model['parameters']
        self.model = model['estimator']
    
    def model_train(self):
        
        # parameter tunning을 위한 searching strategy 정의
        pr_search = GridSearchCV(estimator=self.model,
                                 param_grid=self.param_set,
                                 scoring='roc_auc', # metric
                                 cv = 5, # # of cross validation 
                                 n_jobs=5) # # of CPU core
        
        # 학습 pipeline 정의
        """
        - class의 비율이 불균형하므로 sklearn 기반의 imblearn의 pipelin을 사용하여 smote을 pipeline에 참여
        - smote: KNN 기반으로 이웃 샘플을 선택하여 소수의 클래스 샘플과 이웃 샘플을 합성하는 방법입니다.
        - process
            step 1: 모델 학습을 위해 각 데이터셋의 type에 맞게 전처리
            step 2: smote를 이용하여 sampling
            step 3: 학습 및 hyper-parameter search
        
        """
        train_pipeline = ImbPipeline(steps = [('processor', self.preprocessor),
                                              ('smote', SMOTE()),
                                              ('classifier', pr_search)])
        
        # model fitting
        train_pipeline.fit(self.train_x, 
                           self.train_y)
        
        return train_pipeline