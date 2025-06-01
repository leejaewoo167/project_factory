from sklearn.metrics import classification_report, confusion_matrix
from utils import metric

class Tester:
    def __init__(self,
                 dataset: dict
                 ):
        
        self.test_x = dataset['test_X']
        self.test_y = dataset['test_y']
    
    def model_test(self,
                   model: object):
        """_summary_

        Args:
            model (object): 학습한 모델

        Returns:
            results: 성능 평가 결과(AUC, F1 score(binary, macro, micro)) 
        """
        
        y_pred = model.predict(self.test_x)
        cal_metric = metric(y_truth=self.test_y, 
                            y_hat=y_pred)
        return cal_metric
        
        