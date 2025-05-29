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
        
        y_pred = model.predict(self.test_x)
        cal_metric = metric(y_truth=self.test_y, 
                            y_hat=y_pred)
        return cal_metric
        
        