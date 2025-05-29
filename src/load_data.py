import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from utils import logger

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bounds = {}

    def fit(self, X, y=None):
        self.bounds = []
        for i in range(X.shape[1]):
            q1 = np.percentile(X[:, i], 25)
            q3 = np.percentile(X[:, i], 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds.append((lower, upper))
        return self

    def transform(self, X):
        X_clipped = X.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            X_clipped[:, i] = np.clip(X[:, i], lower, upper)
        return X_clipped

class LoadDataset:
    def __init__(self,
                 isTrain: bool,
                 params:dict):
        self.isTrain = isTrain
        if isTrain:
            self.data = pd.read_csv('../data/train.csv')    
        else:
            self.data = pd.read_csv('../data/test.csv')
            
        self.spilt_ratio = params['test_split_ratio']
    
    # it needs to refactoring about split to train and test 
    def preprocessing(self) -> dict:
        # train phase
        if self.isTrain:
            # 타임스탬프 생성 및 정렬
            self.data['timestamp'] = pd.to_datetime(self.data['date'] + " " + self.data['time'], errors='coerce')
            self.data = self.data.sort_values('timestamp')
            self.data = self.data.drop(index=19327, errors='ignore')  # 결측이 심한 한 행 제거 예시

            # 결측치 많은 열 제거
            missing_counts = self.data.isnull().sum()
            cols_to_drop = missing_counts[missing_counts >= 30000].index
            self.data = self.data.drop(columns=cols_to_drop)

            # 수치형/범주형 컬럼 재정의
            exclude_cols = ['id', 'passorfail', 'timestamp', 'time', 'date', 'name', 'line', 'mold_name']
            numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.difference(exclude_cols).tolist()
            categorical_cols = self.data.select_dtypes(include=['object']).columns.difference(exclude_cols).tolist()

            # 수치형: 시간 보간
            for col in numeric_cols:
                temp = self.data[[col, 'timestamp']].set_index('timestamp')
                temp[col] = temp[col].interpolate(method='time', limit_direction='both')
                temp[col] = temp[col].fillna(temp[col].median())
                self.data[col] = temp[col].values

            # 데이터 분할
            X = self.data[numeric_cols + categorical_cols]
            y = self.data['passorfail'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                stratify=y, 
                                                                test_size=self.spilt_ratio)
            
            # pipeline of numerical data 
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('clipper', IQRClipper()),
                ('scaler', StandardScaler())
            ])

            # pipeline of categorical data
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Overall pipeline
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

            return {"train_X":X_train, 
                    "test_X":X_test, 
                    "train_y":y_train, 
                    "test_y":y_test, 
                    "processor": preprocessor}
            
        else: # test phase
            
            return None