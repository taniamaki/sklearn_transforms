from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class SmoteResample(object):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        return X_resampled, y_resampled
