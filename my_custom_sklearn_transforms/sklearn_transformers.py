from sklearn.base import BaseEstimator, TransformerMixin


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

from sklearn.preprocessing import MinMaxScaler
#Normalizar colunas
class Normalizar:
    def __init__(self):
        self.data = []
        
    def transformar(self):
        mms =  MinMaxScaler()
        cols = [
               "REPROVACOES_DE", "REPROVACOES_EM", "REPROVACOES_MF", "REPROVACOES_GO",
               "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
               "INGLES", "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
        ]
        data2 = self.copy()
        data2[cols] = mms.fit_transform(self[cols])
        return data2
