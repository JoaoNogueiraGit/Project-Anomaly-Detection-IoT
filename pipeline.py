import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class IoTPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        # Carregar o CSV (usando o BoT-IoT como exemplo)
        self.df = pd.read_csv(self.file_path, low_memory=False)
        print(f"Dados carregados: {self.df.shape}")
        return self

    def clean_data(self):
        # 1. Remover espaços nos nomes das colunas
        self.df.columns = self.df.columns.str.strip()
        
        # 2. Lidar com valores infinitos e NaNs (comum em dumps de rede)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        
        # 3. Remover colunas irrelevantes (IDs, Timestamps que podem causar overfitting)
        cols_to_drop = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'daddr'] 
        self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)
        return self

    def encode_and_scale(self, target_column='category'):
        # Separar Features (X) e Target (y)
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Label Encoding para o alvo (transformar 'DDoS' em 1, 'Normal' em 0, etc.)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Normalização (Crucial para algoritmos como KNN, SVM ou Redes Neuronais)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, le

# Exemplo de uso:
# pipeline = IoTPipeline('UNSW_2018_IoT_Botnet_Final_10_Best.csv')
# X, y, encoder = pipeline.load_data().clean_data().encode_and_scale()