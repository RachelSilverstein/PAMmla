import pandas as pd
import numpy as np
from georgiev_params import GEORGIEV_PARAMETERS

class GeorgievEncoder:
    """Mimics structure of sklearn onehot encoder"""
    def __init__(self):
        self.categories = None

    def fit(self, categories):
        cats_list_of_arrays = [np.array(item) for item in categories]
        self.categories = cats_list_of_arrays

    def transform(self, data):
        transformed_data_list = []
        for i in range(len(data)):
            row = data.iloc[i, ]
            enc_array = np.array([[param[AA] for param in GEORGIEV_PARAMETERS] for AA in row])
            transformed_data_list.append(enc_array)
        return transformed_data_list

    def inverse_transform(self, data):
        pass