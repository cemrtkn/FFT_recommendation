from sklearn.base import BaseEstimator, TransformerMixin
import torch
import numpy as np


class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, max, min):
        # Compute the min and scale for each feature
        self.min_ = min
        self.scale_ = max - self.min_
        return self
    
    def partial_fit(self,X):
        new_min = torch.min(X).item()
        new_max = torch.max(X).item()

        # Update min and scale using a simple running min/max approach
        if self.min_ is None:
            self.min_ = new_min
            self.scale_ = new_max - new_min
        else:
            self.min_ = np.minimum(self.min_, new_min)
            self.scale_ = np.maximum(self.scale_, new_max - self.min_)

        return self

    def transform(self, X):
        # Apply the Min-Max scaling formula
        X_scaled = (X - self.min_) / self.scale_
        return X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0] + 1e-6


    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):  # Only set attributes that exist in the instance
                setattr(self, param, value)
            else:
                print(f"Skipping parameter {param}: not applicable for {self.__class__.__name__}")
        return self


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon  # A small constant to prevent taking log(0)
        self.shift_value = None
        self.fit()
    
    def fit(self, X=None, y=None):
        # No fitting required for log transformation, so we just return self
        return self
    
    def partial_fit(self,X):
        new_min = torch.min(X).item()

        # Update min and scale using a simple running min/max approach
        if self.shift_value is None:
            self.shift_value = new_min
        else:
            self.shift_value = np.minimum(self.shift_value, new_min)

        return self
    
    
    def transform(self, X):
        # shift value(min) is negative
        X_shifted = X - self.shift_value
        X_transformed = np.log(X_shifted + self.epsilon)
        return X_transformed

    
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):  # Only set attributes that exist in the instance
                print(param, value)
                setattr(self, param, value)
            else:
                print(f"Skipping parameter {param}: not applicable for {self.__class__.__name__}")
        return self
