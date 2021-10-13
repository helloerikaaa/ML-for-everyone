import math
import numpy as np
from tqdm import tqdm


class NaiveBayes():
    
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        
        for i, c in tqdm(enumerate(self.classes)):
            # filter rows where the label is equal to the given class
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            
            # Mean and variance for each column
            for col in X_where_c.T:
                parameters = {'mean': col.mean(), 'var': col.var()}
                self.parameters[i].append(parameters)
    
    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-4 # prevent division by zero
        coef = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exp = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coef * exp
    
    def _calculate_prior(self, c):
        freq = np.mean(self.y == c)
        return freq
    
    def _classify(self, sample):
        posteriors = []
        
        for i, c in tqdm(enumerate(self.classes)):
            posterior = self._calculate_prior(c)
            
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._calculate_likelihood(params['mean'], params['var'], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        
        y_pred = [self._classify(sample) for sample in X]
        return y_pred
