## import importent packages
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib
import logging

plt.style.use("fivethirtyeight")


class Perceptron:
    def __init__(self, eta: float=None, epochs: int=None):
        self.weights = np.random.randn(3) * 1e-4
        is_training = (eta is not None) and (epochs is not None)
        if is_training:logging.info(f"Initial weights before traning: \n{self.weights}")
            
        self.eta = eta
        self.epochs = epochs
    
    def _z_outcome(self, inputs, weights):
        return np.dot(inputs, weights)
    
    def activation_function(self, z):
        return np.where(z > 0, 1, 0)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        logging.info(f"X with bias: \n{X_with_bias}")
        
        for epoch in range(self.epochs):
            logging.info("--"*20)
            logging.info(f"for epoch >> {epoch+1}")
            logging.info("--"*20)
            
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            logging.info(f"Predicted value after forward pass: \n{y_hat}")
            
            self.error = self.y - y_hat
            logging.info(f"Error: \n{self.error}")
            
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            logging.info(f"Updated weights after epoch: {epoch+1}/{self.epochs}: \n{self.weights}")
            logging.info("##"*20)
            
    def predict(self, X_pred):
        X_pred_with_bias = np.c_[X_pred, -np.ones((len(X_pred), 1))]
        z_pred = self._z_outcome(X_pred_with_bias, self.weights)
        y_outcome = self.activation_function(z_pred)
        
        return y_outcome
    
    def total_loss(self):
        loss = np.sum(self.error)
        logging.info(f"\nTotal loss: {loss}\n")
        return loss
    
    def _create_dir_return_path(self, model_dir, file_name):
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, file_name)
    
    def save(self, file_name, model_dir=None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, file_name)
            joblib.dump(self, model_file_path)
            logging.info(f"Model is saved in {model_file_path}")
        else:
            model_file_path = self._create_dir_return_path('model', file_name)
            joblib.dump(self, model_file_path)
            logging.info(f"Model is saved in {model_file_path}")
    
    def load(self, filepath):
        logging.info("Model loded \n")
        return joblib.load(filepath)
        