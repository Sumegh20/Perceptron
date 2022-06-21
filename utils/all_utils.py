import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import logging

plt.style.use("fivethirtyeight")

def X_y_split(df, target_col='y'):
    logging.info("Split the dataset into X and y")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X,y

def save_plot(df, model, filename='plot.png', plot_dir='plots'):
    def _create_base_plot(df):
        logging.info("Ploting the dataset")
        df.plot(kind="scatter", x='x1', y="x2", c="y", s=100, cmap='coolwarm')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        figure = plt.gcf()
        figure.set_size_inches(10, 8)
    
    def _plot_decision_region(X, y, model, resolution=0.03):
        logging.info("Ploting the decision regions")
        color = ("cyan","lightgreen")
        cmap = ListedColormap(color)
        
        X = X.values
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        x1_min, x1_max = x1.min()-1, x1.max()+1
        x2_min, x2_max = x2.min()-1, x2.max()+1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                              )
        
        y_hat = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.plot()
        
    X, y =  X_y_split(df)
    
    _create_base_plot(df)
    _plot_decision_region(X, y, model)
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    logging.info("Save the plot in {plot_path}")