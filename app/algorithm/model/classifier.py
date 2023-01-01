
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.svm import SVC

model_fname = "model.save"
MODEL_NAME = "bin_class_base_svc_sklearn"


class Classifier(): 
    
    def __init__(self, C = 1.0, kernel = "rbf", degree = 3, tol = 1e-3, **kwargs) -> None:
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.tol = float(tol)
        
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = SVC(C = self.C, degree = self.degree, tol = self.tol, 
            probability=True,
            kernel = self.kernel)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        return self.model.predict(X)          
        
    
    def predict_proba(self, X, verbose=False): 
        return self.model.predict_proba(X) 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


