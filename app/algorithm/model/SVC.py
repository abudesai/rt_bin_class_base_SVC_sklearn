
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.svm import SVC

model_fname = "model.save"
MODEL_NAME = "binary_class_SVC_sklearn"


class SVC_sklearn(): 
    
    def __init__(self, C = 1.0, degree = 3, tol = 1e-3, kernel = "rbf", **kwargs) -> None:
        super(SVC_sklearn, self).__init__(**kwargs)
        self.C = float(C)
        self.degree = int(degree)
        self.tol = float(tol)
        self.kernel = kernel
        
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = SVC(C = self.C, degree = self.degree, tol = self.tol, kernel = self.kernel)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        SVclassifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return SVclassifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


