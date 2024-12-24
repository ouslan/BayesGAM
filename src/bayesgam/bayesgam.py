import numpy as np

class GAM():
    def __init__(self, formula:str, distribution:str, link:str):
        self.formula = formula 
        self.distribution = distribution
        self.link = link
    
    def s(self, x):
        return np.sin(x)
    
    def f(self, x):
        return x 
        
    def lin(self,x):
        return x

    def te(self,x):
        return x

    def intercept(self,x):
        return x
    
    def fit(self, X, Y):
        formula = self.formula
        
        local_scope = {
            's': self.s,
            'f': self.f,
            'l': self.lin,
            'te': self.te,
            'X': X,
            'intercept': self.intercept
        }
        
        result = eval(formula, {"__builtins__": None}, local_scope)

        return result