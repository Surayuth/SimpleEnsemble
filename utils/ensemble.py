import numpy as np
from typing import List
from loguru import logger

class Ensembler:
    def __init__(self, 
            models: List[object], 
            params: List[dict],
            save_path: str = None,
        ):
        """_summary_
        Args:
            models (List[object]): Models for ensembling.
            params (List[dict], optional): Initial hyperparameters of models. 
                The order of the elements must match the order of models in the first input.
            save_path (str, optional): Path to save the weights of trained models. Defaults to None.
        """
        assert len(models) == len(params), "the length of the set of hyperparameters doesn't match the number of input models"
        assert len(models) > 1, "require at least 2 models for ensembling!"
        
        self.models = []
        for model, param in zip(models, params):
            self.models.append(model(**param))
        self.save_path = save_path 
        logger.debug("Currently, no model is saved.")
    
    def fit(self, X_train, y_train):
        for model in self.models:
            model.fit(X_train, y_train)
        
    def predict_proba(self, X_test): # prob only
        preds = np.zeros((len(self.models), len(X_test)))
        for i, model in enumerate(self.models):
            preds[i] = model.predict_proba(X_test)[:, 1]
        return preds

    
    