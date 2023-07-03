import pandas as pd
from pathlib import Path
from typing import Union, List
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

class DataGenerator:
    def __init__(self, 
            train_path: Union[str, Path], 
            test_path: Union[str, Path],
            target_col: str, 
            cv: int = 5,
            group_col: str = None, 
            remove_cols: List = []
        ):
        """_summary_

        Args:
            train_path (Union[str, Path]): Path to the csv file for training set
            test_path (Union[str, Path]): Path to the csv file for test set
            target_col (str): Name of the target column
            cv (int, optional): Number of Cross Validation. Defaults to 5.
            group_col (str, optional): Name of the group column, if it exists.
            remove_cols (List, optional):Names of the removed columns. Defaults to [].
        """
        self.data = pd.read_csv(train_path)
        remove_cols.extend([target_col, group_col])
        selected_cols = []
        for col in (self.data.columns):
            if col not in remove_cols:
                selected_cols.append(col)
        
        self.train = self.data.loc[:, selected_cols] 
        self.target = self.data.loc[:, target_col]
        self.group_col = group_col
        
        self.test = pd.read_csv(test_path).loc[:, selected_cols]
        
        assert all(x == y for x, y in zip(self.train.columns, self.test.columns))
                
        self.cv_train_idxs, self.cv_val_idxs = self._get_cv_idxs(cv)
            
        self._curr_idx = 0
    
    def _get_cv_idxs(self, cv):
        args = [self.train, self.target]

        if self.group_col:
            group = self.train.loc[:, self.group_col]
            args.append(group)
            self.skf = StratifiedGroupKFold(n_splits=cv)
        else:
            self.skf = StratifiedKFold(n_splits=cv)
        
        cv_train_idxs = []
        cv_val_idxs = []
        for train_idxs, val_idxs in self.skf.split(*args):
            cv_train_idxs.append(train_idxs)
            cv_val_idxs.append(val_idxs)
        return cv_train_idxs, cv_val_idxs
    
    def __len__(self):
        return self.skf.get_n_splits()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._curr_idx < len(self):
            train_idxs = self.cv_train_idxs[self._curr_idx]
            val_idxs = self.cv_val_idxs[self._curr_idx]
            
            X_train, y_train = self.train.iloc[train_idxs], self.target.iloc[train_idxs]
            X_val, y_val = self.train.iloc[val_idxs], self.target.iloc[val_idxs]
            X_test = self.test
            self._curr_idx += 1
            return (X_train, y_train), (X_val, y_val), X_test
        
        self._curr_idx = 0
        raise StopIteration


                    





# class UniversityClassIter:    
# def __init__(self, university_class):
#         self._lect = university_class.lecturers
#         self._stud = university_class.students
#         self._class_size = len(self._lect) + len(self._stud)
#         self._current_index = 0    def __iter__(self):
#         return self    def __next__(self):
#         if self._current_index < self._class_size:
#             if self._current_index < len(self._lect):
#                 member = self._lect[self._current_index] 
#             else:
#                 member = self._stud[
#                     self._current_index - len(self._lect)]            self._current_index += 1
#             return member        raise StopIteration