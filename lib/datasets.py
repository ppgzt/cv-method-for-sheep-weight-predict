"""
Module responsible for loading the images used in the experiment.
"""

import skimage.io as ski
import pandas as pd
import numpy  as np
import glob

class Dataset:

    def load_data(
        self, 
        dataframe: pd.DataFrame, 
        img_col_name: str, 
        img_dir: str, 
        truth_col_name: str, 
        transformations: list = [],
        replicators: list = []):

        self.__X_train = []
        self.__Y_train = [] 
        self.__X_test  = []
        self.__Y_test  = []
        
        dataframe.apply(
            lambda row: self.__fill(
                row, 
                img_col_name, 
                img_dir, 
                truth_col_name, 
                transformations, 
                replicators
            ), 
            axis = 1
        )
        
        self.X_train = np.array(self.__X_train)
        self.Y_train = np.array(self.__Y_train)
        self.X_test  = np.array(self.__X_test)
        self.Y_test  = np.array(self.__Y_test)
        
        return (
            (self.X_train, self.Y_train), 
            (self.X_test , self.Y_test )
        )

    def __fill(
        self, 
        row: pd.Series, 
        img_col_name: str, 
        img_dir: str, 
        truth_col_name: str, 
        transformations: list = [],
        replicators: list = []):
        
        x_data = ski.imread(f'{img_dir}/{row[img_col_name]}')
        y_data = row[truth_col_name]

        for trf in transformations:
            x_data = trf.transform(x_data)

        if row['partition'] == 'test':
            self.__X_test.append(x_data)
            self.__Y_test.append(y_data)
        else:
            self.__X_train.append(x_data)
            self.__Y_train.append(y_data)

            for rep in replicators: 
                self.__X_train.append(rep.transform(x_data))
                self.__Y_train.append(y_data)

    def load_img(
        self, 
        img_file_name: str, 
        img_dir: str, 
        transformations: list = []):
        
        x_data = ski.imread(f'{img_dir}/{img_file_name}')
        for trf in transformations:
            x_data = trf.transform(x_data)

        return x_data