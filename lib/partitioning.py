"""
Module responsible for implementing the different data partitioning strategies.
"""
import skimage.io as ski
import pandas as pd
import numpy  as np

from enum import Enum

class SplitRandomBySingleField:

    def split(self, field_name: str, dataset: pd.DataFrame, train_size: float = 0.8) -> pd.DataFrame:

        grouped_by_field = dataset.groupby(
            field_name
        ).size().reset_index()
        
        train_samples = grouped_by_field.sample(
            frac=train_size
        )[field_name].to_list()
    
        grouped_by_field['partition'] = grouped_by_field[field_name].apply(
            lambda x: "train" if x in train_samples else "test"
        )
    
        return grouped_by_field[[field_name, 'partition']]

class SplitBySingleFieldLogic:

    def split(self, field_name: str, dataset: pd.DataFrame) -> pd.DataFrame:

        grouped_by_field = dataset.groupby(
            field_name
        ).size().reset_index()
        
        train_samples = ['Jarbson']
    
        grouped_by_field['partition'] = grouped_by_field[field_name].apply(
            lambda x: "train" if x in train_samples else "test"
        )
    
        return grouped_by_field[[field_name, 'partition']]        