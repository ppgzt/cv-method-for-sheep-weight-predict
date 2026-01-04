"""
Module responsible for loading the .csv with collection metadata information, such as animal identification code, gender, collect and image data.
"""

import sqlite3, firebase_admin, json, datetime

import pandas as pd
import numpy  as np

from firebase_admin import firestore, credentials
from lib import helpers

class MetadataProvider:

    def __init__(self, source_dir_path: str):
        self.supplies_dir_path = f'{source_dir_path}/supplies'
        self.dataset_dir_path  = f'{source_dir_path}/metadata'
        
    def load_dataframe(self):
        df_collects, df_things, df_images = self.__read_dataframe()

        dataset_v0 = df_collects.query(
            "collect_id.isin(['UYb4dOtZoiguKcF7SK69', 'of8VwxX9TG1PMJhhx8kf', 'pCxbeJYAIoIqLgEz87pB'])"
        ).merge(df_things, on='collect_id')

        dataset_v1 = dataset_v0.merge(
            df_images.query('label.notna()')[['thing_id','image_id','begin_at','final_at','depth','label']], 
            on='thing_id'
        )

        valid_jobs = self.__get_valid_jobs()        
        dataset_v2 = dataset_v1.merge(
            valid_jobs, 
            on='thing_id'
        )

        # Remove the images of the animal with TAG 0473 because the weight was recorded incorrectly ;(
        dataset_v3 = dataset_v2.query('tag != "0473"').iloc[:,:]

        # Images of the animal from TAG 0014 where there was an invasion by 2 other animals.
        runs_of_job_12 = self.__get_images_of_job_12_to_remove()
        dataset_v4 = dataset_v3.query(f"depth not in {runs_of_job_12['file_path'].to_list()}")

        birthdates = self.__get_birthdates()        
        dataset_v5 = dataset_v4.merge(
            birthdates.query('status == 0'), 
            on='tag', 
            how='left'
        )

        dataset_v5['birthdate2'] = dataset_v5.apply(
            lambda row: helpers.millisec_to_date(row['birthdate']), 
            axis=1
        )

        dataset_v5['age'] = dataset_v5.apply(
            lambda x: (helpers.millisec_to_date(x['happenedAt']) - x['birthdate2']).days if not np.isnan(x['birthdate']) else None, 
            axis=1
        )

        return dataset_v5

    def __get_valid_jobs(self):
        jobs_status = pd.read_csv(f'{self.supplies_dir_path}/collects_obstaclesx.csv')
        jobs_status.columns = ['place', 'job_id', 'status', 'obs']
        
        con = sqlite3.connect(f'{self.supplies_dir_path}/cvnode-acaua.db')
        cur = con.cursor()
        
        jobs = pd.read_sql_query("SELECT rowid, * from jobs", con, parse_dates=['begin_at', 'final_at'])
        valid_jobs = jobs.merge(
            jobs_status.query("status in ['Suited', 'Intrusion']"), 
            left_on='rowid', 
            right_on='job_id'
        ).groupby('thing_id').agg({'rowid': lambda x: list(x)}).reset_index()
        
        valid_jobs.columns = ['thing_id', 'jobs']
        return valid_jobs

    def __get_images_of_job_12_to_remove(self):
        con = sqlite3.connect(f'{self.supplies_dir_path}/cvnode-acaua.db')
        cur = con.cursor()
        
        return pd.read_sql_query(
            """SELECT r.rowid, i.file_path 
                FROM runs r
                JOIN itens i on i.run_id = r.rowid
                WHERE r.job_id = 12 
                  AND r.rowid > 559
                  AND i.type = 'DEPTH'""", 
            con,
        ) 
        
    def __get_birthdates(self):
        birthdates_list = []

        for file_path in ['farmaa_birthdates.json','farmb_birthdates.json']:
            with open(f'{self.supplies_dir_path}/{file_path}') as json_file:
                birthdates_list.extend(json.load(json_file)['results'])
        
        birthdates = pd.DataFrame.from_records(birthdates_list)
        birthdates.columns = ['user','tag','birthdate','status']
        
        return birthdates

    def __read_dataframe(self):
        df_collects = pd.read_csv(f'{self.dataset_dir_path}/collects.csv')
        df_things = pd.read_csv(f'{self.dataset_dir_path}/things.csv')
        df_images = pd.read_csv(f'{self.dataset_dir_path}/images.csv')

        return (df_collects, df_things, df_images)