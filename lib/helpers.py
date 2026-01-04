"""
Module that contains helper functions.
"""
import datetime
import numpy as np

def millisec_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp / 1000) if not np.isnan(timestamp) else None