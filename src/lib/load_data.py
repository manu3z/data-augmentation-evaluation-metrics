"""
Time-series Data Loader

-----------------------

load_data.py

"""

# Necessary packages
import numpy as np

def load_data(data_path:str, seq_len=None, is_stock_energy:bool=False):
    """Load real-world datasets.

    Args:
    - data_path: path to dataset
    - seq_len: sequence length
    - is_stock_energy: True if dataset corresponds to 'stock' or 'energy'

    Returns:
    - data: preprocessed data.
    """
    ori_data = np.loadtxt(data_path, delimiter = ",",skiprows = 1)
    
    # Flip the data to make chronological data (ONLY for STOCK and ENERGY datasets)
    if is_stock_energy:
        ori_data = ori_data[::-1]

    if (seq_len != None):
        # Preprocess the dataset
        temp_data = []    
        # Cut data by sequence length
        for i in range(0, len(ori_data) - seq_len):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)
            
        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))    
        data = []
        for i in range(len(temp_data)):
            data.append(temp_data[idx[i]])
    else:
        data = ori_data

    return data
