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
    ori_data = np.loadtxt(data_path, delimiter = ",",skiprows = 1)#, usecols=[1,2,3])
    
    # Flip the data to make chronological data (ONLY for STOCK and ENERGY datasets)
    if is_stock_energy:
        ori_data = ori_data[::-1]
    # OR remove first column corresponding to the index in non STOCK datasets
    else:
        # ori_data = ori_data[:,1:]
        pass

    if (seq_len != None):
        # Preprocess the dataset
        temp_data = []    
        # Cut data by sequence length
        for i in range(0, len(ori_data) - seq_len+1):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)
            
        # Mix the datasets (to make it similar to i.i.d)
        # idx = np.random.permutation(len(temp_data))
        # data = []
        # for i in range(len(temp_data)):
        #     data.append(temp_data[idx[i]])
        data = temp_data
    else:
        data = ori_data

    return data


def load_data_multiformat(
        ori_data_path:str, 
        seq_len:int,
        delim:str=',',
        stock_energy:bool=False,
        verbose:bool=True,
):

    # Load original data
    # From .csv
    if ori_data_path[-3:] == 'csv':
        ori_data = np.loadtxt(ori_data_path, delimiter = delim, skiprows=1)#, usecols=[1,2,3]) # Para mis sparse datasets (delim=';')s
        if stock_energy:
            ori_data = ori_data[::-1] # Flip dataset
        if (seq_len != None):
            # Preprocess the dataset
            temp_data = []
            # Cut data by sequence length
            for i in range(0, len(ori_data) - seq_len+1):
                _x = ori_data[i:i + seq_len]
                temp_data.append(_x)
            ori_data = temp_data
        # to array
        ori_data = np.asarray(ori_data)
        if verbose:
            print("Data loaded from .csv: ", ori_data.shape)

    # From .npz
    elif ori_data_path[-3:] == 'npz':
        ori_data = np.load(ori_data_path)["data"]
        if verbose:
            print("Data loaded from .npz: ", ori_data.shape)

    # From .npy
    else:
        ori_data = np.load(ori_data_path)
        if verbose:
            print("Data loaded from .npy: ", ori_data.shape)

    # Expand dimensions if necessary
    if ori_data.ndim<3:
        ori_data = np.expand_dims(ori_data, axis=1)

    return ori_data