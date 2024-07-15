"""Time-series Histograms

-----------------------------

histogram.py

"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lib.load_data import load_data

def plot_histograms(ori_data, gen_data, out_fig_name:str="default", log:bool=True):
    """
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - out_fig_name: name of the output file
        - log: True is histogram is logarithmic
        
    Returns:
        - histogram plot is displayed and saved in 'out/figures/'
    """
    num_subplots = ori_data.shape[2]

    if num_subplots == 1:
        plt.figure(figsize=(15,3.5))
        plt.hist(ori_data[:,-1,:], density=True, bins=50, label='Original Data', log=log)
        plt.hist(gen_data[:,-1,:], density=True, bins=50, alpha=0.8, label='Generated Data', log=log)
        plt.xlabel('Value')
        plt.ylabel(f'{["log " if log else ""][0]}density')
        plt.legend()
        plt.suptitle(f"{["Logarithmic " if log else ""][0]}Histograms of Original Data and Generated Data", y=0.98, weight='normal', size='x-large')
    else:
        fig, axs = plt.subplots(num_subplots, 1, figsize=(15,3*num_subplots))
        for i in range(num_subplots):
            axs[i].hist(ori_data[:,-1,i], density=True, bins=50, label=f'Original Data TS{i+1}', log=log)
            axs[i].hist(gen_data[:,-1,i], density=True, bins=50, alpha=0.7, label=f'Generated Data TS{i+1}', log=log)
            axs[i].legend()
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel(f'{["log " if log else ""][0]}density')
            plt.suptitle(f"{["Logarithmic " if log else ""][0]}Histograms of Original Data and Generated Data", y=0.95, weight='normal', size='x-large')
    plt.savefig(f"out/figures/{out_fig_name}.png")
    plt.show()

if __name__ == "__main__":
        # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("-l", "--log", action='store_true')
    parser.add_argument("--stock_energy", type=bool, default=False)
    parser.add_argument("-o", "--out", type=str, default="histogram")
    args = parser.parse_args()
    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/generated_data_1000e.npy"
    seq_len = 24
    # Load original data
    if ori_data_path[-3:] == 'csv':
        ori_data = np.asarray(load_data(data_path=ori_data_path, seq_len=seq_len, is_stock_energy=args.stock_energy))
    else:
        ori_data = np.load(ori_data_path)
    print("Original data loaded correctly: ", ori_data.shape)
    # Load generated data
    if gen_data_path[-3:] == 'csv':
        gen_data = np.asarray(load_data(data_path=gen_data_path, seq_len=seq_len, is_stock_energy=False))
    else:
        gen_data = np.load(gen_data_path)
    print("Generated data loaded correctly: ", gen_data.shape)
    if ori_data.ndim<3:
        ori_data = np.expand_dims(ori_data, axis=2)
        gen_data = np.expand_dims(gen_data, axis=2)

    # Visualize t-SNE
    print(f"Plotting Histograms with log = {args.log} ...")
    plot_histograms(ori_data, gen_data, out_fig_name=args.out, log=args.log)
    print(f"Image saved as 'out/figures/{args.out}.png'")
