"""Time-series Histograms

-----------------------------

histogram.py

"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lib.load_data import load_data_multiformat

def plot_histograms(ori_data, gen_data, out_fig_name:str="default", log:bool=True, show:bool=False):
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
        plt.suptitle(f'{["Logarithmic " if log else ""][0]}Histograms of Original Data and Generated Data', y=0.98, weight='normal', size='x-large')
    else:
        fig, axs = plt.subplots(num_subplots, 1, figsize=(15,3*num_subplots))
        for i in range(num_subplots):
            axs[i].hist(ori_data[:,-1,i], density=True, bins=50, label=f'Original Data TS{i+1}', log=log)
            axs[i].hist(gen_data[:,-1,i], density=True, bins=50, alpha=0.7, label=f'Generated Data TS{i+1}', log=log)
            axs[i].legend()
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel(f'{["log " if log else ""][0]}density')
            plt.suptitle(f'{["Logarithmic " if log else ""][0]}Histograms of Original Data and Generated Data', y=0.95, weight='normal', size='x-large')
    plt.savefig(f"out/figures/{out_fig_name}.png")
    if show:
        plt.show()

if __name__ == "__main__":
        # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("-l", "--log", action='store_true')
    parser.add_argument("--stock_energy", type=bool, default=False)
    parser.add_argument("-o", "--out", type=str, default="histogram")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=24)
    args = parser.parse_args()

    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/generated_data_1000e.npy"
    seq_len = args.seq_len

    # Load original data
    ori_data = load_data_multiformat(ori_data_path, seq_len, delim=';')
    # Load synthetic data
    gen_data = load_data_multiformat(gen_data_path, seq_len)

    # Visualize histograms
    print(f"Plot histograms (with log = {args.log}) ...")
    plot_histograms(ori_data, gen_data, out_fig_name=args.out, log=args.log, show=args.show)
    print(f"Image saved as 'out/figures/{args.out}.png' \n-----")
