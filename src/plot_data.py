"""Time-series Plot Original and Generated Data

-----------------------------

plot_data.py

"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lib.load_data import load_data_multiformat

def plot_data(ori_data, gen_data, out_fig_name:str="default", plotvals:int=None, show:bool=False):
    """
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - out_fig_name: name of the output file
        
    Returns:
        - data is plotted, displayed and saved in 'out/figures/'
    """
    num_subplots = ori_data.shape[2]
    if plotvals == None:
        ori_data_len = ori_data.shape[0]
        gen_data_len = gen_data.shape[0]
    else:
        ori_data_len = plotvals
        gen_data_len = plotvals

    if num_subplots == 1:
        plt.figure(figsize=(15,3.5))
        plt.plot(np.linspace(0,ori_data_len,ori_data_len), ori_data[:,-1,:], label=f'Original Data')
        plt.plot(np.linspace(0,gen_data_len,gen_data_len), gen_data[:,-1,:], label=f'Generated Data', c='k', alpha=0.7)
        plt.ylabel('Value')
        plt.legend()
        plt.suptitle("Original Data and Generated Data Plot", y=0.98, weight='normal', size='x-large')
    else:
        colors_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','dodgerblue','salmon','goldenrod' ]
        colors_list_dark = ['darkblue','sienna','darkgreen','darkred','purple','saddlebrown','deeppink','dimgrey','olive','darkturquoise','steelblue','orangered','darkgoldenrod' ]
        fig, axs = plt.subplots(num_subplots, 1, figsize=(15,2.5*num_subplots))
        for i in range(num_subplots):
            axs[i].plot(np.linspace(0,ori_data_len,ori_data_len), ori_data[:ori_data_len,-1,i], label=f'Original Data {i+1}', c=colors_list[i])
            axs[i].plot(np.linspace(0,gen_data_len,gen_data_len), gen_data[:gen_data_len,-1,i], label=f'Generated Data {i+1}', c=colors_list_dark[i], alpha=0.7)
            axs[i].set_ylabel('Value')
            axs[i].legend()
            plt.suptitle("Original Data and Generated Data Plots", y=0.95, weight='normal', size='x-large')
            
    plt.savefig(f"out/figures/{out_fig_name}.png")
    if show:
        plt.show()

if __name__ == "__main__":
        # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("--stock_energy", type=bool, default=False)
    parser.add_argument("-s", "--samples_to_plot", type=int, default=None, help="Defines how many samples to plot from the beginning")
    parser.add_argument("-o", "--out", type=str, default="plot")
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()

    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/generated_data_1000e.npy"
    seq_len = None

    # Load original data
    ori_data = load_data_multiformat(ori_data_path, seq_len)
    # Load synthetic data
    gen_data = load_data_multiformat(gen_data_path, seq_len)

    if args.samples_to_plot != None:
        assert args.samples_to_plot <= ori_data.shape[0], "Samples to plot must be smaller than the total values in ori_data"
        assert args.samples_to_plot <= gen_data.shape[0], "Samples to plot must be smaller than the total values in gen_data"

    # Visualize plots
    print("Plot 1 ...")
    plot_data(ori_data, gen_data, out_fig_name=args.out, plotvals=args.samples_to_plot, show=args.show)
    print(f"Image saved as 'out/figures/{args.out}.png' \n-----")
