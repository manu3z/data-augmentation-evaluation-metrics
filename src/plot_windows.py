"""Time-series Plot Original and Generated Data Windows

-----------------------------

plot_windows.py

"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lib.load_data import load_data
import pandas as pd

def plot_windows(ori_data, gen_data, out_fig_name:str="default", windows:int=3, show:bool=False):
    """
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - out_fig_name: name of the output file
        
    Returns:
        - data is plotted, displayed and saved in 'out/figures/'
    """

    nseries = ori_data.shape[-1]
    nrows = nseries
    ncols = windows

    time = list(range(1,25))

    #Plotting some generated samples.
    fig = plt.figure(constrained_layout=True, figsize=(ncols*4,nrows*2))
    # fig.tight_layout()
    subfigs = fig.subfigures(nseries, 1)
    fig.suptitle("Original Data and Generated Data Windows")

    for outerind, subfig in enumerate(subfigs.flat):
        axs = subfig.subplots(1, ncols)
        subfig.suptitle(f'TS{outerind+1}')

        for col in range(ncols):
            obs = np.random.randint(len(ori_data))
            axs[col].plot(ori_data[obs, :, outerind%12], label='Real')
            axs[col].plot(gen_data[obs, :, outerind%12], label='Synth')
            axs[col].legend(loc='upper right')
            axs[col].set_xticks([])        


    # if num_subplots == 1:
    #     plt.figure(figsize=(15,3.5))
    #     plt.plot(np.linspace(0,ori_data_len,ori_data_len), ori_data[:ori_data_len,-1,:], label=f'Original Data')
    #     plt.plot(np.linspace(0,gen_data_len,gen_data_len), gen_data[:gen_data_len,-1,:], label=f'Generated Data', c='k', alpha=0.7)
    #     plt.ylabel('Value')
    #     plt.legend()
    #     plt.suptitle("Original Data and Generated Data Plot", y=0.98, weight='normal', size='x-large')
    # else:
    #     colors_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','dodgerblue','salmon','goldenrod' ]
    #     colors_list_dark = ['darkblue','sienna','darkgreen','darkred','purple','saddlebrown','deeppink','dimgrey','olive','darkturquoise','steelblue','orangered','darkgoldenrod' ]

    #     fig = plt.figure(constrained_layout=True, figsize=(10,3*num_subplots))
    #     subfigs = fig.subfigures(num_subplots, 1)
    #     fig.suptitle("Original Data and Generated Data Plots")
    #     i=0
    #     for outerind, subfig in enumerate(subfigs.flat):
    #         axs = subfig.subplots(2, 1)
    #         subfig.suptitle(f'TS{i+1}')

    #         axs[0].plot(np.linspace(0,ori_data_len,ori_data_len), ori_data[:ori_data_len,-1,i], label=f'Original Data {i+1}', c=colors_list[i])
    #         axs[0].legend(loc='upper right')
    #         axs[0].set_xticks([])
    #         axs[0].set_ylabel('Value')

    #         axs[1].plot(np.linspace(0,gen_data_len,gen_data_len), gen_data[:gen_data_len,-1,i], label=f'Generated Data {i+1}', c=colors_list_dark[i])
    #         axs[1].legend(loc='upper right')
    #         axs[1].set_ylabel('Value')
    #         i=i+1

    plt.savefig(f"out/figures/{out_fig_name}.png")
    if show:
        plt.show()

if __name__ == "__main__":
        # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("--stock_energy", type=bool, default=False)
    parser.add_argument("-w", "--windows", type=int, default=None, help="Defines how many windows to plot per series (default=3)")
    parser.add_argument("-o", "--out", type=str, default="plot_windows")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length (default=24)")
    args = parser.parse_args()
    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/generated_data_1000e.npy"
    seq_len = args.seq_len

    # Load original data
    if ori_data_path[-3:] == 'csv':
        ori_data = np.asarray(load_data(data_path=ori_data_path, seq_len=seq_len, is_stock_energy=args.stock_energy))
        print("Original data loaded from .csv: ", ori_data.shape)
    else:
        ori_data = np.load(ori_data_path)
        print("Original data loaded from .npy: ", ori_data.shape)
    # Load generated data
    if gen_data_path[-3:] == 'csv':
        gen_data = np.asarray(load_data(data_path=gen_data_path, seq_len=seq_len, is_stock_energy=False))
        print("Generated data loaded from .csv: ", gen_data.shape)
    else:
        gen_data = np.load(gen_data_path)
        print("Generated data loaded from .npy: ", gen_data.shape)
    # Expand dimensions if necessary
    if ori_data.ndim<3:
        ori_data = np.expand_dims(ori_data, axis=1)
        gen_data = np.expand_dims(gen_data, axis=1)

    # Check they are both the same size and change if needed
    if len(ori_data) < len(gen_data):
        gen_data = gen_data[:len(ori_data)]
        print("New generated data shape: ", gen_data.shape)
    elif len(ori_data) > len(gen_data):
        ori_data = ori_data[:len(gen_data)]
        print("New original data shape: ", ori_data.shape)

    # Visualize windows
    print("Plotting Original and Generated data...")
    plot_windows(ori_data, gen_data, out_fig_name=args.out, windows=args.windows, show=args.show)
    print(f"Image saved as 'out/figures/{args.out}.png'")
