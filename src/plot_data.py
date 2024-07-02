"""Time-series Plot Original and Generated Data

-----------------------------

plot_data.py

"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lib.load_data import load_data

def plot_data(ori_data, gen_data, out_fig_name:str="default", plotvals:int=None):
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
        colors_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown', ]
        fig, axs = plt.subplots(num_subplots, 1, figsize=(15,2.5*num_subplots))
        for i in range(num_subplots):
            axs[i].plot(np.linspace(0,ori_data_len,ori_data_len), ori_data[:ori_data_len,-1,i], label=f'Original Data {i+1}', c=colors_list[i])
            axs[i].plot(np.linspace(0,gen_data_len,gen_data_len), gen_data[:gen_data_len,-1,i], label=f'Generated Data {i+1}', c='k', alpha=0.7)
            axs[i].set_ylabel('Value')
            axs[i].legend()
            plt.suptitle("Original Data and Generated Data Plots", y=0.95, weight='normal', size='x-large')
    plt.savefig(f"out/figures/{out_fig_name}.png")
    plt.show()

if __name__ == "__main__":
        # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("--stock_energy", type=bool, default=True)
    parser.add_argument("-s", "--samples_to_plot", type=int, default=None, help="Defines how many samples to plot from the beginning")
    parser.add_argument("-o", "--out", type=str, default="plot")
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

    if args.samples_to_plot != None:
        assert args.samples_to_plot <= ori_data.shape[0], "Samples to plot must be smaller than the total values in ori_data"
        assert args.samples_to_plot <= gen_data.shape[0], "Samples to plot must be smaller than the total values in gen_data"

    # Visualize t-SNE
    print("Plotting Original and Generated data...")
    plot_data(ori_data, gen_data, out_fig_name=args.out, plotvals=args.samples_to_plot)
    print(f"Image saved as 'out/figures/{args.out}.png'")
