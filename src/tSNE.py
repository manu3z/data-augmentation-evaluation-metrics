"""Time-series t-SNE visualization

-----------------------------

tSNE.py

"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from lib.load_data import load_data_multiformat
import argparse

def visualizeTSNE(ori_data, generated_data, out_fig_name:str="default", show:bool=False):
    """Using PCA for generated and original data visualization.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
        
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    
    no, seq_len, dim = ori_data.shape  
    
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
    # t-SNE Analysis       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(f"out/figures/{out_fig_name}.png")
    if show:
        plt.show()   

if __name__=="__main__":
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("--stock_energy", type=bool, default=False)
    parser.add_argument("-o", "--out", type=str, default="tSNE")
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

    # Change sizes
    stop = min(len(ori_data), len(gen_data))
    ori_data = ori_data[:stop]
    gen_data = gen_data[:stop]

    # Visualize t-SNE
    print("Plot t-SNE ...")
    visualizeTSNE(ori_data, gen_data, out_fig_name=args.out, show=args.show)
    print(f"Image saved as 'out/figures/{args.out}.png' \n-----")
    