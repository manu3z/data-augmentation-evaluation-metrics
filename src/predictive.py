"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Code author: Jinsung Yoon (jsyoon0823@gmail.com)
Code adaptation: Manuel Sánchez Laguardia

-----------------------------

predictive.py

Note: This code was migrated from TF1 to Tensorflow 2

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from lib.utils import extract_time
from lib.load_data import load_data
import argparse

tf.compat.v1.disable_eager_execution()

def predictive_score_metrics (ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
    
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  

    ## Builde a post-hoc RNN predictive network 
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
    T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")    
    Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

    # Predictor function
    def predictor (x, t):
        """Simple predictor function.
        Args:
          - x: time-series data
          - t: time information
          
        Returns:
          - y_hat: prediction
          - p_vars: predictor variables
        """
        class Mask(tf.keras.Layer):
            def call(self, t):
                return tf.expand_dims(tf.sequence_mask(t, dtype=tf.float32), axis=-1)
        
        with tf.compat.v1.variable_scope("predictor", reuse = tf.compat.v1.AUTO_REUSE) as vs:
            p_cell = tf.keras.layers.GRUCell(units=hidden_dim, activation='tanh', name = 'p_cell')
            p_outputs, p_last_states = tf.keras.layers.RNN(p_cell, return_sequences=True, return_state=True)(x)
            mask = Mask()(t)
            p_outputs = p_outputs * mask
            y_hat_logit = tf.keras.layers.Dense(1, activation=None) (p_outputs)
            y_hat = tf.keras.activations.sigmoid(y_hat_logit)
            p_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]
        
        return y_hat, p_vars
    
    y_pred, p_vars = predictor(X, T)
    # Loss for the predictor
    p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
    # optimizer
    p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)

    ## Training    
    # Session start
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
      
    # Training using Synthetic dataset
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]     

        X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
        T_mb = list(generated_time[i]-1 for i in train_idx)
        Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
        
        # Train predictor
        _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
        print(itt, end="\r", flush=True)

    ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(ori_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)

    # Prediction
    pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
    predictive_score = MAE_temp / no

    return predictive_score


if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("-i", "--iterations", type=int, default=5)
    parser.add_argument("--stock_energy", type=bool, default=False)
    args = parser.parse_args()
    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/stock-data_TimeGAN_tf1_1000e.npy"
    seq_len = 24
    # Load original data
    if ori_data_path[-3:] == 'csv':
        ori_data = np.asarray(load_data(data_path=ori_data_path, seq_len=seq_len, is_stock_energy=args.stock_energy))
    else:
        ori_data = np.load(ori_data_path)
    print("Load original dataset ok: ", ori_data.shape)
    # Load generated data
    if gen_data_path[-3:] == 'csv':
        gen_data = np.asarray(load_data(data_path=gen_data_path, seq_len=seq_len, is_stock_energy=False))
    else:
        gen_data = np.load(gen_data_path)
    print("Load generated dataset ok:", gen_data.shape)

    # Predictive score calculation
    metric_iteration = args.iterations
    predictive_score = list()
    for i in range(metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, gen_data)
        predictive_score.append(temp_pred)
        # Print dynamic iteration state
        print(f"Iteration {i+1} score: {temp_pred}")
        
    print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))
