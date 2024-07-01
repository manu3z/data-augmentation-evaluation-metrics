"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Code author: Jinsung Yoon (jsyoon0823@gmail.com)
Code adaptation: Manuel Sánchez Laguardia

-----------------------------

predictive_metrics.py

Note: This code was migrated from TF1 to Tensorflow 2

Note: Use post-hoc RNN to classify original data and synthetic data
Output: discriminative score (np.abs(classification accuracy - 0.5))

"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from lib.utils import train_test_divide, extract_time, batch_generator
from lib.load_data import load_data
import argparse

tf.compat.v1.disable_eager_execution()

def discriminative_score_metrics (ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        
    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    
        
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
        
    ## Builde a post-hoc RNN discriminator network 
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128
        
    # Inputs
    # Feature
    X = tf.compat.v1.placeholder(shape=[None, max_seq_len, dim], dtype=tf.float32, name = "myinput_x")
    X_hat = tf.compat.v1.placeholder(shape=[None, max_seq_len, dim], dtype=tf.float32, name = "myinput_x_hat")
        
    T = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name = "myinput_t")
    T_hat = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name = "myinput_t_hat")
        
    # discriminator function
    def discriminator (x, t):
        """Simple discriminator function.
        
        Args:
        - x: time-series data
        - t: time information
        
        Returns:
        - y_hat_logit: logits of the discriminator output
        - y_hat: discriminator output
        - d_vars: discriminator variables
        """
        class Mask(tf.keras.Layer):
            def call(self, t):
                return tf.expand_dims(tf.sequence_mask(t, dtype=tf.float32), axis=-1)

        with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE) as vs:
            d_cell = tf.keras.layers.GRUCell(units=hidden_dim, activation='tanh', name = 'd_cell')
            d_outputs, d_last_states = tf.keras.layers.RNN(d_cell, return_sequences=True, return_state=True)(x)
            mask = Mask()(t)
            d_outputs = d_outputs * mask
            # d_last_states = d_last_states * mask
            y_hat_logit = tf.keras.layers.Dense(1, activation=None) (d_last_states)
            y_hat = tf.keras.activations.sigmoid(y_hat_logit)
            d_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]
        
        return y_hat_logit, y_hat, d_vars
    
    y_logit_real, y_pred_real, d_vars = discriminator(X, T)
    y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
    # Loss for the discriminator
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
                                                                        labels = tf.ones_like(y_logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
                                                                        labels = tf.zeros_like(y_logit_fake)))
    d_loss = d_loss_real + d_loss_fake

    # optimizer
    d_solver = tf.compat.v1.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
    ## Train the discriminator   
    # Start session and initialize
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
    train_test_divide(ori_data, generated_data, ori_time, generated_time)
        
    # Training step
    for itt in range(iterations):
            
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
            
        # Train discriminator
        _, step_d_loss = sess.run([d_solver, d_loss], 
                                feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})
        print(itt, end="\r", flush=True)
        
    ## Test the performance on the testing set    
    y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final>0.5))
    discriminative_score = np.abs(0.5-acc)

    return discriminative_score  


if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ori_data", type=str)
    parser.add_argument("-g", "--gen_data", type=str)
    parser.add_argument("-i", "--iterations", type=int, default=5)
    args = parser.parse_args()
    # Define parameters
    ori_data_path = args.ori_data #"src/data/original/stock_data.csv"
    gen_data_path = args.gen_data #"src/data/generated/stock-data_TimeGAN_tf1_1000e.npy"
    seq_len = 24
    # Load original data
    if ori_data_path[-3:] == 'csv':
        ori_data = np.asarray(load_data(data_path=ori_data_path, seq_len=seq_len, is_stock_energy=True))
    else:
        ori_data = np.load(ori_data_path)
    print("Original data loaded correctly: ", ori_data.shape)
    # Load generated data
    if gen_data_path[-3:] == 'csv':
        gen_data = np.asarray(load_data(data_path=gen_data_path, seq_len=seq_len, is_stock_energy=True))
    else:
        gen_data = np.load(gen_data_path)
    print("Generated data loaded correctly: ", gen_data.shape)

    # Discriminative score calculation
    metric_iteration = args.iterations
    discriminative_score = list()
    for i in range(metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, gen_data)
        discriminative_score.append(temp_disc)
        # Print dynamic iteration state
        print(f"Iteration {i+1} score: {temp_disc}")

    print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))
