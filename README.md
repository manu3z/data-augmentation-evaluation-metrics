# data-augmentation-evaluation-metrics
 Metrics/benchmarks to evaluate the performance of Data Augmentation techniques.

## PCA visualization:


Principal component analysis (PCA) is a linear dimensionality reduction technique. The output figure corresponds to the 2 most relevant dimensions.

```bash
python src/PCA.py -o src/data/original/stock_data.csv -g src/data/generated/generated_data.npy
```

## t-SNE visualization

The t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. In this case, the output figure shows the two-dimensional map.  

```bash
python src/tSNE.py -o src/data/original/stock_data.csv -g src/data/generated/generated_data.npy
```

## Discriminative metrics
This method evaluates the classification accuracy between original and synthetic data using post-hoc RNN network.

```bash
python src/
```


## Predictive metrics
This method evaluates the prediction performance on *"train-on-synthetic, test-on-real setting"*. More specifically, a Post-hoc RNN architecture is used to predict one-step ahead and report the performance in terms of the Mean Absolute Error (MAE).

```bash
python src/
```
