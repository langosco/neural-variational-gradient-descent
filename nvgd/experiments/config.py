import os

model_size = 'large'
results_path = "./results/"

# Bayesian neural network inference
data_dir = "./data/"
batch_size = 128
n_samples = 100
evaluate_every = 10
num_iter = 400  # for final run (sweep uses less iterations)
dataset = "mnist" # passed to tfds. change to test on other tfds datasets
