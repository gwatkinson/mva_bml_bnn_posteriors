# What Are Bayesian Neural Network Posteriors Really Like?

This repository contains some code to run simple experiments on a subset of the MNIST using a simple 2 layer MLP.

Most of the code is taken from [the repository associated with the paper](https://github.com/google-research/google-research/tree/master/bnn_hmc), simplified and modified.

## Requirements

To create an environment, run:

```bash
python -m venv .venv
.venv/bin/activate
```

Then, install the requirements:

```bash
pip install tensorflow

pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.65+cuda112 -f \
https://storage.googleapis.com/jax-releases/jax_releases.html

pip install git+https://github.com/deepmind/dm-haiku
pip install tensorflow_datasets
pip install tabulate
pip install optax
pip install -e .
```

## Running the experiments

To run the experiments, run:

```bash
run.sh
```

which creates checkpoints in the `runs` directory.

Then, to plot a visualisation of the posterior, run:

```bash
visualisation.sh
```

## Note

This doesn't work. Some of the code is not functionning as intented, since the original installation instructions are not working anymore.
And the code generate errors when using the latest version of JAX. I tried unsuccessfully to fix the code, but I didn't have the time to fix it before the deadline.


## Citation

The paper :
> Pavel Izmailov, Sharad Vikram, Matthew D Hoffman, and Andrew Gordon Gordon Wilson. **What are bayesian neural network posteriors really like?** *In International conference on machine learning, pages 4629–4640. PMLR, 2021*

And the repository 
> https://github.com/google-research/google-research/tree/master/bnn_hmc
