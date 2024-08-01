# Deep Learning course project

## General notes

* Report for 1st Homework: <https://docs.google.com/document/d/1WmKpuUvtOvTI1ixtbD94QP9vkXnKT654aua3m1N1Y6s/edit#heading=h.f3cva36wqjst>
* Report for 2nd Homework: <https://docs.google.com/document/d/15txipLXKO3fFXNwf4ADRjHOcn8ISf8KXt6-DVO0aF0A/edit?usp=sharing>

This archive contains everything you need to complete Homework 1. It contains
the following files:

## Docker integration:

* Building container:
```sh
docker build -t apre .
```

* Running container:
```sh
docker run -it --rm --name apre apre
```

### download_kuzushiji_mnist.py

- allows you to download the Kuzushiji-MNIST dataset to a compressed .npz file,
  which hw1-q2.py and hw1-q3.py can load.

### hw1-q1.py

- contains skeleton code for Question 1, which covers classification with
  the perception, logistic regression, and the multi-layer perceptron, with
  implementation in numpy.

### hw1-q2.py

- contains skeleton code for Question 2, which covers classification with
  logistic regression and the multi-layer perceptron, with implementation in
  torch.

## Setup and installation

1. Download above datasets into the corresponding resources folder

2. Install all the required packages, run the following command:

```bash
pip install -r requirements.txt
```
