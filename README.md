Intra cluster filtering
=======================

This library provides features for identifying intra-cluster outliers and systematically removing them from the training dataset with minimal code modification and a limited number of new hyperparameters.

Notice that currently supports only pytorch, there is a roadmap to also include this solutions in tensorflow.

Motivation
----------

In the process of optimization through gradient descent, outliers in the data can lead to a decrease in generalization metrics. However, it is important to note that not all misclassified data points are outliers. Therefore, it is proposed to perform segmentation via unsupervised learning on the activations caused by these data points in a hidden layer of the neural network topology, with the goal of identifying the outliers.



Installation instructions
-------------------------

To get the software running you need to install the requirements and this package aswell

1. creation of virtual environment for python (optional, recomended)
```
virtualenv -p python3 venv
```
2. Source the virtual environment
```
source ./venv/bin/activate
```
3. install of requirements
```
pip install -r requirements.txt
```
4. install this package
```
python setupt.py develop
```
Meanwhile is in beta it is recommended to use de develop options instead of install, but feel free to use the install option
                   
Examples
========

The idea of the examples is to help developers to understand how to implement this library on their own projects

Iris dataset sample
-------------------

This is a classical dataset of iris flower clasification, in this case there is just one intra cluster outlier,
in order to detect it using this library we have provided a sample.

To run just:
1. Source the environment
```
source ./venv/bin/activate
```
2. Change directory to sample code
```
cd examples/iris
```
3. run the sample
```
python iris.py
```