# CycleGAN
 
This notebook is a demonstration of implementing CycleGAN

Most of the implementations follow the instruction in [this paper](https://arxiv.org/abs/1703.10593)

The provided codes work on any dataset, so feel free to use your own dataset to train a CycleGAN

### Dataset
The dataset is the image dataset of Yosemite national park. The seasons, summer and winter, will be the domain

### Background 
To train a CycleGAN, the labels are not required. The goal is to train a generator that learns the mapping between domain X and domain Y

### File Introduction
* model.py: The model structure
* loss_fun.py: The loss function
* helpers.py: Some functions that have been used throughout the building of CycleGAN
* train.py: The training step of CycleGAN
* run.py: Run this file to train your own CycleGAN. You have to specify your own dataset's directory

### Special Thanks to Udacity deep learning Nanodegree Program
The code in this Repo is built upon the skeleton provided by the udacity community. Most of the parts are written with my own effort, except that the udacity offer heavy skeleton in the actual training part and the helper functions. I am able to understand the implementation of CycleGAN after having implemented by myself
