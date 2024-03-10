## Rotational Invariance Learning of Satellite Images

## Project Overview

In this project, we explore the challenge of rotational variances in satellite imagery and their impact on machine learning models tasked with object detection.  The aim is to investigate whether a rotation-invariant 2D convolutional neural network (CNN) can perform this task more efficiently compared to a standard CNN.

*The project report can be found in the `notebooks` directory.*

## Steerable CNN

We leverage the Steerable CNN framework, as outlined in [Cohen, Taco, and Welling, "Steerable CNNs" (2016)](https://arxiv.org/pdf/1612.08498.pdf).

For computation, we rely on the `escnn` package.  

## Getting Started

To replicate our environment and run the model, install the necessary libraries listed in the `environment.yaml` file. 

## Wandb.ai Logging Integration
Our model is configured to log training metrics using Weights & Biases (wandb.ai). To use this feature, you must have an active wandb.ai account. 