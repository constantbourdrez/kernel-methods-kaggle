# kernel-methods-kaggle

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)


## Introduction

Repository for the kaggle challenge of the kernel methods course of the MVA/IASD master

## Installation

To use the repo you must execute these commands:

```bash
# Example installation steps
git clone https://github.com/constantbourdrez/kernel-methods-kaggle
cd kernel-methods-kaggle
pip install -r requirements.txt
```

## Usage

If you want to generate the submission csv:

```bash
# Example usage
python start.py --kernel_type desiredkernel --method desiredmethod
```

## File Structure


- `classifiers.py`: Python script for classification tasks. It contains the implementationf of a kernel logistic regression and a svm
- `hyperparameters_tuning.py`: Python script for hyperparameter tuning for svm.
- `kernels.py`: Python script where one can find several classes of kernel
- `pipeline.py`: Python script for training the model and compute validation metrics.
- `start.py`: Python script to generate the output.
- `utils.py`: Utility functions for the project such as dataloaders, accuracy computation, etc.
