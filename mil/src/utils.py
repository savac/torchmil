import pandas as pd
import numpy as np
from typing import List
import torch


def make_dataset_from_dataframe(
    df: pd.DataFrame, feature_names: List[str], target_name: str, bag_name: str
):
    """Given a dataframe, create a dataset suitable for training a MILR model.

    Args:
        df (pd.DataFrame): Full dataset with features, target variable, and a variable
            denoting the bag membership
        feature_names (List[str]): List of features
        target_name (str): Target variable name
        bag_name (str): Bag variable name

    Returns:
        X (np.ndarray): Array of features
        y (np.ndarray): Vector of target variable
        bags (np.ndarray): List of vectors containing indices in X denoting bag membership
    """
    bags = []
    y = []
    for mn in df[bag_name].unique():
        this_bag = df[(df.molecule_name == mn)]
        bags.append(this_bag.index.values)
        y.append(this_bag[target_name].values[0])
    y = np.array(y)
    X = df[feature_names].values
    return X, y, bags


def invlogit(x):
    return 1 / (1 + torch.exp(-x))


def logsumexp_safe(x, r, mask):
    # x domain should be [0, 1] so we set the padded values to 0
    x *= mask.float()
    m = x.amax(dim=1, keepdim=True)
    x0 = x - m
    bag_lengths = mask.float().sum(dim=1, keepdim=True)
    return (
        r * m
        + torch.log(
            torch.sum(torch.exp(r * x0) * mask.float(), dim=1, keepdim=True)
            / bag_lengths
        )
    ) / r


def generalized_mean(x, r, mask):
    # x domain should be [0, 1] so we set the padded values to 0
    x *= mask.float()
    bag_lengths = mask.float().sum(dim=1, keepdim=True)
    return torch.pow(
        torch.sum(torch.pow(x, r), dim=1, keepdim=True) / bag_lengths, 1 / r
    )
