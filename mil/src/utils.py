import pandas as pd
import numpy as np
from typing import List


def make_dataset_from_dataframe(df: pd.DataFrame, feature_names: List[str], target_name: str, bag_name: str):
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
