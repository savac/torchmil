# Project Name

torchmil

## Description

This repo contains the implementation of a model for Multi-Instance Logistic Regression in PyTorch.


## Usage

Instructions on how to use your project.

Here's how to create an instance of the `MILR` class and train it:

##### Download example data
```bash
mkdir data
get_example_data.sh
```

##### Fit MILR model
```python
from milr import MILR
from mil import make_dataset_from_dataframe

# Initialize the model
model = MILR()

# Prepare your data
raw_data = loadarff('./data/musk2/musk.arff')
data = pd.DataFrame(raw_data[0])
data.rename(columns={'class': 'target'}, inplace=True)
data.target = data.target.astype(int)

feature_names = [f'f{i}' for i in range(1, 167)]
X, y, bags = make_dataset_from_dataframe(data, feature_names, 'target', 'molecule_name')

# Train the model
model.fit(X, y, bags, epochs=100, lr=1e-2, bag_fn='max')```
