# torchmil

## Description

This repo contains the implementation of Multi-Instance Logistic Regression in PyTorch. 

The library offers several alternatives for definining the relationship between the instance probability, $p_{ij}$, and the
bag probability, $p_i$ as outlined in Section 2.4.1 here[^1].

| Name        | Relation           | Option name  |
| ------------- |:-------------:| -----:|
| Max           | $p(i) = max(p_{ij})$  | max |
| Logsumexp     |                       |   logsumexp |
| Generalized mean  |       | generalized_mean  |
| Product           |       | prduct            |
| Likelihood ration |       | likelihood_ratio  |


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
model.fit(X, y, bags, epochs=100, lr=1e-2, bag_fn='max')
```



[^1]: [Multiple Instance Learning: Algorithms and Applications](https://api.semanticscholar.org/CorpusID:2153770)
