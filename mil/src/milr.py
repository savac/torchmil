import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .utils import invlogit, logsumexp_safe, generalized_mean

torch.manual_seed(1)


class MILR(nn.Module):
    """MILR: model for multi-instance logistic regression."""

    def __init__(self):
        super().__init__()

        self.linear = None
        self.bag_fn = None

    def _predict_instance(self, X):
        return invlogit(self.linear(X))

    def forward(self, X, bags, bags_mask, bag_fn):
        self.bag_fn = bag_fn

        instance_probs = self._predict_instance(X)
        bagged_instance_probs = instance_probs[bags].squeeze()

        if bag_fn == "max":
            bag_probs = bagged_instance_probs.amax(dim=1, keepdim=True)

        elif bag_fn == "logsumexp":
            bag_probs = logsumexp_safe(
                bagged_instance_probs, self.softmax_parameter, bags_mask
            )

        elif bag_fn == "generalized_mean":
            bag_probs = generalized_mean(
                bagged_instance_probs, self.softmax_parameter, bags_mask
            )

        else:
            raise NotImplementedError

        bag_probs = torch.cat([1 - bag_probs, bag_probs], dim=1)
        bag_log_probs = torch.log(bag_probs)

        return bag_log_probs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bags: list[np.ndarray],
        epochs=10,
        optimizer=None,
        lr=1e-4,
        bag_fn="max",
        softmax_parameter=None,
    ):
        X, y, bags = self._cast_inputs(X, y, bags)

        padding_value = X.shape[0]

        bags, bags_mask = self._process_bags(bags, padding_value)

        self.linear = torch.nn.Linear(X.shape[1], 1)

        if softmax_parameter is None:
            self.softmax_parameter = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.register_parameter("softmax_parameter", softmax_parameter)

        loss_function = nn.NLLLoss()

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        self.metrics = []
        for _ in range(epochs):
            self.zero_grad()
            log_probs = self(X, bags, bags_mask, bag_fn)
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()

            if loss.isnan():
                raise ValueError("Loss is NaN.")

            self.metrics.append(
                {
                    "loss": loss.item(),
                    "accuracy": (torch.argmax(log_probs, axis=1) == y)
                    .float()
                    .mean()
                    .item(),
                }
            )

    def predict(self, X: np.ndarray, bags: list[np.ndarray]) -> np.ndarray:
        """Predicts the class of the bags.

        Args:
            X (np.ndarray): features
            bags (list[np.ndarray]): bag indices

        Returns:
            np.ndarray: bag classes
        """
        probs = self.predict_proba(X, bags)
        return np.argmax(probs, axis=1)

    def predict_instance(self, X: np.ndarray) -> np.ndarray:
        """Predicts the class of the instances.

        Args:
            X (np.ndarray): features

        Returns:
            np.ndarray: instance classes
        """
        probs = self.predict_proba_instance(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray, bags: list[np.ndarray]) -> np.ndarray:
        """Predicts the bag class probabilities.

        Args:
            X (np.ndarray): features
            bags (list[np.ndarray]): bag indices

        Returns:
            np.ndarray: bag class probabilities
        """
        X, _, bags = self._cast_inputs(X, None, bags)
        bags, bags_mask = self._process_bags(bags, padding_value=X.shape[0])
        with torch.no_grad():
            log_probs = self(X, bags, bags_mask, self.bag_fn)
            return torch.exp(log_probs).numpy()

    def predict_proba_instance(self, X: np.ndarray) -> np.ndarray:
        """Predicts the instance class probabilities.

        Args:
            X (np.ndarray): features

        Returns:
            np.ndarray: instance class probabilities
        """
        X, _, _ = self._cast_inputs(X, None, None)
        with torch.no_grad():
            instance_probs = self._predict_instance(X)
            return torch.cat([1 - instance_probs, instance_probs], dim=1).numpy()

    def _cast_inputs(self, X, y, bags):
        X = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            y = torch.as_tensor(y, dtype=torch.long)
        if bags is not None:
            bags = [torch.as_tensor(bag, dtype=torch.long) for bag in bags]
        return X, y, bags

    def _process_bags(self, bags, padding_value):
        bags = torch.nn.utils.rnn.pad_sequence(
            bags, batch_first=True, padding_value=padding_value
        )
        padding_mask = bags == padding_value
        # padding value needs to be a valid index
        bags.masked_fill_(padding_mask, 0)
        return bags, ~padding_mask
