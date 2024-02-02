import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

torch.manual_seed(1)


class MILR(nn.Module):
    """MILR: model for multi-instance logistic regression."""

    def __init__(self):
        super().__init__()

        self.linear = None
        self.bag_fn = None

    def forward(self, X, bags, padding_mask, bag_fn):
        self.bag_fn = bag_fn

        instance_logits = self.linear(X)

        if bag_fn == "max":
            bags_on_rows = instance_logits[bags].squeeze()
            bags_on_rows.masked_fill_(padding_mask, -np.inf)
            bags_max = bags_on_rows.amax(dim=1, keepdim=True)
            bags_logits = torch.cat(
                [torch.zeros(bags_max.shape[0], 1), bags_max], dim=1
            )
            bags_log_probs = F.log_softmax(bags_logits, dim=1)
        else:
            raise NotImplementedError

        return bags_log_probs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bags: list[np.ndarray],
        epochs=10,
        optimizer=None,
        lr=1e-4,
        bag_fn="max",
    ):
        X, y, bags = self._cast_inputs(X, y, bags)

        padding_value = X.shape[0]

        bags, padding_mask = self._process_bags(bags, padding_value)

        self.linear = torch.nn.Linear(X.shape[1], 1)

        loss_function = nn.NLLLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        self.metrics = []
        for _ in range(epochs):
            # NB: PyTorch accumulates gradients.
            self.zero_grad()
            log_probs = self(X, bags, padding_mask, bag_fn)
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()

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
        return torch.argmax(probs, axis=1).numpy()

    def predict_proba(self, X, bags) -> np.ndarray:
        """Predicts the bag class probabilities.

        Args:
            X (np.ndarray): features
            bags (list[np.ndarray]): bag indices

        Returns:
            np.ndarray: bag class probabilities
        """
        X, _, bags = self._cast_inputs(X, None, bags)
        with torch.no_grad():
            log_probs = self.forward(X, bags, self.bag_fn)
            return torch.exp(log_probs).numpy()

    def _cast_inputs(self, X, y, bags):
        X = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            y = torch.as_tensor(y, dtype=torch.long)
        bags = [torch.as_tensor(bag, dtype=torch.long) for bag in bags]
        return X, y, bags

    def _process_bags(self, bags, padding_value):
        bags = torch.nn.utils.rnn.pad_sequence(
            bags, batch_first=True, padding_value=padding_value
        )
        padding_mask = bags == padding_value
        # padding value needs to be a valid index
        bags.masked_fill_(padding_mask, 0)
        return bags, padding_mask
