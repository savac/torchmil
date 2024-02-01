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

    def forward(self, X, bags, padding_mask, bag_fn):
        instance_logits = self.linear(X)

        if bag_fn == 'max':
            bags_on_rows = instance_logits[bags].squeeze()
            bags_on_rows.masked_fill_(padding_mask, -np.inf)
            bags_max = bags_on_rows.amax(dim=1, keepdim=True)
            bags_logits = torch.cat([
                torch.zeros(bags_max.shape[0], 1),
                bags_max
            ], dim=1)
            bags_log_probs = F.log_softmax(bags_logits, dim=1)
        else:
            raise NotImplementedError

        return bags_log_probs

    def fit(self, X: np.ndarray, y: np.ndarray, bags: list[np.ndarray], epochs=10, optimizer=None, lr=1e-4, bag_fn='max'):
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y)
        padding_value = X.shape[0]
        bags = [torch.as_tensor(bag) for bag in bags]
        bags = torch.nn.utils.rnn.pad_sequence(bags, batch_first=True, padding_value=padding_value)
        padding_mask = bags == padding_value
        # now that we have a mask we can change padding to a valid index
        bags.masked_fill_(padding_mask, 0)

        self.linear = torch.nn.Linear(X.shape[1], 1)

        loss_function = nn.NLLLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        self.losses = []
        for _ in range(epochs):
            # NB: PyTorch accumulates gradients.
            self.zero_grad()
            log_probs = self(X, bags, padding_mask, bag_fn)
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...