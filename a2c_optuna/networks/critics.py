import numpy as np
import torch
from torch import distributions, nn, optim
from torch.nn import functional as F

from a2c_optuna.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )
        self.mse = nn.MSELoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)
        q_values = q_values.unsqueeze(1)

        # update the critic using the observations and q_values
        pred = self.network(obs)
        loss = self.mse(pred, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        return ptu.to_numpy(self.network(obs)).squeeze()
