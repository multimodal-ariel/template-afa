"""Standard MLP model, for acquisition, acquires based on a fixed order.
We can also use this model for evaluation with constant predictive model.
"""

import torch.nn.functional as F

from .base import BaseModel
from .standard_layers import MLP


class FixedMLP(BaseModel):
    """Simple MLP model. This is the standard model for the acquisition function.
    Acquires based on a fixed global ordering. We can also use this model
    for evaluation with constant prediction model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_fixed_order = True
        self.predictor = MLP(
            in_dim=self.in_dim,
            hidden_dim=config["hidden_dim"],
            out_dim=self.out_dim,
            num_hidden=config["num_hidden"],
        )

    def forward(self, x, mask):
        return self.predictor(self.input_layer(x, mask))

    def predict(self, x, mask):
        # Predict has to give the distribution, not logits.
        return F.softmax(self.forward(x, mask), dim=-1)

    def loss_func(self, x, y, mask, data_mask=None):
        return F.cross_entropy(self.forward(x, mask), y)
