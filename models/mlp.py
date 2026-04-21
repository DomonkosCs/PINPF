import torch
import torch.nn as nn

from utils import DEFAULT_FEATURES, compute_input_dim


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        neurons,
        activation,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, neurons))
        self.layers.append(activation)
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(neurons, neurons))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralFlowModel(nn.Module):
    def __init__(
        self,
        state_dim=2,
        meas_dim=1,
        layers=6,
        neurons_per_layer=64,
        input_dim=None,
        activation=nn.SiLU(),
        extra_features: frozenset = None,
    ):
        if extra_features is None:
            extra_features = DEFAULT_FEATURES
        if input_dim is None:
            input_dim = compute_input_dim(state_dim, meas_dim, extra_features)
        super().__init__()
        self.extra_features = extra_features
        self.f_net = MLP(
            input_dim=input_dim,
            output_dim=state_dim,
            hidden_layers=layers,
            neurons=neurons_per_layer,
            activation=activation,
        )

    def forward(self, inputs):
        return self.f_net(inputs)


def load_neural_flow_model(model, ckp_path, device=torch.device("cpu")):
    model.to(device)
    ckp = torch.load(ckp_path, map_location=device)
    # Support both new checkpoint format (dict with model_state_dict)
    # and legacy format (raw state_dict)
    if isinstance(ckp, dict) and "model_state_dict" in ckp:
        model.load_state_dict(ckp["model_state_dict"])
    else:
        model.load_state_dict(ckp)
    model.eval()
    return model
