import torch
import torch.nn as nn


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
        omit_grad_features: bool = False,
    ):
        if input_dim is None:
            # Features: x, lam, z, [grad_log_p, grad_log_h], log_h
            if omit_grad_features:
                input_dim = state_dim + meas_dim + 1 + 1
            else:
                input_dim = 3 * state_dim + meas_dim + 1 + 1
        super().__init__()
        self.use_checkpointing = False
        self.omit_grad_features = omit_grad_features
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
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    return model
