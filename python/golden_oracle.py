import mlp_model

import os
import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2

# retrieves a sample input (with added batch dimension) from test data
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
test_data = datasets.MNIST(root='data', train=False, download=False, transform=transform)
sample_input, _ = test_data[0]
sample_input = sample_input.unsqueeze(0)


# creating a new model instance and loading in trained params, on eval mode
model = mlp_model.MLPClassifier()
model.load_state_dict(torch.load('models/mlp_model.pt', weights_only=True))
model.eval()

# captures numeric output tensors for each layer, not just logit outputs
activations = {}

def get_activations(layer):
    def hook(module, input, output):
        activations[layer] = output.detach().clone()

    return hook

# registering a hook that runs after each layer to capture tensors
for i, layer in enumerate(model.stack):
    layer.register_forward_hook(get_activations(f"stack.{i}"))


output = model(sample_input)

os.makedirs('golden', exist_ok=True)

np.save('golden/golden_input', sample_input)

for name, tensor in activations.items():
    np.save(f'golden/{name}', tensor)
