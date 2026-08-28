import torch
import numpy as np
from torch import nn
import json
import mlp_model


model = mlp_model.MLPClassifier()
model.load_state_dict(torch.load('models/mlp_model.pt', weights_only=True))

# retrieves the static parameters (weights+biases) from state_dict, not the activations
# like in golden_oracle.py (result of input and parameters)

data = np.load('golden/golden_input.npy')
graph_inputs = [{'name': 'golden_input', 'shape': list(data.shape), 'dtype': str(data.dtype)}]


nodes = []
current_input = 'golden_input'

for i, layer in enumerate(model.stack):
    if isinstance(layer, nn.Linear):
        op = 'Linear'
        inputs = [current_input, f"stack.{i}.weight", f"stack.{i}.bias"]
    else:
        op = 'ReLU'
        inputs = [current_input]

    current_input = f"stack.{i}"
    output = f"stack.{i}"
    node = {'op': op, 'inputs': inputs, 'output': output}
    nodes.append(node)

outputs = [nodes[-1]['output']]


model_dump = {'graph_inputs': graph_inputs, 'nodes': nodes, 'outputs': outputs}

with open('graph.json', 'w') as file:
    json.dump(model_dump, file, indent=2)





    
