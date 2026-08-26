import torch
import json

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data = torch.load("models/mlp_model.pt", map_location=device, weights_only=True)

# print(type(data))
# print(data.keys())

inputs = [{"name": "input_0", "shape": data.values()[0].shape, "dtype": data.values()[0].dtype}]
outputs = []
nodes = []
op_name = "Linear"




for key, tensor in data.items():


    
