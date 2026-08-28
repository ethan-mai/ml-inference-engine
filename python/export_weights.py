import os
import numpy as np
import torch

tensors = torch.load('models/mlp_model.pt', weights_only=True, map_location='cpu')

os.makedirs('weights', exist_ok=True)

for key, tensor in tensors.items():
    np.save(f'weights/{key}', tensor.detach().numpy())