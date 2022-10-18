import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(2,3, requires_grad=True, dtype=torch.float64)
y = torch.randn(2,3, requires_grad=True, dtype=torch.float64)

print("test mse")

torch.autograd.gradcheck(mse, (yhat, y))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

X = torch.randn(4,3, requires_grad=True, dtype=torch.float64)
W = torch.randn(3,2, requires_grad=True, dtype=torch.float64)
B = torch.randn(2, requires_grad=True, dtype=torch.float64)

print("test linear")

torch.autograd.gradcheck(linear, (X, W, B))