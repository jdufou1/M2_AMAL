import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


nb_data = 100

# Les données supervisées
x = torch.randn(nb_data, 13, dtype=torch.float64)
y = torch.randn(nb_data, 3, dtype=torch.float64)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, dtype=torch.float64)
b = torch.randn(3, dtype=torch.float64)

learning_rate = 0.005

writer = SummaryWriter()
for n_iter in range(1000):
    ## TODO:  Calcul du forward (loss)
    ctx_lin = Context()
    yhat = Linear.forward(ctx_lin,x,w,b)
    
    ctx_mse = Context()
    loss = MSE.forward(ctx_mse,yhat,y)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss.mean(), n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss.mean()}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    
    yhat_back, _ = MSE.backward(ctx_mse,loss)
    _,grad_w,grad_b = Linear.backward(ctx_lin,yhat_back.double())
    ##  TODO:  Mise à jour des paramètres du modèle
    w -= learning_rate * grad_w / nb_data
    b -= learning_rate * grad_b / nb_data