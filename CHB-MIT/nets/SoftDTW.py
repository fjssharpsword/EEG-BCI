import torch
import pysdtw

#https://github.com/toinsson/pysdtw
if __name__ == "__main__":
    
    device = torch.device('cuda') #torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # the input data includes a batch dimension
    X = torch.rand((8, 512, 18), device=device, requires_grad=True)
    Y = torch.rand((8, 512, 18), device=device)

    # optionally choose a pairwise distance function
    fun = pysdtw.distance.pairwise_l2_squared

    # create the SoftDTW distance function
    sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=True)

    # soft-DTW discrepancy, approaches DTW as gamma -> 0
    res = sdtw(X, Y)

    # define a loss, which gradient can be backpropagated
    loss = res.sum()
    loss.backward()

    # X.grad now contains the gradient with respect to the loss