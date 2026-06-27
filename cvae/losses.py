import torch


def mae_loss(input, output):
    return torch.mean(torch.mean(torch.abs(input - output), dim=(1, 2, 3)))


def KL_loss(mu, log_var):
    return torch.mean(0.5 * torch.mean(-1 - log_var + (torch.square(mu) + torch.exp(log_var)), axis=-1))


def KL_loss_double_gaussian(mu1, log_var1, mu2):
    return torch.mean(0.5 * torch.mean(-1 - log_var1 + (torch.square(mu1 - mu2) + torch.exp(log_var1)), axis=-1))


def vade_loss(gamma, ZLogVar, ZMean, gamma_layer, batch_size):
    theta3, u3, lam3 = gamma_layer.get_tensors(batch_size)

    gamma_t1 = gamma.unsqueeze(1).repeat(1, theta3.shape[1], 1)

    a = 0.5 * torch.sum(gamma_t1 * torch.sum(torch.log(lam3) +
                                             torch.exp(ZLogVar) / lam3 +
                                             torch.square(ZMean - u3) / lam3, axis=2), axis=(1))
    d = torch.sum(torch.log(torch.mean(theta3, axis=2) / gamma_t1) * gamma_t1, axis=(1))

    return torch.mean(a - d)


def contractive_loss(Z, X, device=torch.device('cuda')):
    Z.backward(torch.ones(Z.size()).to(device), retain_graph=True)
    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.mean(torch.sqrt(torch.sum(torch.square(X.grad), axis=(1, 2, 3))))
    X.grad.data.zero_()
    return loss2


def transf_invariant_loss(X, mu, encoder):
    rotloss = torch.zeros((X.shape[0], 3), device=X.device)

    for k in range(3):
        XR = torch.rot90(X, dims=(-2, -1), k=k + 1)
        mur, sigr, Zr = encoder(XR)

        lossi = KL_loss_double_gaussian(mur, sigr, mu)
        rotloss[:, k] = lossi

    Linv = torch.mean(torch.sum(rotloss, dim=1))
    Lres = torch.mean(torch.min(rotloss, dim=1)[0])

    return Linv, Lres


classification_loss = torch.nn.BCEWithLogitsLoss()
