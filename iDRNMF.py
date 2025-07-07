import torch
import numpy as np

def iDRNMF(X, r, e21, e22, e2cauchy, iteration=300, gamma=1, eps=1e-6):
    
    """
    Args:
        X (torch.Tensor): Input data matrix of shape (d, n).
        r (int): Number of clusters (rank).
        e21 (float): Normalization constant for 2,1 loss.
        e22 (float): Normalization constant for 2,2 (Frobenious) loss.
        e2cauchy (float): Normalization constant for 2,cauchy loss.
        iteration (int): Number of iterations (default 300).
        gamma (float): Parameter for Cauchy loss (default 1).
        eps (float): Small epsilon to avoid division by zero (default 1e-6).

    Returns:
        W (torch.Tensor): Basis matrix of shape (d, r).
        H (torch.Tensor): Coefficient matrix of shape (r, n).
    """

    d, n = X.shape

    # Initialize W and H randomly
    W = torch.rand(d, r)
    H = torch.rand(r, n)

    lossFunctions = ["2,1", "2,2", "2,cauchy"]
    lossFuncWeight = [1, 1, 1]
    eNormalization = [e21, e22, e2cauchy]

    lossFuncCount = len(lossFuncWeight)
    nonZeroInd = np.nonzero(lossFuncWeight)[0]
    lossCount = len(nonZeroInd)

    lam = [lossFuncWeight[i] / lossCount for i in range(lossFuncCount)]

    for i in range(iteration):

        # Compute residual norms per column
        e = torch.norm(X - W @ H, dim=0)

        d21 = 1 / torch.maximum(e, torch.tensor(eps))
        d22 = torch.ones(n)
        d2cauchy = 1 / torch.maximum(e**2 + gamma**2, torch.tensor(eps))
        dAll = [d21, d22, d2cauchy]

        dFinal = torch.zeros(n)

        for j in range(lossFuncCount):
            dFinal += (lossFuncWeight[j] * lam[j] / eNormalization[j]) * dAll[j]

        D = torch.diag(dFinal)

        # Update W
        Wu = X @ D @ H.T
        Wd = W @ H @ D @ H.T
        W = W * (Wu / torch.maximum(Wd, torch.tensor(eps)))

        # Update H
        Hu = W.T @ X @ D
        Hd = W.T @ W @ H @ D
        H = H * (Hu / torch.maximum(Hd, torch.tensor(eps)))

        # Compute each loss component
        e21_val = lossFuncWeight[0] * (torch.sum(torch.norm(X - W @ H, dim=0))) / eNormalization[0]
        e22_val = lossFuncWeight[1] * (torch.norm(X - W @ H)**2) / eNormalization[1]
        e2cauchy_val = lossFuncWeight[2] * (torch.sum(torch.log(torch.norm(X - W @ H, dim=0)**2 + gamma**2))) / eNormalization[2]

        eFinal = [e21_val.item(), e22_val.item(), e2cauchy_val.item()]

        omega = 1 / (i + 2)
        lamStar = [0] * lossFuncCount
        maxIndex = eFinal.index(max(eFinal))
        lamStar[maxIndex] = 1

        for j in range(lossFuncCount):
            lam[j] = (1 - omega) * lam[j] + omega * lamStar[j]

    return W, H
