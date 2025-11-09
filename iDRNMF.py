import os
import warnings
import numpy as np
import torch
import scipy.io

# ---------------------------
# Environment and constants
# ---------------------------
warnings.filterwarnings("ignore")
EPS = 1e-10  # numerical stability
DEVICE = torch.device("cpu")  # or "cuda" if GPU available


def set_seed(seed=0):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------
# NMF primitives
# ---------------------------
def multiplicative_update_W_H(X, W, H, D_diag):
    
    D_row = D_diag.unsqueeze(0)  
    
    X_D = X * D_row
    H_D = H * D_row

    # Update W
    W *= (X_D @ H.t()) / (W @ (H @ H_D.t()) + EPS)

    # Update H
    WT = W.t()
    H *= (WT @ X_D) / ((WT @ W) @ (H * D_row) + EPS)

    W.clamp_(min=0.0)
    H.clamp_(min=0.0)
    return W, H


def compute_instance_residuals(X, W, H):
    return torch.norm(X - W @ H, dim=0)



def pure_nmf_2_2(X, W, H, max_iter=100):
    for _ in range(max_iter):
        W *= (X @ H.t()) / (W @ (H @ H.t()) + EPS)
        H *= (W.t() @ X) / ((W.t() @ W) @ H + EPS)
    return W, H, torch.norm(X - W @ H) ** 2


def pure_nmf_2_1(X, W, H, max_iter=100):
    for _ in range(max_iter):
        e = compute_instance_residuals(X, W, H)
        D = 1.0 / torch.maximum(e, torch.tensor(EPS, device=DEVICE))
        W, H = multiplicative_update_W_H(X, W, H, D)
    return W, H, torch.sum(torch.norm(X - W @ H, dim=0))


def pure_nmf_cauchy(X, W, H, gamma=1.0, max_iter=100):
    for _ in range(max_iter):
        e = compute_instance_residuals(X, W, H)
        D = 1.0 / (e ** 2 + gamma ** 2)
        W, H = multiplicative_update_W_H(X, W, H, D)
    loss = torch.sum(torch.log(torch.norm(X - W @ H, dim=0) ** 2 + gamma ** 2))
    return W, H, loss


# ---------------------------
# Main iDRNMF algorithm
# ---------------------------
def idrnmf(X, n_components,
           max_iter=300, gamma=1.0,
           eta0=0.9, eta_final=0.01,
           compute_zeta_iter=100,
           init_seed=0,
           normalize_columns=True,
           early_stop_tol=1e-6,
           verbose=True):
    
    set_seed(init_seed)
    m, n = X.shape

    # Optional column normalization
    if normalize_columns:
        X = X / (torch.norm(X, dim=0, keepdim=True) + EPS)

    # Initialization
    W = torch.rand(m, n_components, device=DEVICE)
    H = torch.rand(n_components, n, device=DEVICE)

    # Compute normalization constants (ζτ)
    if verbose:
        print("Computing normalization constants (ζ)...")
    _, _, z21 = pure_nmf_2_1(X, W.clone(), H.clone(), compute_zeta_iter)
    _, _, z22 = pure_nmf_2_2(X, W.clone(), H.clone(), compute_zeta_iter)
    _, _, zcau = pure_nmf_cauchy(X, W.clone(), H.clone(), gamma, compute_zeta_iter)
    z21, z22, zcau = [float(z.item()) for z in [z21, z22, zcau]]

    if verbose:
        print(f"ζ21={z21:.3e}, ζ22={z22:.3e}, ζcau={zcau:.3e}")

    # Initialize parameters
    lam = torch.ones(3, device=DEVICE) / 3.0
    eps21, eps22, epsc = 1 / z21, 1 / z22, 1 / zcau

    def eta(t): return eta0 + (eta_final - eta0) * (t / max_iter)

    prev_obj = None
    for t in range(max_iter):
        e = compute_instance_residuals(X, W, H)
        d1 = 1 / (e + EPS)
        d2 = torch.ones_like(e)
        dc = 1 / (e ** 2 + gamma ** 2)
        D = lam[0] * eps21 * d1 + lam[1] * eps22 * d2 + lam[2] * epsc * dc
        W, H = multiplicative_update_W_H(X, W, H, D)

        # Compute loss components
        e = compute_instance_residuals(X, W, H)
        L1 = torch.sum(e).item() * eps21
        L2 = torch.norm(X - W @ H).pow(2).item() * eps22
        Lc = torch.sum(torch.log(e ** 2 + gamma ** 2)).item() * epsc
        losses = np.array([L1, L2, Lc])
        obj = float(np.sum(lam.cpu().numpy() * losses))

        if prev_obj and abs(prev_obj - obj) < early_stop_tol:
            if verbose:
                print(f"Converged at iter {t+1}, Δobj={abs(prev_obj - obj):.2e}")
            break
        prev_obj = obj

        # Update λ 
        j = np.argmax(losses)
        lam_star = torch.zeros_like(lam)
        lam_star[j] = 1.0
        lam = (1 - eta(t)) * lam + eta(t) * lam_star
        lam /= lam.sum()

        if verbose and (t % 10 == 0 or t == max_iter - 1):
            print(f"Iter {t+1:03d} | Obj={obj:.3e} | λ={lam.cpu().numpy()}")

    if verbose:
        print("Optimization finished.\n")
    return W, H


# ---------------------------
# Script entry
# ---------------------------
if __name__ == "__main__":
    # === Load dataset ===
    mat_path = "D://MNIST.mat"  # ← EDIT your dataset path
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Dataset not found: {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    X_np = mat["X"]
    y_np = mat["y"].flatten().astype(int)

    # Ensure X has shape (features × samples)
    if X_np.shape[0] == len(y_np):
        X_np = X_np.T
    elif X_np.shape[0] < X_np.shape[1]:
        X_np = X_np.T

    print(f"Loaded X shape: {X_np.shape} (features × samples)")

    # Convert to torch tensor
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)

    # === Run iDRNMF ===
    r = len(np.unique(y_np))
    W, H = idrnmf(
        X,
        n_components=r,
        max_iter=300,
        gamma=1.0,
        eta0=0.9,
        eta_final=0.05,
        compute_zeta_iter=100,
        init_seed=0,
        normalize_columns=True,
        early_stop_tol=1e-7,
        verbose=True
    )

    # Optional: save factorization
    # torch.save(W, "W_idrnmf.pt")
    # torch.save(H, "H_idrnmf.pt")
