import torch
import numpy as np

# Should use double precision for numerical computations
torch.set_default_dtype(torch.float64)

class NNLS(torch.autograd.Function):
    """
    A FISTA implementation of non-negative least squares.
    Solves min ||Ax - b|| s.t. x >= 0
            x
    """
    @staticmethod
    def forward(ctx, A, b, max_iter=100, tol=1e-6):
        # Ensure shapes
        if b.dim() == 1:
            b = b.unsqueeze(1)
        A_d, b_d = A.double(), b.double()
        m, n = A_d.shape

        # Precompute
        L = torch.linalg.norm(A_d, 2) ** 2
        invL = 1.0 / L
        ATb = A_d.T @ b_d

        # Initialize
        x = torch.zeros((n, b_d.shape[1]), device=A.device, dtype=A_d.dtype)
        y = x.clone()
        t = 1.0

        # FISTA loop
        for _ in range(max_iter):
            grad = A_d.T @ (A_d @ y - b_d)
            x_next = (y - invL * grad).clamp(min=0.0)
            if torch.norm(x_next - x) < tol:
                x = x_next
                break
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = x_next + ((t - 1) / t_next) * (x_next - x)
            x, t = x_next, t_next

        # Save for backward
        ctx.save_for_backward(A_d, b_d, x)
        ctx.invL = invL
        ctx.L = L
        return x.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        A_d, b_d, x = ctx.saved_tensors
        invL = ctx.invL
        m, n = A_d.shape
        
        # given A, b, x, upstream grad grad_output = dL/dx:
        I = (x.squeeze(-1) > 0)
        A_I = A_d[:, I]
        G   = A_I.T @ A_I
        H   = torch.linalg.solve(G, grad_output[I])  # H = G^{-1} * dL_dx[I]
        v   = A_I @ x[I]                             
        r   = b_d - v

        # gradient w.r.t b
        grad_b = A_I @ H
        
        # gradient w.r.t A
        grad_A = torch.zeros_like(A_d)
        grad_A[:, I] = r @ H.T - A_I @ H - A_I @ H @ x[I].T
        
        return grad_A, grad_b, None, None
    


# Convenience wrapper
def nnls(A, b, **kwargs):
    return NNLS.apply(A, b, kwargs.get('max_iter', 100), kwargs.get('tol', 1e-6))