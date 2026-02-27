"""
Shared GPyTorch model classes and kernel builders used by e_Train.py, f_Evaluate.py,
and crossval utilities. Extracted to eliminate three duplicate definitions.
"""

import gpytorch
import torch


class UncertainInputRBFKernel(gpytorch.kernels.Kernel):
    """
    Closed-form RBF kernel that marginalises over Gaussian input noise.

    The effective covariance between x1 and x2 integrates out additive
    isotropic Gaussian noise on each input dimension:

        k(x1, x2) = det(I + 2*Lambda^{-2}*Sigma)^{-1/2}
                    * exp(-0.5 * (x1-x2)^T (Lambda^2 + 2*Sigma)^{-1} (x1-x2))

    where Lambda is the ARD lengthscale diagonal and Sigma is the diagonal
    input-noise variance vector ``input_variance``.
    """

    has_lengthscale = True

    def __init__(self, input_variance, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("input_variance", input_variance)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[-2], device=x1.device, dtype=x1.dtype)

        lengthscale = self.lengthscale.squeeze()
        if lengthscale.dim() == 0:
            lengthscale = lengthscale.repeat(x1.shape[-1])

        ls2 = lengthscale.pow(2)
        denom = torch.clamp(ls2 + 2.0 * self.input_variance, min=1e-10)

        sq_dist = ((x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2) / denom).sum(dim=-1)
        det_term = torch.sqrt(torch.prod(ls2 / denom))
        return det_term * torch.exp(-0.5 * sq_dist)


def build_base_kernel(kernel_name, use_uncertain_kernel, input_uncertainty_var, ard_dims):
    """
    Construct the base GP covariance kernel based on configuration.

    Parameters
    ----------
    kernel_name : str
        One of ``"rbf"``, ``"matern32"``, ``"matern52"`` (default for anything else).
    use_uncertain_kernel : bool
        If True, returns an ``UncertainInputRBFKernel`` regardless of ``kernel_name``
        (with a warning if ``kernel_name != "rbf"``).
    input_uncertainty_var : torch.Tensor or None
        Per-feature input variance tensor; required when ``use_uncertain_kernel`` is True.
    ard_dims : int or None
        Number of ARD lengthscale dimensions, or None for isotropic.
    """
    if use_uncertain_kernel:
        if kernel_name != "rbf":
            print(
                f"[WARN] Uncertain-input kernel supports only the RBF closed form. "
                f"Overriding kernel '{kernel_name}' -> 'rbf'."
            )
        return UncertainInputRBFKernel(input_variance=input_uncertainty_var, ard_num_dims=ard_dims)
    if kernel_name == "rbf":
        return gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
    if kernel_name == "matern32":
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
    return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)


class ExactGPRegressor(gpytorch.models.ExactGP):
    """Exact GP regression model with a constant mean and a scaled covariance kernel."""

    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
