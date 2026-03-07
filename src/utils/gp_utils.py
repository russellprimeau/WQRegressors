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


class UncertainInputMatern52Kernel(gpytorch.kernels.Kernel):
    """
    Monte Carlo approximation of a Matérn-5/2 kernel expectation under additive
    input uncertainty.

    This approximates:
        E[k_Matern52((x1 + e1), (x2 + e2))]
    where the uncertainty samples are provided as precomputed delta draws
    (e1 - e2) per feature, or are generated from Gaussian moments as fallback.
    """

    has_lengthscale = True

    def __init__(
        self,
        input_variance,
        noise_delta_samples=None,
        mc_samples=64,
        mc_seed=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_buffer("input_variance", input_variance)

        if noise_delta_samples is not None:
            noise_delta_samples = torch.as_tensor(noise_delta_samples, dtype=torch.float32)
            if noise_delta_samples.ndim != 2:
                raise ValueError(
                    f"noise_delta_samples must be 2-D [n_mc, n_features], got {tuple(noise_delta_samples.shape)}"
                )
            self.register_buffer("noise_delta_samples", noise_delta_samples)
            self.mc_samples = int(noise_delta_samples.shape[0])
        else:
            self.register_buffer("noise_delta_samples", None)
            self.mc_samples = max(1, int(mc_samples))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(mc_seed))
            unit_noise = torch.randn(
                (self.mc_samples, int(input_variance.shape[-1])),
                generator=generator,
                dtype=torch.float32,
            )
            self.register_buffer("_unit_noise", unit_noise)

    def _delta_samples(self, device, dtype):
        if self.noise_delta_samples is not None:
            return self.noise_delta_samples.to(device=device, dtype=dtype)
        scale = torch.sqrt(torch.clamp(2.0 * self.input_variance, min=0.0)).to(device=device, dtype=dtype)
        return self._unit_noise.to(device=device, dtype=dtype) * scale

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[-2], device=x1.device, dtype=x1.dtype)

        lengthscale = self.lengthscale.squeeze()
        if lengthscale.dim() == 0:
            lengthscale = lengthscale.repeat(x1.shape[-1])
        lengthscale = torch.clamp(lengthscale, min=1e-10)

        deltas = self._delta_samples(device=x1.device, dtype=x1.dtype)
        base_diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        diff = base_diff.unsqueeze(0) + deltas[:, None, None, :]

        scaled = diff / lengthscale
        r2 = torch.sum(scaled.pow(2), dim=-1)
        r = torch.sqrt(torch.clamp(r2, min=1e-12))

        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device, dtype=x1.dtype))
        k = (1.0 + sqrt5 * r + (5.0 / 3.0) * r2) * torch.exp(-sqrt5 * r)
        return torch.mean(k, dim=0)


def build_base_kernel(
    kernel_name,
    use_uncertain_kernel,
    input_uncertainty_var,
    ard_dims,
    uncertainty_noise_deltas=None,
    uncertain_kernel_mc_samples=64,
    uncertain_kernel_mc_seed=0,
):
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
        if kernel_name == "rbf":
            return UncertainInputRBFKernel(input_variance=input_uncertainty_var, ard_num_dims=ard_dims)
        if kernel_name == "matern52":
            return UncertainInputMatern52Kernel(
                input_variance=input_uncertainty_var,
                noise_delta_samples=uncertainty_noise_deltas,
                mc_samples=uncertain_kernel_mc_samples,
                mc_seed=uncertain_kernel_mc_seed,
                ard_num_dims=ard_dims,
            )
        raise ValueError(
            f"Unsupported uncertain-input kernel '{kernel_name}'. "
            "Use 'rbf' or 'matern52' when use_uncertain_input_kernel=True."
        )
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
