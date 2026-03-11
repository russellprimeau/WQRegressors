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


def _parse_kernel_spec(kernel_name: str) -> tuple[str | None, bool]:
    name = str(kernel_name or "").lower().strip()
    if not name:
        name = "matern52"
    parts = [part.strip() for part in name.split("+") if part.strip()]
    has_linear = "linear" in parts
    base_parts = [part for part in parts if part != "linear"]
    if len(base_parts) > 1:
        raise ValueError(
            f"Unsupported kernel combination '{kernel_name}'. "
            "At most one base kernel is allowed with an optional '+linear' term."
        )
    base = base_parts[0] if base_parts else None
    return base, has_linear


def describe_effective_kernel(kernel_name: str, use_uncertain_kernel: bool) -> str:
    base, has_linear = _parse_kernel_spec(kernel_name)
    if base is None and not has_linear:
        base = "matern52"

    if use_uncertain_kernel:
        if base is None:
            raise ValueError(
                "Uncertain-input kernel requires a base kernel (rbf or matern52)."
            )
        if base not in {"rbf", "matern52"}:
            raise ValueError(
                f"Unsupported uncertain-input kernel '{base}'. "
                "Use 'rbf' or 'matern52' when use_uncertain_input_kernel=True."
            )
        effective_base = f"uncertain_{base}"
    else:
        effective_base = base or "linear"

    if has_linear and effective_base != "linear":
        return f"{effective_base}+linear"
    return effective_base


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
        One of ``"rbf"``, ``"matern32"``, ``"matern52"`` (default for anything else),
        optionally suffixed with ``+linear`` (e.g., ``"matern52+linear"``).
    use_uncertain_kernel : bool
        If True, uses uncertain-input kernels (rbf or matern52 only).
    input_uncertainty_var : torch.Tensor or None
        Per-feature input variance tensor; required when ``use_uncertain_kernel`` is True.
    ard_dims : int or None
        Number of ARD lengthscale dimensions, or None for isotropic.
    """
    base_name, has_linear = _parse_kernel_spec(kernel_name)
    if base_name is None and not has_linear:
        base_name = "matern52"
    elif base_name is None and has_linear:
        base_name = "linear"

    if use_uncertain_kernel:
        if base_name is None:
            raise ValueError(
                "Uncertain-input kernel requires a base kernel (rbf or matern52)."
            )
        if base_name == "rbf":
            base_kernel = UncertainInputRBFKernel(input_variance=input_uncertainty_var, ard_num_dims=ard_dims)
        elif base_name == "matern52":
            base_kernel = UncertainInputMatern52Kernel(
                input_variance=input_uncertainty_var,
                noise_delta_samples=uncertainty_noise_deltas,
                mc_samples=uncertain_kernel_mc_samples,
                mc_seed=uncertain_kernel_mc_seed,
                ard_num_dims=ard_dims,
            )
        else:
            raise ValueError(
                f"Unsupported uncertain-input kernel '{base_name}'. "
                "Use 'rbf' or 'matern52' when use_uncertain_input_kernel=True."
            )
    else:
        if base_name == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif base_name == "matern32":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        elif base_name == "matern52" or base_name is None:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        elif base_name == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_dims)
        else:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)

    if has_linear and base_name != "linear":
        linear_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_dims)
        return base_kernel + linear_kernel
    return base_kernel


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _build_constraint(min_value, max_value):
    if min_value is None and max_value is None:
        return None
    if min_value is not None and max_value is not None:
        if max_value <= min_value:
            raise ValueError(
                f"Constraint max ({max_value}) must be > min ({min_value})."
            )
        return gpytorch.constraints.Interval(min_value, max_value)
    if min_value is not None:
        return gpytorch.constraints.GreaterThan(min_value)
    return gpytorch.constraints.LessThan(max_value)


def _iter_kernels(module):
    seen = set()
    for child in module.modules():
        if isinstance(child, gpytorch.kernels.Kernel):
            if id(child) not in seen:
                seen.add(id(child))
                yield child


def _parse_gamma_prior(prior_cfg):
    if not prior_cfg:
        return None
    if isinstance(prior_cfg, gpytorch.priors.Prior):
        return prior_cfg
    if not isinstance(prior_cfg, dict):
        return None
    prior_type = str(prior_cfg.get("type", "gamma")).lower().strip()
    if prior_type != "gamma":
        raise ValueError(f"Unsupported prior type '{prior_type}'.")
    shape = prior_cfg.get("shape", prior_cfg.get("alpha", None))
    rate = prior_cfg.get("rate", prior_cfg.get("beta", None))
    shape = _coerce_float(shape)
    rate = _coerce_float(rate)
    if shape is None or rate is None:
        return None
    if shape <= 0 or rate <= 0:
        raise ValueError("GammaPrior shape and rate must be > 0.")
    return gpytorch.priors.GammaPrior(shape, rate)


def apply_gp_constraints_and_priors(model, likelihood, hyper_cfg: dict) -> dict:
    """Apply optional constraints/priors from hyperparameters to GP model and likelihood."""
    meta = {
        "lengthscale_constraint": None,
        "outputscale_constraint": None,
        "noise_constraint": None,
        "outputscale_prior": None,
        "noise_prior": None,
    }

    ls_min = _coerce_float(hyper_cfg.get("lengthscale_min"))
    ls_max = _coerce_float(hyper_cfg.get("lengthscale_max"))
    ls_constraint = _build_constraint(ls_min, ls_max)
    if ls_constraint is not None:
        applied = 0
        for kernel in _iter_kernels(model.covar_module.base_kernel):
            if hasattr(kernel, "raw_lengthscale"):
                kernel.register_constraint("raw_lengthscale", ls_constraint)
                applied += 1
        if applied:
            meta["lengthscale_constraint"] = {"min": ls_min, "max": ls_max, "kernels": applied}

    os_min = _coerce_float(hyper_cfg.get("outputscale_min"))
    os_max = _coerce_float(hyper_cfg.get("outputscale_max"))
    os_constraint = _build_constraint(os_min, os_max)
    if os_constraint is not None and hasattr(model.covar_module, "register_constraint"):
        model.covar_module.register_constraint("raw_outputscale", os_constraint)
        meta["outputscale_constraint"] = {"min": os_min, "max": os_max}

    noise_min = _coerce_float(hyper_cfg.get("noise_min"))
    noise_max = _coerce_float(hyper_cfg.get("noise_max"))
    noise_constraint = _build_constraint(noise_min, noise_max)
    if noise_constraint is not None and hasattr(likelihood, "register_constraint"):
        likelihood.register_constraint("raw_noise", noise_constraint)
        meta["noise_constraint"] = {"min": noise_min, "max": noise_max}

    outputscale_prior = _parse_gamma_prior(hyper_cfg.get("outputscale_prior"))
    if outputscale_prior is not None:
        model.covar_module.register_prior("outputscale_prior", outputscale_prior, "outputscale")
        meta["outputscale_prior"] = "gamma"

    noise_prior = _parse_gamma_prior(hyper_cfg.get("noise_prior"))
    if noise_prior is not None:
        likelihood.register_prior("noise_prior", noise_prior, "noise")
        meta["noise_prior"] = "gamma"

    return meta


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
