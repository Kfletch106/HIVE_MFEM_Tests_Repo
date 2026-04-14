# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:36:53 2026

@author: dr19382
"""
#Functions for performing forward models in standard SGVP
#This is not DKL

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.optim import NGD
from sklearn.cluster import KMeans
from scipy.stats.qmc import LatinHypercube
from linear_operator.operators import DiagLinearOperator
import matplotlib.pyplot as plt

# ── Standardisation / Destandardisation ──────────────────────────────────────
def TrainInputStandardisation(x_raw_data, y_raw_data, yLog=False):
    X_min = x_raw_data.min(axis=0)
    X_max = x_raw_data.max(axis=0)
    Xs = (x_raw_data - X_min) / (X_max - X_min + 1e-9)

    if not yLog:
        y_mean = y_raw_data.mean()
        y_std  = y_raw_data.std() + 1e-9
        Ys = (y_raw_data - y_mean) / y_std
        Ydestand = [y_mean, y_std]
    else:
        eps = torch.tensor(1e-9, dtype=torch.float32)
        y_t = torch.tensor(y_raw_data, dtype=torch.float32)
        Y_log = torch.log(y_t + eps)
        y_log_mean = Y_log.mean()
        y_log_std  = Y_log.std() + eps
        Ys = ((Y_log - y_log_mean) / y_log_std).numpy()
        Ydestand = [y_log_mean, y_log_std]

    return X_min, X_max, Ydestand, Xs, Ys


def TestRunStandardisation(x_raw_input, X_min, X_max):
    X_out = (x_raw_input - X_min) / (X_max - X_min + 1e-9)
    return X_out

#Map standardised GP predictions back to original output space.
def OutputDestandardisation(output_raw_means, output_raw_var, Ydestand, yLog=False):
    if not yLog:
        y_mean, y_std = Ydestand
        y_mean_pred = output_raw_means * y_std + y_mean
        y_var_pred  = output_raw_var * (y_std ** 2)
        y_std_pred  = torch.sqrt(y_var_pred)
    else:
        eps = torch.tensor(1e-9, dtype=torch.float32)
        y_log_mean, y_log_std = Ydestand
        log_pred    = output_raw_means * y_log_std + y_log_mean
        y_mean_pred = torch.exp(log_pred) - eps
        log_var     = output_raw_var * (y_log_std ** 2)
        y_var_pred  = (torch.exp(log_var) - 1) * torch.exp(2 * log_pred + log_var)
        y_std_pred  = torch.sqrt(y_var_pred)

    return (
        y_mean_pred.cpu().numpy(),
        y_var_pred.cpu().numpy(),
        y_std_pred.cpu().numpy(),
    )

#Downsample using regular bin grid, keeps coverage and proportionality, better then random
def stratified_downsample(coords, T, factor, n_bins=10):
    from scipy.stats import binned_statistic_dd

    xyz    = coords[:, :3]
    ranges = [(xyz[:, d].min(), xyz[:, d].max()) for d in range(3)]
    _, _, binnumber = binned_statistic_dd(
        xyz, np.zeros(len(xyz)), bins=n_bins, range=ranges
    )

    selected = []
    for b in np.unique(binnumber):
        idx = np.where(binnumber == b)[0]
        k   = max(1, len(idx) // factor)
        selected.append(np.random.choice(idx, size=k, replace=False))

    chosen = np.concatenate(selected)
    return coords[chosen], T[chosen]


# ── Inducing Point Initialisation ─────────────────────────────────────────────
#Initialise inducing points as a mix of random subset and K-means centroids.
def init_inducing_points(X, M=200, split=0.3):
    M_rand   = int(split * M)
    rand_idx = torch.randperm(X.size(0))[:M_rand]
    Z_random = X[rand_idx].cpu()

    X_np   = X.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=M - M_rand, n_init=10).fit(X_np)
    Z_kmeans = torch.tensor(kmeans.cluster_centers_, dtype=X.dtype)

    return torch.cat([Z_random, Z_kmeans], dim=0).to(X.device)


# ── Stratified Inducing Point Initialisation ──────────────────────────────────
#Inducing point selection but making use of spatial clusters to ensure better coverage
def init_inducing_points_spatial(X, M=400, coord_dims=3, global_dims=2):
    M_spatial = int(M * 0.7)
    M_global  = M - M_spatial

    X_spatial = X[:, :coord_dims].cpu().numpy()
    X_global  = X[:, coord_dims:].cpu().numpy()

    km_s = KMeans(n_clusters=M_spatial, n_init=10).fit(X_spatial)
    km_g = KMeans(n_clusters=M_global,  n_init=10).fit(X_global)

    # Pair each spatial centroid with a random global centroid
    g_idx    = np.random.choice(M_global, size=M_spatial)
    Z_paired = np.hstack([km_s.cluster_centers_, km_g.cluster_centers_[g_idx]])

    # Add a small set of purely global-varying points
    s_idx    = np.random.choice(M_spatial, size=M_global)
    Z_global = np.hstack([km_s.cluster_centers_[s_idx], km_g.cluster_centers_])

    Z = np.vstack([Z_paired, Z_global])
    return torch.tensor(Z, dtype=X.dtype, device=X.device)


# ── SVGP Model ────────────────────────────────────────────────────────────────
#Sparse Variational GP with an additive Matern kernel
class SingleTaskSVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        coord_dims = 3,
        global_dims = 2,
        init_lengthscale=0.2,
        init_outputscale=1.0,
        init_noise=1e-3,
        jitter_zz=1e-3,
        jitter_xx=1e-3,
        nu=2.5,
    ):
        torch.nn.Module.__init__(self)

        if inducing_points.dim() != 2:
            raise ValueError("Expected inducing_points shape (M, D).")

        M, D = inducing_points.shape
        self._jitter_xx = float(jitter_xx)
        self._Z_param   = torch.nn.Parameter(inducing_points.clone().detach())

        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=M
        )
        var_strat = gpytorch.variational.VariationalStrategy(
            self, self._Z_param, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)

        self.variational_strategy._jitter_val = float(jitter_zz)

        self.mean_module = gpytorch.means.ConstantMean()

        Active_Coords = list(range(0, (coord_dims), 1))
        Active_Global = list(range((coord_dims), (coord_dims+global_dims), 1))
        ls_constraint = gpytorch.constraints.Interval(1e-5, 10)
        self.spatial_kernel = gpytorch.kernels.MaternKernel(
            nu=nu, ard_num_dims=coord_dims, active_dims=Active_Coords,
            lengthscale_constraint=ls_constraint,
        )
        self.global_kernel = gpytorch.kernels.MaternKernel(
            nu=nu, ard_num_dims=global_dims, active_dims=Active_Global,
            lengthscale_constraint=ls_constraint,
        )

        self.additive_kernel = gpytorch.kernels.AdditiveKernel(self.spatial_kernel, self.global_kernel)
        
        self.product_kernel = gpytorch.kernels.ProductKernel(self.spatial_kernel, self.global_kernel)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(self.additive_kernel, self.product_kernel),
            outputscale_constraint=gpytorch.constraints.Interval(1e-4, 100),
        )

        with torch.no_grad():
            self.spatial_kernel.lengthscale = init_lengthscale
            self.global_kernel.lengthscale  = init_lengthscale
            self.covar_module.outputscale   = init_outputscale

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(1e-6, 10)
        )
        self.likelihood.noise = init_noise

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)

        # Stabilise K_xx with diagonal jitter
        diag    = torch.full((covar_x.size(-1),), self._jitter_xx,
                             device=covar_x.device, dtype=covar_x.dtype)
        covar_x = covar_x + DiagLinearOperator(diag)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── Training ──────────────────────────────────────────────────────────────────
# Train an SVGP with a combined NGD + Adam scheme, early stopping, and multi-restart
def TrainHybridSVGP(
    model,
    likelihood,
    train_x,
    train_y,
    ngd_opt,        
    adam_opt,
    *,
    X_val=None,
    y_val=None,
    iters=600,
    batch_size=2048,
    warmup_epochs=40,
    max_grad_norm=10.0,
    restarts=3,
    scheduler_step=250,
    scheduler_gamma=0.5,
    patience=30,
    improvement_tol=1e-5,
    device="cuda",
    verbose=True,
):
    
    device   = torch.device(device)
    train_x  = train_x.to(device)
    train_y  = train_y.to(device)
    N        = train_x.size(0)

    use_val = (X_val is not None) and (y_val is not None)
    if use_val:
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        monitor_label = "Val RMSE"
    else:
        monitor_label = "Train ELBO"

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    mll           = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)
    best_metric   = float("inf")
    best_state    = None
    loss_history  = []

    scheduler = torch.optim.lr_scheduler.StepLR(
        adam_opt, step_size=scheduler_step, gamma=scheduler_gamma
    )

    for r in range(restarts):
        if verbose:
            print(f"\n── Restart {r + 1}/{restarts} ──")

        model.train()
        likelihood.train()

        patience_count = 0

        for epoch in range(iters):
            requires_grad = epoch >= warmup_epochs
            model.variational_strategy.inducing_points.requires_grad_(requires_grad)

            running_loss = 0.0

            for xb, yb in loader:
                adam_opt.zero_grad()
                ngd_opt.zero_grad()

                out  = model(xb)
                loss = -mll(out, yb.squeeze(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                adam_opt.step()
                ngd_opt.step()

                running_loss += loss.item()

            scheduler.step()
            avg_loss = running_loss / len(loader)

            # ── Compute monitoring metric ─────────────────────────────────
            if use_val:
                model.eval(); likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    preds    = likelihood(model(X_val))
                    monitor  = torch.sqrt(
                        ((preds.mean - y_val.squeeze()) ** 2).mean()
                    ).item()
                model.train(); likelihood.train()
            else:
                monitor = avg_loss

            loss_history.append({
                "epoch":    epoch,
                "elbo":     avg_loss,
                "monitor":  monitor,
            })

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:>4d}: ELBO = {avg_loss:.5f}  |  "
                      f"{monitor_label} = {monitor:.5f}")

            # ── Early stopping ────────────────────────────────────────────
            if monitor < best_metric - improvement_tol:
                best_metric = monitor
                best_state  = {
                    "model":      {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "likelihood": {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
                }
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    if verbose:
                        print(f"  → Early stopping at epoch {epoch} "
                              f"(best {monitor_label} = {best_metric:.5f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.eval()
    likelihood.eval()

    return {"best_metric": best_metric, "best_state": best_state,
            "loss_history": loss_history}


# ── Prediction ────────────────────────────────────────────────────────────────
#Run a forward prediction pass (eval mode, no grad). Processes X_query in chunks to avoid OOM on large meshes.
def predict_svgp(model, likelihood, X_query, device, chunk_size=4096):
    model.eval()
    likelihood.eval()

    X_query = X_query.to(device)

    means, variances = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for chunk in X_query.split(chunk_size):
            preds = likelihood(model(chunk))
            means.append(preds.mean)
            variances.append(preds.variance)

    return torch.cat(means), torch.cat(variances)
