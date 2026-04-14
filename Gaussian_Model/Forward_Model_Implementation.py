# -*- coding: utf-8 -*-
"""
Forward Model Test.py
SVGP (Sparse Variational Gaussian Process) pipeline.
"""

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
import subprocess
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy

#Import functions from other scripts in this folder
import Read_Write_Functions as RWF
import Forward_Functions as FF

# ── Sampling ──────────────────────────────────────────────────────────────────
def LatinHyperSampler(NoSamples, dimensions, Sample_Params):
    LowerBounds = Sample_Params[:, 0] * Sample_Params[:, 1]
    UpperBounds = Sample_Params[:, 0] * Sample_Params[:, 2]

    sampler = LatinHypercube(d=dimensions)
    unit_samples = sampler.random(n=NoSamples)
    return LowerBounds + (UpperBounds - LowerBounds) * unit_samples

# ── Synthetic Data ────────────────────────────────────────────────────────────
def SyntheticHeatGenerator(U, F, No_Nodes, L=10, k_solid=1.5, noise_std=0.4):
    """
    Conduction-dominated 2-D temperature field with an interior heat source
    and a water-cooled barrier wall.

    Physical setup
    --------------
    - The domain [0,L]×[0,L] represents a solid or slow-moving fluid.
    - A volumetric heat source (e.g. electronics, reaction zone) sits at an
      interior point (x_src, y_src), with total power U [W].
    - The RIGHT wall (x = L) is a cooled barrier through which water flows at
      rate F [L/min].  This sets a convective heat-transfer coefficient h(F)
      via a Dittus-Boelter-type correlation:
          h(F) = h_base * (F / F_nominal)^0.8
      giving a Robin (mixed) boundary condition:
          -k · ∂T/∂x|_{x=L} = h(F) · (T - T_coolant)
    - The other three walls are thermally adiabatic (insulated).

    Parameters
    ----------
    U         : float – heat source power [W]  (e.g. 8–12)
    F         : float – coolant flow rate [L/min]  (e.g. 4–6)
    No_Nodes  : int   – total grid nodes (must be a perfect square)
    L         : float – domain side length [m]
    k_solid   : float – solid thermal conductivity [W/(m·K)]
    noise_std : float – baseline std of additive observation noise [°C]

    Returns
    -------
    X_flat, Y_flat, T_flat : ndarray (No_Nodes,) – coordinates and temperature
    """
    n = int(np.sqrt(No_Nodes))
    x_1d = np.linspace(0, L, n)
    y_1d = np.linspace(0, L, n)
    X, Y = np.meshgrid(x_1d, y_1d)

    T_coolant = 15.0    # coolant inlet temperature [°C]
    T_ambient = 22.0    # initial domain / adiabatic-wall temperature [°C]

    x_src = 0.30 * L
    y_src = 0.55 * L

    r2 = (X - x_src) ** 2 + (Y - y_src) ** 2

    sigma_near = 0.08 * L
    source_near = (U * 4.5) * np.exp(-r2 / (2 * sigma_near ** 2))

    sigma_far = 0.30 * L
    source_far = (U * 1.2) * np.exp(-r2 / (2 * sigma_far ** 2))

    source_field = (U ** 1.3 / U) * source_near + source_far

    F_nominal = 5.0
    h_base    = 800.0
    h_eff     = h_base * (F / F_nominal) ** 0.8

    Bi = h_eff * L / k_solid

    delta_cool = np.clip(k_solid / h_eff, 0.01 * L, 3.0 * L)

    dist_to_cool_wall = L - X
    cool_attenuation = 1.0 - np.exp(-dist_to_cool_wall / delta_cool)

    T = T_coolant + (source_field + (T_ambient - T_coolant)) * cool_attenuation

    sigma_y_spread = 0.20 * L + 0.15 * L / (Bi ** 0.5 + 0.5)
    lateral_spread = (U * 0.8) * np.exp(
        -((Y - y_src) ** 2) / (2 * sigma_y_spread ** 2)
    ) * cool_attenuation

    T = T + lateral_spread

    delta_adiab = 0.05 * L
    adiab_bump = (U * 0.3) * (
        np.exp(-Y / delta_adiab) +
        np.exp(-(L - Y) / delta_adiab) +
        np.exp(-X / delta_adiab)
    )
    T = T + adiab_bump

    T_excess = np.clip(T - T_ambient, 0.0, None)
    local_noise = noise_std * (1.0 + 0.5 * T_excess / (U * 4.0 + 1e-6))
    T += local_noise * np.random.randn(*T.shape)

    return X.flatten(), Y.flatten(), T.flatten()

# ── Plot Single Field ─────────────────────────────────────────────────────────
def plot_temperature_field(X, Y, Z, T, title="3D Heat Distribution"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(X, Y, Z, c=T, cmap="inferno", s=8)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Temperature")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    ax.set_title(title)

    x_range = np.max(X) - np.min(X)
    y_range = np.max(Y) - np.min(Y)
    z_range = np.max(Z) - np.min(Z)
    max_range = max(x_range, y_range, z_range) / 2

    x_mid = (np.max(X) + np.min(X)) / 2
    y_mid = (np.max(Y) + np.min(Y)) / 2
    z_mid = (np.max(Z) + np.min(Z)) / 2

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    plt.tight_layout()
    plt.show()

# ── Plot Side by Side ─────────────────────────────────────────────────────────
def plot_temperature_fields_side_by_side(X, Y, Z, T1, T2,
                                         title1="Field 1",
                                         title2="Field 2",
                                         cmap="inferno"):

    X  = np.asarray(X).reshape(-1)
    Y  = np.asarray(Y).reshape(-1)
    Z  = np.asarray(Z).reshape(-1)
    T1 = np.asarray(T1).reshape(-1)
    T2 = np.asarray(T2).reshape(-1)

    if not (len(X) == len(Y) == len(Z) == len(T1) == len(T2)):
        raise ValueError(
            f"Array lengths differ: X={len(X)}, Y={len(Y)}, Z={len(Z)}, T1={len(T1)}, T2={len(T2)}"
        )

    fig = plt.figure(figsize=(14, 6))

    Tmin = min(np.min(T1), np.min(T2))
    Tmax = max(np.max(T1), np.max(T2))

    def set_equal_axes(ax):
        max_range = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2
        mid_x = np.mean([np.min(X), np.max(X)])
        mid_y = np.mean([np.min(Y), np.max(Y)])
        mid_z = np.mean([np.min(Z), np.max(Z)])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1 = fig.add_subplot(121, projection='3d')
    #sc1 = ax1.scatter(X, Y, Z, c=T1, cmap=cmap, vmin=Tmin, vmax=Tmax, s=8)
    sc1 = ax1.scatter(
        X, Y, Z,
        c=T1,
        cmap=cmap,
        vmin=Tmin,
        vmax=Tmax,
        s=8,
        depthshade=False 
    )
    ax1.set_title(title1)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_equal_axes(ax1)

    ax2 = fig.add_subplot(122, projection='3d')
    #sc2 = ax2.scatter(X, Y, Z, c=T2, cmap=cmap, vmin=Tmin, vmax=Tmax, s=8)
    sc2 = ax2.scatter(
        X, Y, Z,
        c=T2,
        cmap=cmap,
        vmin=Tmin,
        vmax=Tmax,
        s=8,
        depthshade=False
    )
    ax2.set_title(title2)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    set_equal_axes(ax2)

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], pad=0.02, fraction=0.05, location="right")
    cbar.set_label("Temperature")

    plt.tight_layout()
    plt.show()

#%%
# ──────────────────────────────────────────────────────────
# Main
#  ──────────────────────────────────────────────────────────
#Change these to True if you want to run new simulations
NewTraining = False
NewTesting = False
#  ── Files ──────────────────────────────────────────────────────────
ParametersFile = r"\\wsl.localhost\Ubuntu-22.04\home\kfletch123\GeneralFolder\HIVEsim\HIVE\HIVE_MFEM_Repo\Baseline_HTC\Parameters.i"
OutputFile     = r"\\wsl.localhost\Ubuntu-22.04\home\kfletch123\GeneralFolder\HIVEsim\HIVE\HIVE_MFEM_Repo\Baseline_HTC\THeat_Flow_TV_HTC_ex.e"
HippoVariables = ['coil_current', 'flow_rate']

# ── Simulation Parameters ─────────────────────────────────────────────────────
Power_Params  = [2200, 0.85, 1.15]   # [nominal, lower_factor, upper_factor]
Flow_Params   = [7.5,  0.85, 1.15]   # [nominal, lower_factor, upper_factor]
Sample_Params = np.array([Power_Params, Flow_Params])
NoSamples     = 40
dimensions    = 2
num_epochs    = 600
Downsample_Factor = 12
No_Inducing   = 1000
coord_dims    = 3
global_dims   = 2

# ── Hyper-parameters ──────────────────────────────────────────────────────────
init_outputscale  = 0.6
init_lengthscales = 0.2
init_noise        = 1e-3
adam_lr           = 5e-4
ngd_lr            = 0.01

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Latin Hypercube Sampling ──────────────────────────────────────────────────
train_scaled_samples = LatinHyperSampler(NoSamples, dimensions, Sample_Params)
Power_Samples = train_scaled_samples[:, 0]
Flow_Samples  = train_scaled_samples[:, 1]

#%%
# ── Generate Training Data ────────────────────────────────────────────────────
Data_Record = []

if NewTraining==True:
    for i in range(len(Power_Samples)):
        #Write the current sample values
        RWF.HippoWrite(ParametersFile, HippoVariables[0], Power_Samples[i])
        RWF.HippoWrite(ParametersFile, HippoVariables[1], Flow_Samples[i])
    
        #Run the simulation
        result = subprocess.run(
            [
                "wsl", "bash", "-lc",
                "source ~/.bashrc && /home/kfletch123/GeneralFolder/HIVEsim/HIVE/HIVE_MFEM_Repo/Baseline_HTC/run.sh"
            ],
            capture_output=True,
            text=True
        )
    
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("Return code:", result.returncode)
    
        if result.returncode != 0:
            raise RuntimeError(
                f"Simulation failed at sample {i} "
                f"(U={Power_Samples[i]:.1f}, F={Flow_Samples[i]:.3f}).\n"
                f"STDERR: {result.stderr}"
            )
    
        #Read Outputs
        (block_coords, T) = RWF.HippoExodusReader(OutputFile, output=False)
        coords = vtk_to_numpy(block_coords)
    
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
    
        U_col = np.full((X.shape[0], 1), Power_Samples[i])
        F_col = np.full((X.shape[0], 1), Flow_Samples[i])
    
        sample_array = np.hstack([X.reshape(-1, 1),
                                   Y.reshape(-1, 1),
                                   Z.reshape(-1, 1),
                                   U_col, F_col,
                                   T.reshape(-1, 1)])    # (No_Nodes, 6)
        Data_Record.append(sample_array)
        print(i)
    
    Final_Array = np.vstack(Data_Record)
    plot_temperature_field(X, Y, Z, T)
    
    #Save Data
    np.save("SavedTrainingData_Full.npy", Final_Array)

#%%
Final_Array = np.load("SavedTrainingData_Full.npy")

# ── Stratified Downsampling ───────────────────────────────────────────────────
No_Nodes_actual = (Final_Array[:, 3] == Final_Array[0, 3]).sum()  # nodes per run
down_bins = max(3, int((No_Nodes_actual / (4 * Downsample_Factor)) ** (1/3)))
print(f"Mesh nodes per run: {No_Nodes_actual},  down_bins: {down_bins}")

U_vals = Final_Array[:, 3]
F_vals = Final_Array[:, 4]

# Unique operating conditions (one row per simulation run)
samples = np.unique(np.column_stack([U_vals, F_vals]), axis=0)

downsampled_records = []
for U, F in samples:
    mask   = (U_vals == U) & (F_vals == F)
    sample = Final_Array[mask]

    coords = sample[:, :3]
    T      = sample[:, 5]

    coords_ds, T_ds = FF.stratified_downsample(
        coords, T, Downsample_Factor, n_bins=down_bins
    )

    U_col = np.full((coords_ds.shape[0], 1), U)
    F_col = np.full((coords_ds.shape[0], 1), F)

    sample_ds = np.hstack([
        coords_ds,
        U_col,
        F_col,
        T_ds.reshape(-1, 1)
    ])
    downsampled_records.append(sample_ds)

Final_Array = np.vstack(downsampled_records)
print(f"Downsampled training set: {Final_Array.shape[0]} points "
      f"({Final_Array.shape[0] / len(samples):.0f} per run)")

#%%
#Training

#Uncomment if need to debug
#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ── Validation split (by run, not by node) ────────────────────────────────────
rng          = np.random.default_rng(seed=42)
n_val_runs   = max(1, len(samples) // 5)       
val_UF       = samples[rng.choice(len(samples), size=n_val_runs, replace=False)]
val_mask     = np.zeros(len(Final_Array), dtype=bool)
for U, F in val_UF:
    val_mask |= (Final_Array[:, 3] == U) & (Final_Array[:, 4] == F)
train_mask = ~val_mask

print(f"Training points: {train_mask.sum()},  Validation points: {val_mask.sum()}")

X_train_np = Final_Array[train_mask, :5]
Y_train_np = Final_Array[train_mask, 5].reshape(-1, 1)
X_val_np   = Final_Array[val_mask,   :5]
Y_val_np   = Final_Array[val_mask,   5].reshape(-1, 1)

# ── Prepare Training Tensors ──────────────────────────────────────────────────
X_min, X_max, Ydestand, X_train_std, Y_train_std = FF.TrainInputStandardisation(
    X_train_np, Y_train_np, yLog=False
)

# Validation: standardise using training statistics only
X_val_std = FF.TestRunStandardisation(X_val_np, X_min, X_max)
y_mean, y_std = Ydestand
Y_val_std = (Y_val_np - y_mean) / y_std

X_train_stand = torch.tensor(X_train_std, dtype=torch.float32, device=device)
X_train_stand = torch.clamp(X_train_stand, 0.0, 1.0)
y_train_stand = torch.tensor(Y_train_std, dtype=torch.float32, device=device)

X_val_t = torch.tensor(X_val_std, dtype=torch.float32, device=device)
y_val_t = torch.tensor(Y_val_std, dtype=torch.float32, device=device)

# ── Build Model ───────────────────────────────────────────────────────────────
Initial_Inducing = FF.init_inducing_points_spatial(
    X_train_stand, M=No_Inducing, coord_dims=coord_dims, global_dims=global_dims
)
Initial_Inducing = torch.clamp(Initial_Inducing, 0.0, 1.0)

model = FF.SingleTaskSVGP(
    Initial_Inducing,
    coord_dims,
    global_dims,
    init_lengthscales,
    init_outputscale,
    init_noise).to(device)

likelihood = model.likelihood.to(device)

# ── Build Optimisers ──────────────────────────────────────────────────────────
num_data = X_train_stand.shape[0]

#NGD owns variational parameters; built here and passed in
ngd_opt = NGD(model.variational_parameters(), num_data=num_data, lr=ngd_lr)

#Variational params belong to NGD; everything else goes to Adam
adam_params = [
    {"params": model.covar_module.parameters(),                      "lr": adam_lr},
    {"params": model.mean_module.parameters(),                        "lr": adam_lr},
    {"params": model.likelihood.parameters(),                         "lr": adam_lr},
    {"params": [model.variational_strategy.inducing_points],          "lr": adam_lr},
]
adam_opt = torch.optim.Adam(adam_params)

# ── Train ─────────────────────────────────────────────────────────────────────
model.train()
likelihood.train()

results = FF.TrainHybridSVGP(
    model, likelihood,
    X_train_stand, y_train_stand,
    ngd_opt, adam_opt,
    X_val=X_val_t,          
    y_val=y_val_t,           
    iters=num_epochs,
    device=str(device),
    verbose=True,
)

print(f"\nTraining complete. Best val RMSE (standardised): {results['best_metric']:.6f}")

# ── Training curve ────────────────────────────────────────────────────────────
history = results["loss_history"]
epochs  = [h["epoch"]   for h in history]
elbo    = [h["elbo"]    for h in history]
monitor = [h["monitor"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, elbo);    ax1.set_title("Train ELBO"); ax1.set_xlabel("Epoch")
ax2.plot(epochs, monitor); ax2.set_title("Val RMSE (standardised)"); ax2.set_xlabel("Epoch")
plt.tight_layout(); plt.show()

#%%
# ── Generate Test Queries ─────────────────────────────────────────────────────
NoTests = 8

# Sample test inputs slightly interior to training bounds (avoids extrapolation)
FP_min = 1.025 * Power_Params[0] * Power_Params[1]
FP_max = 0.975 * Power_Params[0] * Power_Params[2]
FF_min = 1.025 * Flow_Params[0]  * Flow_Params[1]
FF_max = 0.975 * Flow_Params[0]  * Flow_Params[2]

FP_query = np.random.uniform(FP_min, FP_max, size=NoTests)
FF_query = np.random.uniform(FF_min, FF_max, size=NoTests)

Actuals    = []
Rand_Query = []

Query_Coords = []

if NewTesting==True:
    for i in range(NoTests):
        RWF.HippoWrite(ParametersFile, HippoVariables[0], FP_query[i])
        RWF.HippoWrite(ParametersFile, HippoVariables[1], FF_query[i])
    
        result = subprocess.run(
            [
                "wsl", "bash", "-lc",
                "source ~/.bashrc && /home/kfletch123/GeneralFolder/HIVEsim/HIVE/HIVE_MFEM_Repo/Baseline_HTC/run.sh"
            ],
            capture_output=True,
            text=True
        )
    
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("Return code:", result.returncode)
    
        if result.returncode != 0:
            raise RuntimeError(
                f"Test simulation failed at sample {i} "
                f"(U={FP_query[i]:.1f}, F={FF_query[i]:.3f}).\n"
                f"STDERR: {result.stderr}"
            )
    
        (block_coords, T) = RWF.HippoExodusReader(OutputFile, output=False)
        coords = vtk_to_numpy(block_coords)
    
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
    
        U_col = np.full((X.shape[0], 1), FP_query[i])
        F_col = np.full((X.shape[0], 1), FF_query[i])
    
        Query = np.hstack([X.reshape(-1, 1),
                           Y.reshape(-1, 1),
                           Z.reshape(-1, 1),
                           U_col, F_col])    # (No_Nodes, 5)
    
        Rand_Query.append(Query)
        Actuals.append(T.reshape(-1, 1))
        Query_Coords.append(coords[:, :3])   # store per-test XYZ
        plot_temperature_field(X, Y, Z, T)
        print(i)
    
    np.save("SavedTestingQuery_1.npy",  np.array(Rand_Query,    dtype=object))
    np.save("SavedTestingData_1.npy",   np.array(Actuals,       dtype=object))
    np.save("SavedTestingCoords_1.npy", np.array(Query_Coords,  dtype=object))

#%%
Rand_Query   = np.load("SavedTestingQuery_1.npy",  allow_pickle=True)
Actuals      = np.load("SavedTestingData_1.npy",   allow_pickle=True)
Query_Coords = np.load("SavedTestingCoords_1.npy", allow_pickle=True)

# %%
def plot_temperature_fields_side_by_side(X, Y, Z, T1, T2,
                                         title1="Field 1",
                                         title2="Field 2",
                                         cmap="inferno"):

    X  = np.asarray(X, dtype=float).reshape(-1)
    Y  = np.asarray(Y, dtype=float).reshape(-1)
    Z  = np.asarray(Z, dtype=float).reshape(-1)
    T1 = np.asarray(T1, dtype=float).reshape(-1)
    T2 = np.asarray(T2, dtype=float).reshape(-1)

    if not (len(X) == len(Y) == len(Z) == len(T1) == len(T2)):
        raise ValueError("Array lengths differ")

    # 🔑 Prevent planar-Z mplot3d crash
    if np.ptp(Z) == 0.0:
        Z = Z + 1e-9

    fig = plt.figure(figsize=(14, 6))

    Tmin = min(T1.min(), T2.min())
    Tmax = max(T1.max(), T2.max())

    def set_equal_axes(ax):
        max_range = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2
        mid_x = (X.min() + X.max()) / 2
        mid_y = (Y.min() + Y.max()) / 2
        mid_z = (Z.min() + Z.max()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(X, Y, Z, c=T1, cmap=cmap,
                      vmin=Tmin, vmax=Tmax, s=8,
                      depthshade=False)
    ax1.set_title(title1)
    set_equal_axes(ax1)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(X, Y, Z, c=T2, cmap=cmap,
                      vmin=Tmin, vmax=Tmax, s=8,
                      depthshade=False)
    ax2.set_title(title2)
    set_equal_axes(ax2)

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], pad=0.02, fraction=0.05)
    cbar.set_label("Temperature")

    plt.subplots_adjust(wspace=0.25)
    plt.show()

# ── Predict and Evaluate ──────────────────────────────────────────────────────
Preds         = []
RMSE          = []
Mean_Percents = []

for i in range(NoTests):
    # Standardise
    X_query_std = FF.TestRunStandardisation(Rand_Query[i], X_min, X_max)
    X_query_std = np.asarray(X_query_std, dtype=np.float32)
    X_query_t   = torch.tensor(X_query_std, dtype=torch.float32)

    # Predict (standardised space) — chunked internally to avoid OOM
    mean_std, var_std = FF.predict_svgp(model, likelihood, X_query_t, device)

    # Destandardise
    y_mean_pred, y_var_pred, y_std_pred = FF.OutputDestandardisation(
        mean_std, var_std, Ydestand, yLog=False
    )

    Preds.append((y_mean_pred, y_var_pred, y_std_pred))

    # Evaluate
    Y_actual = Actuals[i]
    diff    = abs(Y_actual.ravel() - y_mean_pred.ravel())#Y_actual - y_mean_pred

    rmse = np.sqrt(np.mean(diff ** 2))
    RMSE.append(rmse)

    pct_err = 100 * diff.ravel() / (Y_actual.ravel() + 1e-9)
    Mean_Percents.append(float(np.mean(pct_err)))
    print(i)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n── Prediction Results ──")
for i in range(NoTests):
    print(f"  Test {i+1}: RMSE = {RMSE[i]:.4f} | Mean % error = {Mean_Percents[i]:.2f}%")

print(f"\nOverall Mean RMSE      : {np.mean(RMSE):.4f}")
print(f"Overall Mean % Error   : {np.mean(Mean_Percents):.2f}%")

# ── Prediction Plots ──────────────────────────────────────────────────────────y
for i in range(len(Actuals)):
    y_mean_pred = Preds[i][0]
    y_actual    = Actuals[i]
    Xi = Query_Coords[i][:, 0]
    Yi = Query_Coords[i][:, 1]
    Zi = Query_Coords[i][:, 2]
    plot_temperature_fields_side_by_side(
        Xi.ravel(), Yi.ravel(), Zi.ravel(), y_actual.ravel(), y_mean_pred.ravel(),
        title1="Actual Distribution",
        title2="Predicted Distribution"
    )
    print(i)
