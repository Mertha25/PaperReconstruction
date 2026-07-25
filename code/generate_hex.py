"""
generate_hex.py
===============
Generates the HEX parametric dataset for thermal field reconstruction.

GOVERNING EQUATION (convection-diffusion, steady state):
    rho * cp * (ux * dT/dx + uy * dT/dy) = k * (d2T/dx2 + d2T/dy2)

    Left side  = convection : heat carried by the moving fluid
    Right side = conduction : heat diffused through the material

GEOMETRY:
    Rectangular domain Lx=0.3m x Ly=0.1m on a 64x128 grid.
    3 horizontal passes with prescribed velocity:
        Pass 1 (top)    : fluid flows LEFT to RIGHT  (+u0)
        Pass 2 (middle) : fluid flows RIGHT to LEFT  (-u0)
        Pass 3 (bottom) : fluid flows LEFT to RIGHT  (+u0)

VARIABLE PARAMETERS (Latin Hypercube Sampling over 4D space):
    T_in   : hot fluid inlet temperature  [340, 400] K
    T_cold : cold wall temperature        [280, 300] K
    u0     : flow velocity magnitude      [0.01, 0.1] m/s
    k      : thermal conductivity         [1, 100] W/(m.K)

SENSOR PLACEMENT (random, distributed across all passes):
    M sensors are placed randomly in the interior of the domain,
    covering all 3 passes. This follows the TFRD benchmark approach
    (Chen et al. 2023) and ensures the model receives global thermal
    information from all regions of the exchanger.
    Sensor positions vary between samples during training, which forces
    the model to learn the underlying physics rather than memorizing
    fixed measurement configurations. At inference time, any fixed
    sensor layout can be used.

INPUT ENCODING (Strategy A, following Prof. KYA Section 4.6):
    Each sample is encoded as a 3-channel tensor (3, NY, NX):
        Channel 0 : sensor temperature values at observed positions, 0 elsewhere
        Channel 1 : binary mask -- 1 at sensor positions, 0 elsewhere
        Channel 2 : T_in broadcast over the entire grid (operating condition)

OUTPUT: 5000 samples -> 4000 train / 500 val / 500 test
        Each .mat file contains:
            T_field  (float32, NY x NX)    : full temperature field [K]
            X_input  (float32, 3 x NY x NX): Strategy A sparse image
            u_obs    (float32, M)           : sensor readings [K]
            u_pos    (int32,   M x 2)       : sensor positions (row, col)
            params   (float32, 4)           : [T_in, T_cold, u0, k]

COLORMAP CONVENTION (following KYA Figure 1):
    Blue  = cold regions
    Yellow/Red = hot regions
    Colormap: 'RdYlBu_r'  (blue-cold, yellow-hot)

USAGE:
    pip install numpy scipy matplotlib
    python generate_hex.py
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import qmc


# ===============================================================================
# SECTION 1 -- PHYSICAL CONSTANTS AND GRID
# ===============================================================================

# Fluid physical constants (water-like coolant)
RHO    = 1000.0        # fluid density            [kg/m3]
CP     = 4182.0        # specific heat capacity   [J/(kg.K)]
RHO_CP = RHO * CP      # volumetric heat capacity [J/(m3.K)]

# Grid dimensions (following KYA Section 4.4: 64x128 grid)
NY = 64                # number of rows    (height direction, y-axis)
NX = 128               # number of columns (width  direction, x-axis)
LX = 0.3               # domain width  [m]
LY = 0.1               # domain height [m]
DX = LX / (NX - 1)    # grid spacing in x ~ 0.00236 m
DY = LY / (NY - 1)    # grid spacing in y ~ 0.00159 m
N  = NY * NX           # total nodes = 8192

# 3-pass row boundaries (i=0 is top row, i=NY-1 is bottom row)
P1_START, P1_END = 0,           NY // 3       # rows  0..20 : flow right (+u0)
P2_START, P2_END = NY // 3,     2 * NY // 3   # rows 21..41 : flow left  (-u0)
P3_START, P3_END = 2 * NY // 3, NY            # rows 42..63 : flow right (+u0)

# Parameter space bounds (KYA Section 4.4)
PARAM_BOUNDS = {
    'T_in'  : (340.0, 400.0),   # hot inlet temperature  [K]
    'T_cold': (280.0, 300.0),   # cold wall temperature  [K]
    'u0'    : (0.01,  0.1  ),   # flow velocity          [m/s]
    'k'     : (1.0,   100.0),   # thermal conductivity   [W/(m.K)]
}

# Colormap (blue=cold, yellow/red=hot, following KYA Figure 1)
CMAP_FIELD = 'RdYlBu_r'


# ===============================================================================
# SECTION 2 -- VELOCITY FIELD
# ===============================================================================

def build_velocity_field(u0):
    """
    Build the prescribed 3-pass velocity field.

    The velocity is purely horizontal (uy = 0 everywhere).
    This prescription avoids solving Navier-Stokes while capturing
    the essential multi-pass convection pattern.

    Args:
        u0 (float): flow velocity magnitude [m/s]

    Returns:
        ux (ndarray): shape (NY, NX) -- x-velocity at every node [m/s]
        uy (ndarray): shape (NY, NX) -- y-velocity (all zeros)
    """
    ux = np.zeros((NY, NX), dtype=np.float64)
    ux[P1_START:P1_END, :] = +u0   # Pass 1: left to right
    ux[P2_START:P2_END, :] = -u0   # Pass 2: right to left
    ux[P3_START:P3_END, :] = +u0   # Pass 3: left to right
    return ux, np.zeros((NY, NX), dtype=np.float64)


# ===============================================================================
# SECTION 3 -- SPARSE LINEAR SYSTEM ASSEMBLY
# ===============================================================================

def build_system(T_in, T_cold, u0, k):
    """
    Assemble the sparse linear system A.T = b for the PDE:
        k.(d2T/dx2 + d2T/dy2) - rho.cp.ux.dT/dx = 0

    Discretization schemes:
        Diffusion  : central differences (2nd-order accurate)
        Convection : upwind scheme (1st-order, unconditionally stable)

    The upwind scheme selects the upstream neighbor based on flow direction:
        ux >= 0 (flow right) : dT/dx ~ (T[i,j] - T[i,j-1]) / dx
        ux <  0 (flow left)  : dT/dx ~ (T[i,j+1] - T[i,j]) / dx

    Boundary conditions:
        Top wall    (i=0)    : T = T_cold  (Dirichlet -- outer cold wall)
        Bottom wall (i=NY-1) : T = T_cold  (Dirichlet -- outer cold wall)
        Left wall   (j=0)    : T = T_in    for Pass 1 rows (hot fluid inlet)
                               dT/dx = 0   for Pass 2,3 rows (Neumann)
        Right wall  (j=NX-1) : dT/dx = 0   all rows (Neumann -- U-turn/outlet)
        Corner nodes         : assigned to top/bottom walls (avoid conflict)
    """
    ux, _ = build_velocity_field(u0)
    I, J  = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')

    top_mask   = (I == 0)
    bot_mask   = (I == NY - 1)
    left_mask  = (J == 0)
    right_mask = (J == NX - 1)
    int_mask   = ~(top_mask | bot_mask | left_mask | right_mask)

    def flat(i_a, j_a):
        return i_a * NX + j_a

    b = np.zeros(N, dtype=np.float64)
    rows_list, cols_list, vals_list = [], [], []

    def add(r, c, v):
        rows_list.append(np.atleast_1d(r).ravel())
        cols_list.append(np.atleast_1d(c).ravel())
        vals_list.append(np.atleast_1d(v).ravel())

    # -- Interior nodes -------------------------------------------------------
    i_int = I[int_mask]; j_int = J[int_mask]; n_int = flat(i_int, j_int)

    # Diffusion in x: k/dx2 * [T(i,j-1) - 2*T(i,j) + T(i,j+1)]
    add(n_int, flat(i_int, j_int-1), np.full(len(n_int),  k/DX**2))
    add(n_int, n_int,                np.full(len(n_int), -2*k/DX**2))
    add(n_int, flat(i_int, j_int+1), np.full(len(n_int),  k/DX**2))

    # Diffusion in y: k/dy2 * [T(i-1,j) - 2*T(i,j) + T(i+1,j)]
    add(n_int, flat(i_int-1, j_int), np.full(len(n_int),  k/DY**2))
    add(n_int, n_int,                np.full(len(n_int), -2*k/DY**2))
    add(n_int, flat(i_int+1, j_int), np.full(len(n_int),  k/DY**2))

    # Convection (upwind scheme)
    ux_int = ux[i_int, j_int]
    pos    = ux_int >= 0    # flow right: upwind from left
    neg    = ~pos            # flow left : upwind from right

    n_p, u_p = n_int[pos], ux_int[pos]
    add(n_p, n_p,                              -RHO_CP * u_p / DX)
    add(n_p, flat(i_int[pos], j_int[pos] - 1), RHO_CP * u_p / DX)

    n_n, u_n = n_int[neg], ux_int[neg]
    add(n_n, flat(i_int[neg], j_int[neg] + 1), -RHO_CP * u_n / DX)
    add(n_n, n_n,                               RHO_CP * u_n / DX)

    # -- Top wall: Dirichlet T = T_cold ---------------------------------------
    n_top = flat(I[top_mask], J[top_mask])
    add(n_top, n_top, np.ones(len(n_top))); b[n_top] = T_cold

    # -- Bottom wall: Dirichlet T = T_cold ------------------------------------
    n_bot = flat(I[bot_mask], J[bot_mask])
    add(n_bot, n_bot, np.ones(len(n_bot))); b[n_bot] = T_cold

    # -- Left wall (exclude corners) ------------------------------------------
    left_i = I[left_mask].flatten()
    left_i = left_i[(left_i > 0) & (left_i < NY - 1)]
    n_left = flat(left_i, np.zeros_like(left_i))

    p1 = (left_i >= P1_START) & (left_i < P1_END)
    add(n_left[p1],  n_left[p1],  np.ones(p1.sum()))
    b[n_left[p1]] = T_in                              # hot inlet

    add(n_left[~p1], n_left[~p1],      np.ones((~p1).sum()))
    add(n_left[~p1], n_left[~p1] + 1, -np.ones((~p1).sum()))

    # -- Right wall (exclude corners): Neumann dT/dx = 0 ----------------------
    right_i = I[right_mask].flatten()
    right_i = right_i[(right_i > 0) & (right_i < NY - 1)]
    n_right = flat(right_i, np.full_like(right_i, NX - 1))
    add(n_right, n_right,     np.ones(len(n_right)))
    add(n_right, n_right - 1, -np.ones(len(n_right)))

    A = sp.coo_matrix(
        (np.concatenate(vals_list),
         (np.concatenate(rows_list).astype(np.int32),
          np.concatenate(cols_list).astype(np.int32))),
        shape=(N, N)
    ).tocsr()
    return A, b


# ===============================================================================
# SECTION 4 -- SOLVER
# ===============================================================================

def solve_hex(T_in, T_cold, u0, k):
    """
    Solve the convection-diffusion PDE and return the 2D temperature field.
    Uses scipy sparse direct solver (SuperLU factorization).

    Args:
        T_in   (float): hot inlet temperature [K]
        T_cold (float): cold wall temperature  [K]
        u0     (float): flow velocity [m/s]
        k      (float): thermal conductivity [W/(m.K)]

    Returns:
        T (ndarray): shape (NY, NX), dtype float32 -- temperature field [K]
    """
    A, b   = build_system(T_in, T_cold, u0, k)
    T_flat = spla.spsolve(A, b)
    return T_flat.reshape(NY, NX).astype(np.float32)


# ===============================================================================
# SECTION 5 -- LATIN HYPERCUBE SAMPLING
# ===============================================================================

def sample_parameters(n_samples, seed=42):
    """
    Sample n_samples configurations over the 4D parameter space using LHS.

    Latin Hypercube Sampling (LHS) guarantees uniform coverage of the
    parameter space: it divides each dimension into n equal intervals
    and places exactly one sample per interval. This is far superior to
    pure random sampling for the same sample budget.

    Args:
        n_samples (int): number of configurations
        seed      (int): random seed for reproducibility

    Returns:
        List of dicts with keys: T_in, T_cold, u0, k
    """
    sampler   = qmc.LatinHypercube(d=4, seed=seed)
    unit_cube = sampler.random(n=n_samples)
    low  = [PARAM_BOUNDS[p][0] for p in ['T_in', 'T_cold', 'u0', 'k']]
    high = [PARAM_BOUNDS[p][1] for p in ['T_in', 'T_cold', 'u0', 'k']]
    scaled = qmc.scale(unit_cube, low, high)
    return [{'T_in': float(r[0]), 'T_cold': float(r[1]),
             'u0':  float(r[2]), 'k':      float(r[3])}
            for r in scaled]


# ===============================================================================
# SECTION 6 -- SENSOR PLACEMENT (RANDOM, DISTRIBUTED ACROSS ALL PASSES)
# ===============================================================================

def place_sensors(T_field, n_sensors=16):
    """
    Randomly place M sensors distributed across the interior of the domain.

    Placement strategy (following KYA Figure 1 and Chen et al. 2023):
        Sensors are placed randomly in the interior (avoiding boundary nodes),
        covering all 3 passes of the exchanger. Different positions are drawn
        for each sample during training, which forces the model to learn the
        underlying physics rather than memorizing specific measurement layouts.

    This approach has two key advantages over fixed placement:
        1. Global coverage: the model sees measurements from all 3 passes
           in expectation, providing a complete thermal picture.
        2. Robustness: by training with varying configurations, the model
           generalizes to any sensor layout at inference time.

    Different values of M (8, 12, 16) are tested at evaluation time
    to study robustness to sensor count reduction.

    Args:
        T_field   (ndarray): shape (NY, NX) -- full temperature field [K]
        n_sensors (int)    : number of sensors M

    Returns:
        u_obs (ndarray): shape (M,)   -- sensor temperature readings [K]
        u_pos (ndarray): shape (M, 2) -- sensor positions (row, col)
    """
    # Random positions in the interior (exclude boundary rows/cols)
    rows = np.random.randint(1, NY - 1, size=n_sensors)
    cols = np.random.randint(1, NX - 1, size=n_sensors)
    positions = np.stack([rows, cols], axis=1).astype(np.int32)
    readings  = np.array([T_field[r, c] for r, c in positions],
                         dtype=np.float32)
    return readings, positions


# ===============================================================================
# SECTION 7 -- STRATEGY A INPUT ENCODING (KYA SECTION 4.6)
# ===============================================================================

def encode_sparse_image(u_obs, u_pos, params):
    """
    Encode sparse sensor observations as a 3-channel image (Strategy A).

    This encoding is recommended by KYA (Section 4.6) for Procedia:
    it allows all models (MLP, U-Net, ViT, FNO) to receive the exact
    same input format for a fair comparison.

    The 3 channels provide:
        Channel 0 -- WHAT is measured:
            Temperature values at sensor positions, 0.0 elsewhere.
            This gives the model the actual measurement values.

        Channel 1 -- WHERE sensors are located:
            Binary mask: 1.0 at sensor positions, 0.0 everywhere else.
            This tells the model which positions have been observed,
            avoiding confusion between "T=0" and "no measurement here".

        Channel 2 -- Global operating context:
            Hot inlet temperature T_in broadcast over the entire grid.
            This encodes the global thermal regime of the sample,
            allowing the model to condition its reconstruction on
            the expected temperature range.

    Args:
        u_obs  (ndarray): shape (M,)   -- sensor readings [K]
        u_pos  (ndarray): shape (M, 2) -- sensor positions (row, col)
        params (dict)   : {'T_in', 'T_cold', 'u0', 'k'}

    Returns:
        X (ndarray): shape (3, NY, NX), dtype float32 -- model input tensor
    """
    X = np.zeros((3, NY, NX), dtype=np.float32)
    for idx, (r, c) in enumerate(u_pos):
        X[0, r, c] = u_obs[idx]   # channel 0: sensor value
        X[1, r, c] = 1.0          # channel 1: binary mask
    X[2, :, :] = params['T_in']   # channel 2: T_in broadcast
    return X


# ===============================================================================
# SECTION 8 -- DATASET GENERATION
# ===============================================================================

def generate_dataset(n_total=5000, n_sensors=16,
                     output_dir=r'D:\PaperKya\HEX_dataset',
                     seed=42):
    """
    Generate the full HEX parametric dataset.

    Dataset split (following standard ML practice):
        train : 4000 samples (80%) -- used for model training
        val   :  500 samples (10%) -- used for hyperparameter tuning
        test  :  500 samples (10%) -- used for final evaluation only

    Each sample contains a unique parameter configuration and a unique
    random sensor placement, ensuring the model sees diverse training
    conditions and sensor layouts.

    Args:
        n_total    (int): total number of samples (default 5000)
        n_sensors  (int): number of sensors per sample (default 16)
        output_dir (str): root output directory
        seed       (int): random seed for reproducibility
    """
    np.random.seed(seed)

    n_train = int(n_total * 0.80)           # 4000
    n_val   = int(n_total * 0.10)           #  500
    n_test  = n_total - n_train - n_val     #  500 LHS generated, 450 saved
    # NOTE: test/ folder will contain 450 LHS + 50 hard = 500 samples total
    n_test_lhs = n_test - len(HARD_CONFIGS) * 5   # 500 - 50 = 450 LHS samples

    splits = [
        ('train', slice(0,               n_train)),
        ('val',   slice(n_train,         n_train + n_val)),
        ('test',  slice(n_train + n_val, n_train + n_val + n_test_lhs)),
    ]

    print("=" * 62)
    print("  HEX Dataset Generator")
    print("  Physics    : 2D Convection-Diffusion (steady state)")
    print(f"  Grid       : {NY} x {NX} = {N} nodes")
    print(f"  Domain     : {LX}m x {LY}m  (3 passes)")
    print(f"  Samples    : {n_total}  "
          f"({n_train} train / {n_val} val / {n_test_lhs} LHS test + 50 hard)")
    print(f"  Sensors    : M = {n_sensors}  "
          f"(random placement, all passes covered)")
    print(f"  Sampling   : Latin Hypercube Sampling (4 parameters)")
    print(f"  Colormap   : RdYlBu_r  (blue=cold, yellow=hot)")
    print("=" * 62)

    print(f"\nSampling {n_total} configurations via LHS...")
    all_params = sample_parameters(n_total, seed=seed)

    t_global = time.time()

    for split_name, slc in splits:
        folder = os.path.join(output_dir, split_name)
        os.makedirs(folder, exist_ok=True)
        split_params = all_params[slc]
        n_split      = len(split_params)

        print(f"\n[{split_name.upper()}] Generating {n_split} samples...")
        t0 = time.time()

        for idx, params in enumerate(split_params):

            # 1. Solve PDE for this parameter configuration
            T_field = solve_hex(params['T_in'], params['T_cold'],
                                params['u0'],   params['k'])

            # 2. Place M sensors randomly (different positions per sample)
            u_obs, u_pos = place_sensors(T_field, n_sensors=n_sensors)

            # 3. Encode input as Strategy A 3-channel image
            X_input = encode_sparse_image(u_obs, u_pos, params)

            # 4. Save .mat file
            sio.savemat(
                os.path.join(folder, f'sample_{idx:04d}.mat'),
                {
                    'T_field': T_field,
                    'X_input': X_input,
                    'u_obs'  : u_obs,
                    'u_pos'  : u_pos,
                    'params' : np.array([params['T_in'], params['T_cold'],
                                         params['u0'],   params['k']],
                                        dtype=np.float32),
                }
            )

            # Progress log every 500 samples
            if (idx + 1) % 500 == 0:
                Pe      = RHO_CP * params['u0'] * DX / params['k']
                elapsed = time.time() - t0
                eta     = elapsed / (idx + 1) * (n_split - idx - 1)
                print(f"  [{idx+1:5d}/{n_split}]  "
                      f"T_in={params['T_in']:.0f}K  "
                      f"T_cold={params['T_cold']:.0f}K  "
                      f"u0={params['u0']:.3f}m/s  "
                      f"k={params['k']:.0f}W/mK  "
                      f"Pe={Pe:.1f}  "
                      f"T=[{T_field.min():.1f},{T_field.max():.1f}]K  "
                      f"ETA={eta/60:.1f}min")

    total = time.time() - t_global
    print(f"\nTotal generation time: {total/60:.1f} minutes")
    print(f"Dataset saved to: {output_dir}")


# ===============================================================================
# SECTION 9 -- HARD/EXTREME TEST CASES
# ===============================================================================

# Definition of the 10 extreme parameter configurations
# Each case targets a specific physical difficulty:
#   - High Pe : convection-dominated, steep gradients, asymmetric fields
#   - Low Pe  : diffusion-dominated, smooth fields (easy but verify robustness)
#   - Max ΔT  : T_in - T_cold at maximum → largest thermal stress
#   - Min ΔT  : T_in - T_cold at minimum → smallest thermal contrast
#   - Corners : combinations of all-max or all-min parameter values
#
# Pe = RHO_CP * u0 * DX / k
# Pe_max ~ 4.182e6 * 0.1 * 0.00236 / 1.0   ~ 986  (convection-dominated)
# Pe_min ~ 4.182e6 * 0.01 * 0.00236 / 100  ~ 0.99 (diffusion-dominated)

HARD_CONFIGS = [
    # name            T_in   T_cold   u0      k      reason
    ('hard_Pe_max',   400.0, 280.0,  0.100,   1.0),  # Pe~986, max DeltaT
    ('hard_Pe_min',   340.0, 300.0,  0.010, 100.0),  # Pe~0.99, min DeltaT
    ('hard_DT_max_k', 400.0, 280.0,  0.100, 100.0),  # max DeltaT + max k
    ('hard_Pe_hi_DT', 340.0, 300.0,  0.100,   1.0),  # Pe max + min DeltaT
    ('hard_Pe_lo_k',  400.0, 280.0,  0.010,   1.0),  # min u0 + min k
    ('hard_mid_Pe_hi',370.0, 290.0,  0.100,   1.0),  # high Pe, mid range
    ('hard_mid_Pe_lo',370.0, 290.0,  0.010, 100.0),  # low Pe, mid range
    ('hard_Tc_max',   400.0, 300.0,  0.100,   1.0),  # max T_cold + max Pe
    ('hard_Tc_min',   340.0, 280.0,  0.010, 100.0),  # min T_cold + min Pe
    ('hard_mid_Pe_m', 400.0, 280.0,  0.055,   1.0),  # intermediate Pe, max DT
]


def generate_hard_test_cases(output_dir, n_sensors=16, seed=999):
    """
    Generate and save the 10 hard/extreme test cases.

    These cases are saved directly in the test/ folder alongside
    the regular LHS samples. They are named 'hard_*.mat' to distinguish
    them from regular samples ('sample_*.mat'). This allows reporting
    two sub-scores from the same test set:
        - LHS samples  (500, named sample_*.mat) : general performance
        - Hard samples (50,  named hard_*.mat)   : robustness to extremes

    Each hard case is saved with 5 different random sensor placements
    (seeds 0-4) to get statistically stable error estimates despite
    only having 10 configurations.

    Args:
        output_dir (str): root dataset directory (same as generate_dataset)
        n_sensors  (int): number of sensors per sample
        seed       (int): base random seed
    """
    hard_dir = os.path.join(output_dir, 'test')
    # Hard cases are saved directly in test/ folder
    # Named 'hard_*.mat' to distinguish from LHS samples ('sample_*.mat')

    print("\n" + "=" * 62)
    print("  Generating Hard/Extreme Test Cases")
    print(f"  {len(HARD_CONFIGS)} configurations x 5 sensor placements")
    print(f"  = {len(HARD_CONFIGS) * 5} hard test samples")
    print("=" * 62)

    sample_idx = 0
    summary    = []

    for cfg_name, T_in, T_cold, u0, k in HARD_CONFIGS:
        Pe = RHO_CP * u0 * DX / k

        print(f"\n  {cfg_name}")
        print(f"    T_in={T_in:.0f}K  T_cold={T_cold:.0f}K  "
              f"u0={u0:.3f}m/s  k={k:.0f}W/mK  Pe={Pe:.1f}")

        # Solve PDE once for this configuration
        T_field = solve_hex(T_in, T_cold, u0, k)
        print(f"    T_field: [{T_field.min():.2f}, {T_field.max():.2f}] K")

        # Save 5 replicates with different sensor placements
        for replicate in range(5):
            np.random.seed(seed + sample_idx)
            u_obs, u_pos = place_sensors(T_field, n_sensors=n_sensors)
            params_dict  = {'T_in': T_in, 'T_cold': T_cold,
                            'u0': u0, 'k': k}
            X_input = encode_sparse_image(u_obs, u_pos, params_dict)

            filename = f'{cfg_name}_r{replicate:02d}.mat'
            sio.savemat(
                os.path.join(hard_dir, filename),
                {
                    'T_field'  : T_field,
                    'X_input'  : X_input,
                    'u_obs'    : u_obs,
                    'u_pos'    : u_pos,
                    'params'   : np.array([T_in, T_cold, u0, k],
                                          dtype=np.float32),
                    'Pe'       : np.float32(Pe),
                    'cfg_name' : cfg_name,
                    'replicate': np.int32(replicate),
                }
            )
            sample_idx += 1

        summary.append({
            'name'  : cfg_name,
            'T_in'  : T_in, 'T_cold': T_cold,
            'u0'    : u0,   'k'     : k,
            'Pe'    : Pe,
            'DeltaT': T_in - T_cold,
            'T_min' : float(T_field.min()),
            'T_max' : float(T_field.max()),
        })

    print(f"\n  Saved {sample_idx} hard test samples to: {hard_dir}")
    print("\n  Summary:")
    print(f"  {'Name':<22} {'Pe':>8} {'DeltaT':>8} {'T_min':>7} {'T_max':>7}")
    print("  " + "-" * 58)
    for s in summary:
        print(f"  {s['name']:<22} {s['Pe']:>8.1f} {s['DeltaT']:>8.1f} "
              f"{s['T_min']:>7.1f} {s['T_max']:>7.1f}")

    return summary


def visualize_hard_cases(output_dir):
    """
    Plot all 10 hard configurations side by side for paper discussion.
    Shows how field complexity varies with Pe and DeltaT.
    """
    hard_dir = os.path.join(output_dir, 'test')
    n_cases  = len(HARD_CONFIGS)
    extent   = [0, LX, LY, 0]

    fig, axes = plt.subplots(2, n_cases // 2, figsize=(n_cases * 2, 9))
    axes = axes.ravel()

    fig.suptitle(
        'Hard/Extreme Test Cases — Temperature Fields\n'
        'Used to evaluate model robustness beyond the LHS test set',
        fontsize=11, fontweight='bold'
    )

    for i, (cfg_name, T_in, T_cold, u0, k) in enumerate(HARD_CONFIGS):
        # Load replicate 0
        path = os.path.join(hard_dir, f'{cfg_name}_r00.mat')
        data = sio.loadmat(path)
        T    = data['T_field'].squeeze()
        Pe   = float(data['Pe'])
        pos  = data['u_pos'].squeeze()
        obs  = data['u_obs'].squeeze()

        ax = axes[i]
        im = ax.imshow(T, cmap=CMAP_FIELD, origin='upper',
                       extent=extent, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.7, label='T (K)')

        # Sensor positions
        sx = pos[:, 1] * DX; sy = pos[:, 0] * DY
        ax.scatter(sx, sy, c='white', s=20, edgecolors='black',
                   lw=0.5, zorder=5)

        # Pass boundaries
        for y_bnd in [LY/3, 2*LY/3]:
            ax.axhline(y_bnd, color='white', lw=0.8, ls='--', alpha=0.6)

        ax.set_title(
            f'{cfg_name.replace("hard_", "")}\n'
            f'Pe={Pe:.0f}  ΔT={T_in-T_cold:.0f}K\n'
            f'[{T.min():.0f},{T.max():.0f}]K',
            fontsize=7.5
        )
        ax.set_xlabel('x (m)', fontsize=7)
        if i % (n_cases // 2) == 0:
            ax.set_ylabel('y (m)', fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'hard_test_cases.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Hard cases figure saved: {path}")


# ===============================================================================
# SECTION 10 -- VISUALIZATION
# ===============================================================================

def visualize_sample(sample_path, save=True):
    """
    Visualize one sample with 3 panels.

    Panel 1: Full 2D temperature field (ground truth)
             Colormap: RdYlBu_r (blue=cold, yellow/red=hot)
             following KYA Figure 1 convention.

    Panel 2: Prescribed velocity field showing the 3-pass structure.

    Panel 3: Sparse sensor positions overlaid on the temperature field,
             colored by their measured temperature value.

    Args:
        sample_path (str) : path to a .mat file
        save        (bool): save figure as .png next to the .mat file
    """
    data  = sio.loadmat(sample_path)
    T     = data['T_field'].squeeze()
    pos   = data['u_pos'].squeeze()
    obs   = data['u_obs'].squeeze()
    p     = data['params'].squeeze()
    T_in, T_cold, u0, k = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    Pe    = RHO_CP * u0 * DX / k

    ux, _ = build_velocity_field(u0)
    extent = [0, LX, LY, 0]   # [xmin, xmax, ymax, ymin] for imshow

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.suptitle(
        f'HEX Sample  |  T_in={T_in:.0f}K   T_cold={T_cold:.0f}K   '
        f'u0={u0:.4f}m/s   k={k:.1f}W/(m.K)   Pe={Pe:.2f}',
        fontsize=11, fontweight='bold'
    )

    # ── Panel 1: Ground truth temperature field ────────────────────────────
    im1 = axes[0].imshow(T, cmap=CMAP_FIELD, origin='upper',
                         extent=extent, aspect='auto',
                         vmin=T.min(), vmax=T.max())
    axes[0].set_title(f'Ground truth temperature field\n'
                      f'min={T.min():.1f}K   max={T.max():.1f}K',
                      fontsize=9)
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    cb1 = plt.colorbar(im1, ax=axes[0], label='T (K)')
    # Pass boundaries
    for y_bnd in [LY/3, 2*LY/3]:
        axes[0].axhline(y_bnd, color='white', lw=1.2, ls='--', alpha=0.6)

    # ── Panel 2: Velocity field (pass structure) ────────────────────────────
    axes[1].imshow(ux, cmap='bwr', origin='upper',
                   extent=extent, aspect='auto',
                   vmin=-u0 * 1.1, vmax=u0 * 1.1)
    for y_bnd in [LY/3, 2*LY/3]:
        axes[1].axhline(y_bnd, color='black', lw=1.5, ls='--', alpha=0.7)
    axes[1].text(0.01, LY*0.17, 'Pass 1 -->', color='white', fontsize=8)
    axes[1].text(0.01, LY*0.50, '<-- Pass 2', color='white', fontsize=8)
    axes[1].text(0.01, LY*0.83, 'Pass 3 -->', color='white', fontsize=8)
    axes[1].set_title('Prescribed velocity field\nBlue = right (+u0)  '
                      'Red = left (-u0)', fontsize=9)
    axes[1].set_xlabel('x (m)')

    # ── Panel 3: Sensor positions overlaid on temperature field ─────────────
    axes[2].imshow(T, cmap=CMAP_FIELD, origin='upper',
                   extent=extent, aspect='auto', alpha=0.75)
    for y_bnd in [LY/3, 2*LY/3]:
        axes[2].axhline(y_bnd, color='white', lw=1.2, ls='--', alpha=0.6)

    sx = pos[:, 1] * DX   # col index -> x position [m]
    sy = pos[:, 0] * DY   # row index -> y position [m]
    sc = axes[2].scatter(sx, sy, c=obs, cmap=CMAP_FIELD,
                         s=80, edgecolors='black', linewidths=0.8,
                         zorder=5, label=f'M={len(pos)} sensors',
                         vmin=T.min(), vmax=T.max())
    axes[2].set_title(f'Sparse sensor observations\n'
                      f'Strategy A -- model input  (M={len(pos)})',
                      fontsize=9)
    axes[2].set_xlabel('x (m)')
    axes[2].legend(fontsize=8, loc='lower right')
    plt.colorbar(sc, ax=axes[2], label='T_sensor (K)')

    plt.tight_layout()
    if save:
        out = sample_path.replace('.mat', '_viz.png')
        plt.savefig(out, dpi=130, bbox_inches='tight')
        print(f"Figure saved: {out}")
    plt.show()


def visualize_dataset_overview(output_dir, n_show=6):
    """
    Show n_show samples from the training set side by side.
    Illustrates the diversity of thermal fields across parameter configurations.
    """
    train_dir = os.path.join(output_dir, 'train')
    files     = sorted(os.listdir(train_dir))
    indices   = np.linspace(0, len(files) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 3.2, 6))
    fig.suptitle(
        'HEX Dataset Overview -- Training samples\n'
        'Colormap: RdYlBu_r (blue=cold, yellow/red=hot)',
        fontsize=11, fontweight='bold'
    )
    extent = [0, LX, LY, 0]

    for col, idx in enumerate(indices):
        data    = sio.loadmat(os.path.join(train_dir, files[idx]))
        T       = data['T_field'].squeeze()
        pos     = data['u_pos'].squeeze()
        obs     = data['u_obs'].squeeze()
        p       = data['params'].squeeze()
        Pe      = RHO_CP * p[2] * DX / p[3]

        # Row 0: temperature field
        im = axes[0, col].imshow(T, cmap=CMAP_FIELD, origin='upper',
                                  extent=extent, aspect='auto')
        axes[0, col].set_title(
            f'T_in={p[0]:.0f}K\nu0={p[2]:.3f}  Pe={Pe:.1f}',
            fontsize=7.5
        )
        for y_bnd in [LY/3, 2*LY/3]:
            axes[0, col].axhline(y_bnd, color='w', lw=1, ls='--', alpha=0.5)
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)

        # Row 1: sensor positions overlay
        axes[1, col].imshow(T, cmap=CMAP_FIELD, origin='upper',
                             extent=extent, aspect='auto', alpha=0.7)
        sx = pos[:, 1] * DX; sy = pos[:, 0] * DY
        axes[1, col].scatter(sx, sy, c=obs, cmap=CMAP_FIELD, s=50,
                              edgecolors='black', lw=0.6, zorder=5,
                              vmin=T.min(), vmax=T.max())
        for y_bnd in [LY/3, 2*LY/3]:
            axes[1, col].axhline(y_bnd, color='w', lw=1, ls='--', alpha=0.5)
        axes[1, col].set_title(f'M={len(pos)} sensors', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(output_dir, 'dataset_overview.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Overview saved: {path}")


def print_dataset_info(output_dir):
    """Print dataset structure and one sample statistics."""
    print("\n=== HEX DATASET STRUCTURE ===")
    for split in ['train', 'val', 'test']:
        folder = os.path.join(output_dir, split)
        all_files  = [f for f in os.listdir(folder) if f.endswith('.mat')]
        lhs_files  = [f for f in all_files if f.startswith('sample_')]
        hard_files = [f for f in all_files if f.startswith('hard_')]
        if hard_files:
            print(f"  {split:6s} : {len(all_files)} samples "
                  f"({len(lhs_files)} LHS + {len(hard_files)} hard)")
        else:
            print(f"  {split:6s} : {len(all_files)} samples")

    path = os.path.join(output_dir, 'train', 'sample_0000.mat')
    data = sio.loadmat(path)
    print("\nSample structure (train/sample_0000.mat):")
    for key, val in data.items():
        if not key.startswith('_'):
            print(f"  {key:10s} | shape={str(val.shape):16s} | "
                  f"dtype={str(val.dtype):10s} | "
                  f"min={float(val.min()):8.2f}  "
                  f"max={float(val.max()):8.2f}")
    print("=" * 44)


# ===============================================================================
# MAIN
# ===============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    OUTPUT_DIR = r'D:\PaperKya\HEX_dataset'

    # Step 1: Generate the main dataset (5000 LHS samples)
    generate_dataset(
        n_total    = 5000,
        n_sensors  = 16,
        output_dir = OUTPUT_DIR,
        seed       = 42,
    )

    # Step 2: Generate hard/extreme test cases and append to test/ folder
    generate_hard_test_cases(
        output_dir = OUTPUT_DIR,
        n_sensors  = 16,
        seed       = 999,
    )

    # Step 3: Print dataset structure
    print_dataset_info(OUTPUT_DIR)

    # Step 4: Visualize one sample from each split
    for split in ['train', 'val', 'test']:
        visualize_sample(
            os.path.join(OUTPUT_DIR, split, 'sample_0000.mat')
        )

    # Step 5: Visualize all hard test cases
    visualize_hard_cases(OUTPUT_DIR)

    # Step 6: Overview of 6 diverse training samples
    visualize_dataset_overview(OUTPUT_DIR, n_show=6)
