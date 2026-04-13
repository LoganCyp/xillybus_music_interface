import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage import maximum_filter
from typing import Tuple

# Center frequency, update as necessary
FC = 2.8e9
C_LIGHT = 3e8
WAVELENGTH = C_LIGHT / FC

# Number of elements
NR = 8
ROWS = 2
COLS = 4
# Element spacing (meters)
D = 0.051         

# Scan grid
N_TH = 200
N_PH = 200
THETA_SCAN = np.linspace(-np.pi / 2, np.pi / 2, N_TH) # azimuth
PHI_SCAN   = np.linspace(-np.pi / 4, np.pi / 4, N_PH) # elevation

# Peak-picker neighbourhood (samples)
NEIGH = 5

# Element positions: x = col*d, y = 0, z = row*d  shape (NR, 3)
_idx = np.arange(NR)
POS  = np.stack([D * (_idx % COLS),
                 np.zeros(NR),
                 D * (_idx // COLS)], axis=1)

# Extract x, y, z for plotting
x = POS[:, 0]
y = POS[:, 1]
z = POS[:, 2]

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot points
# ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
ax.scatter(x, y, z)

# Labels
ax.set_xlabel('X (Cols)')
ax.set_ylabel('Y (Fixed at 0)')
ax.set_zlabel('Z (Rows)')
ax.set_title('3D Element Positions')

plt.show()

def _steering_matrix() -> np.ndarray:
    """
    Build the steering matrix for every (theta, phi) grid point.

    Returns
    -------
    A : (NR, N_TH * N_PH) complex128
        Each column is the steering vector for one grid point.
        Column order is row-major: index = i*N_PH + j.
    """
    th = THETA_SCAN[:, None]   # (N_TH, 1)
    ph = PHI_SCAN[None, :]     # (1, N_PH)

    # Direction cosines, flattened to (N_TH*N_PH,)
    ux = (np.sin(th) * np.cos(ph)).ravel()
    uy = (np.cos(th) * np.cos(ph)).ravel()
    uz = (np.ones_like(th) * np.sin(ph)).ravel()

    D_mat = np.stack([ux, uy, uz], axis=1)   # (N_TH*N_PH, 3)

    # POS @ D_mat.T  ->  (NR, N_TH*N_PH)
    proj = POS @ D_mat.T

    return np.exp(2j * np.pi * proj / WAVELENGTH)

_A = _steering_matrix()    # (NR, N_TH*N_PH)


def _estimate_d_sig(eigvals: np.ndarray) -> int:
    """
    Largest-gap heuristic on ascending eigenvalues.
    """
    diffs = np.diff(eigvals)
    print(diffs)
    k_inc = int(np.argmax(diffs))
    return eigvals.size - (k_inc + 1)


def _music_spectrum(Vn: np.ndarray) -> np.ndarray:
    """
    Compute the MUSIC pseudospectrum over the full scan grid.
    """
    # (n_noise, N_TH*N_PH) = Vn^H @ A
    VnH_a = Vn.conj().T @ _A

    # ||Vn^H a||^2 for each grid point -> (N_TH*N_PH,)
    denom = (VnH_a.real ** 2 + VnH_a.imag ** 2).sum(axis=0)

    return (1.0 / np.maximum(denom, 1e-30)).reshape(N_TH, N_PH)


def _pick_peaks(spectrum: np.ndarray,
                d_sig: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local-maximum filter + top-K selection.

    Returns
    -------
    angles     : (K, 2)  [[theta_deg, phi_deg], ...]
    powers_db  : (K,)    peak values in dB
    """
    local_max = spectrum == maximum_filter(spectrum, size=NEIGH)
    cand_idx  = np.argwhere(local_max)       # (M, 2)
    cand_vals = spectrum[local_max]          # (M,)

    order = np.argsort(cand_vals)[::-1]
    K = min(d_sig, len(order))
    top   = order[:K]

    ti = cand_idx[top, 0]
    pj = cand_idx[top, 1]

    angles = np.stack([np.degrees(THETA_SCAN[ti]),
                       np.degrees(PHI_SCAN[pj])], axis=1)   # (K, 2)

    powers_db = 10.0 * np.log10(cand_vals[top])             # (K,)

    return angles, powers_db

def run_music(
    R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Run the full MUSIC DOA pipeline on a covariance matrix.

    Parameters
    ----------
    R : (NR, NR) complex ndarray
        Upper-triangle covariance matrix as output by the FPGA.
        The lower triangle is zero; it is reconstructed internally via:
            R_full = R + R^H - diag(R)

    Returns
    -------
    angles      : (K, 2) float ndarray
                  Each row is [theta_deg, phi_deg]. K = d_sig.
                  Sorted by descending peak power.

    powers_db   : (K,) float ndarray
                  MUSIC pseudospectrum value at each peak [dB].

    d_sig       : int
                  Estimated number of impinging signals.

    spectrum    : (N_TH, N_PH) float ndarray
                  Full linear pseudospectrum over the scan grid.
                  Axes correspond to THETA_SCAN and PHI_SCAN.
    """
    R = np.asarray(R, dtype=complex)
    if R.shape != (NR, NR):
        raise ValueError(f"R must be ({NR}, {NR}), got {R.shape}")

    # Reconstruct full Hermitian matrix from upper triangle.
    # The FPGA only accumulates j >= i so the lower triangle is zeros.
    # R + R^H double-counts the diagonal, so subtract it once.
    R = R + R.conj().T - np.diag(np.diag(R))

    # Eigendecomposition - eigh guarantees real ascending eigenvalues
    eigvals, eigvecs = np.linalg.eigh(R)

    # Signal count via largest eigenvalue gap
    d_sig = _estimate_d_sig(eigvals)

    # Noise subspace: first (NR - d_sig) eigenvectors
    Vn = eigvecs[:, :NR - d_sig]                 # (NR, n_noise)

    # MUSIC pseudospectrum
    spectrum = _music_spectrum(Vn)               # (N_TH, N_PH)

    # Peak picking
    angles, powers_db = _pick_peaks(spectrum, d_sig)

    return angles, powers_db, d_sig, spectrum


# ---------------------------------------------------------------------------
# Convenience: build R from raw snapshot matrix
# ---------------------------------------------------------------------------

def covariance_from_snapshots(x: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix from a snapshot matrix.

    Parameters
    ----------
    x : (NR, N_snapshots) complex ndarray

    Returns
    -------
    R : (NR, NR) complex ndarray  (full Hermitian, not upper-triangle only)
    """
    x = np.asarray(x, dtype=complex)
    return (x @ x.conj().T) / x.shape[1]


if __name__ == "__main__":

    def _sv(th_deg, ph_deg):
        th, ph = np.radians(th_deg), np.radians(ph_deg)
        ux = np.sin(th) * np.cos(ph)
        uy = np.cos(th) * np.cos(ph)
        uz = np.sin(ph)
        proj = POS @ np.array([ux, uy, uz])
        return np.exp(2j * np.pi * proj / WAVELENGTH)

    a1, a2, a3 = _sv(20, 5), _sv(-15, -8), _sv(40, 12)
    R_full = (1.0 * np.outer(a1, a1.conj()) +
              0.8 * np.outer(a2, a2.conj()) +
              0.5 * np.outer(a3, a3.conj()) +
              0.05 * np.eye(NR))

    # Simulate FPGA output: upper triangle only
    R_upper = np.triu(R_full)

    angles, powers_db, d_sig, spectrum = run_music(R_upper)

    print(angles.shape)

    print(f"d_sig          : {d_sig}")
    print(f"spectrum shape : {spectrum.shape}")
    for k, (ang, pdb) in enumerate(zip(angles, powers_db), 1):
        print(f"  [{k}]  theta={ang[0]:7.2f} deg   phi={ang[1]:7.2f} deg   "
              f"P={pdb:.2f} dB")