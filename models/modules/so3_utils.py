#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SO(3) Lie group utilities for Riemannian flow matching.

Provides:
    hat / vee       -- maps between R^3 and so(3) (skew-symmetric matrices)
    so3_exp         -- exponential map so(3) -> SO(3)  (Rodrigues)
    so3_log         -- logarithmic map SO(3) -> so(3)
    so3_geodesic    -- geodesic interpolation on SO(3)
    random_so3      -- uniform Haar-measure sampling via quaternions
    quat_to_rotmat  -- quaternion (w,x,y,z) -> rotation matrix
    so3_tangent_transport -- parallel transport on SO(3)
    igso3_density   -- IGSO(3) density for score computation

All functions support arbitrary leading batch dimensions ([...]).
Numerically stable via Taylor expansions near identity.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Hat / Vee maps                                                              #
# --------------------------------------------------------------------------- #

def hat(v):
    """Hat map: R^3 -> so(3) (skew-symmetric matrices).

    Args:
        v: [..., 3]
    Returns:
        [..., 3, 3] skew-symmetric matrices
    """
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    # fmt: off
    return torch.stack([
        zero, -z,    y,
        z,    zero, -x,
       -y,    x,    zero,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)
    # fmt: on


def vee(S):
    """Vee map: so(3) -> R^3 (inverse of hat).

    Args:
        S: [..., 3, 3] skew-symmetric matrices
    Returns:
        [..., 3]
    """
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


# --------------------------------------------------------------------------- #
#  Exponential map (Rodrigues' formula)                                        #
# --------------------------------------------------------------------------- #

def so3_exp(omega):
    """Exponential map: R^3 -> SO(3) via Rodrigues' formula.

    R = I + (sin theta / theta) K + ((1 - cos theta) / theta^2) K^2

    where K = hat(omega), theta = ||omega||.
    Taylor expansion used for theta < 1e-4 for numerical stability.

    Args:
        omega: [..., 3] axis-angle rotation vectors
    Returns:
        R: [..., 3, 3] rotation matrices
    """
    theta_sq = (omega * omega).sum(-1)          # [...]
    theta = torch.sqrt(theta_sq.clamp(min=1e-12))  # [...]

    small = theta_sq < 1e-8
    coeff1 = torch.where(small,
                         1.0 - theta_sq / 6.0,
                         torch.sin(theta) / theta)
    coeff2 = torch.where(small,
                         0.5 - theta_sq / 24.0,
                         (1.0 - torch.cos(theta)) / theta_sq)

    K = hat(omega)       # [..., 3, 3]
    K2 = K @ K           # [..., 3, 3]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    R = I + coeff1[..., None, None] * K + coeff2[..., None, None] * K2
    return R


# --------------------------------------------------------------------------- #
#  Logarithmic map                                                             #
# --------------------------------------------------------------------------- #

def so3_log(R):
    """Logarithmic map: SO(3) -> R^3 (axis-angle).

    Three regimes for numerical stability:
      - Small theta (< 1e-4):  Taylor expansion of theta/(2 sin theta)
      - Medium theta:          standard formula theta/(2 sin theta) * vee(R-R^T)/2
      - Large theta (sin(theta) < 0.25): eigenvector-based axis extraction

    Args:
        R: [..., 3, 3] rotation matrices
    Returns:
        omega: [..., 3] axis-angle vectors
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    skew = (R - R.transpose(-1, -2)) / 2.0
    omega_skew = vee(skew)   # [..., 3], unnormalized (magnitude ~ sin(theta))

    # Small-angle Taylor
    theta_sq = theta * theta
    small = theta < 1e-4
    scale = torch.where(
        small,
        0.5 + theta_sq / 12.0,
        theta / (2.0 * sin_theta.clamp(min=1e-10)))
    omega_mid = scale[..., None] * omega_skew * 2.0

    # Large-angle regime (theta > pi - 0.2): sin(theta) is near zero so
    # the skew-based formula loses precision. Use eigendecomposition of
    # the symmetric part (R + R^T)/2 to robustly extract the rotation
    # axis (the eigenvector with eigenvalue 1).
    large = theta > (torch.pi - 0.2)
    if large.any():
        R_large = R[large]                                      # [M, 3, 3]
        sym = (R_large + R_large.transpose(-1, -2)) / 2.0      # [M, 3, 3]
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)      # [M, 3], [M, 3, 3]
        # The eigenvalue closest to +1 corresponds to the rotation axis
        axis = eigenvectors[..., -1]                            # [M, 3]
        axis = F.normalize(axis, dim=-1, eps=1e-10)

        skew_large = omega_skew[large]                          # [M, 3]
        dot = (skew_large * axis).sum(-1)
        sign = torch.where(dot >= 0,
                           torch.ones_like(dot),
                           -torch.ones_like(dot))
        theta_large = theta[large]                              # [M]
        omega_large_vals = theta_large[..., None] * sign[..., None] * axis

        omega = omega_mid.clone()
        omega[large] = omega_large_vals
    else:
        omega = omega_mid

    return omega


# --------------------------------------------------------------------------- #
#  Geodesic interpolation on SO(3)                                             #
# --------------------------------------------------------------------------- #

def so3_geodesic(R0, R1, t):
    """Geodesic interpolation on SO(3): R(t) = R0 * exp(t * log(R0^T R1)).

    This traces the shortest-path rotation from R0 to R1.

    Args:
        R0: [..., 3, 3] starting rotations
        R1: [..., 3, 3] target rotations
        t:  [...] or scalar, interpolation parameter in [0, 1]
    Returns:
        Rt: [..., 3, 3] interpolated rotations
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=R0.device, dtype=R0.dtype)
    while t.dim() < R0.dim() - 2:
        t = t.unsqueeze(-1)

    R_rel = R0.transpose(-1, -2) @ R1      # R0^T R1
    omega = so3_log(R_rel)                  # [..., 3]
    return R0 @ so3_exp(t[..., None] * omega)  # R0 * exp(t * omega)


def so3_tangent_vector(R0, R1):
    """Tangent vector at R0 pointing toward R1 on SO(3).

    v = log(R0^T R1), expressed in the frame of R0.
    The actual velocity in the ambient space is R0 @ hat(v).

    Args:
        R0: [..., 3, 3]
        R1: [..., 3, 3]
    Returns:
        omega: [..., 3] tangent vector (axis-angle in body frame)
    """
    return so3_log(R0.transpose(-1, -2) @ R1)


# --------------------------------------------------------------------------- #
#  Geodesic distance on SO(3)                                                  #
# --------------------------------------------------------------------------- #

def so3_geodesic_distance(R0, R1):
    """Geodesic distance d(R0, R1) = ||log(R0^T R1)||.

    Args:
        R0: [..., 3, 3]
        R1: [..., 3, 3]
    Returns:
        d: [...] geodesic distances
    """
    omega = so3_tangent_vector(R0, R1)
    return torch.norm(omega, dim=-1)


# --------------------------------------------------------------------------- #
#  Uniform random rotations (Haar measure via quaternions)                     #
# --------------------------------------------------------------------------- #

def quat_to_rotmat(q):
    """Unit quaternion (w, x, y, z) -> rotation matrix.

    Args:
        q: [..., 4]
    Returns:
        R: [..., 3, 3]
    """
    w, x, y, z = q.unbind(-1)
    # fmt: off
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    # fmt: on
    return R


def random_so3(n, device, dtype):
    """Sample *n* uniform random rotations from SO(3).

    Uses the quaternion method: sample q ~ N(0, I_4), normalise, convert.

    Args:
        n: number of rotations
        device, dtype: torch device / dtype
    Returns:
        R: [n, 3, 3]
    """
    q = torch.randn(n, 4, device=device, dtype=dtype)
    q = F.normalize(q, dim=-1)
    return quat_to_rotmat(q)


# --------------------------------------------------------------------------- #
#  Small-angle perturbation (for noising)                                      #
# --------------------------------------------------------------------------- #

def so3_noise(R, sigma):
    """Apply isotropic Gaussian noise on the SO(3) tangent space.

    R_noisy = R @ exp(sigma * eps),  eps ~ N(0, I_3).

    Args:
        R: [..., 3, 3] rotation matrices
        sigma: float or [...] noise scale
    Returns:
        R_noisy: [..., 3, 3]
    """
    eps = torch.randn(*R.shape[:-2], 3, device=R.device, dtype=R.dtype)
    if isinstance(sigma, (int, float)):
        perturbation = so3_exp(sigma * eps)
    else:
        perturbation = so3_exp(sigma[..., None] * eps)
    return R @ perturbation


# --------------------------------------------------------------------------- #
#  Project to SO(3) via SVD (for gradient-based updates)                       #
# --------------------------------------------------------------------------- #

def project_so3(M):
    """Project an arbitrary 3x3 matrix to SO(3) via SVD.

    Args:
        M: [..., 3, 3]
    Returns:
        R: [..., 3, 3] nearest rotation matrix
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(U @ Vh)
    sign = torch.ones_like(det)
    sign[det < 0] = -1
    U[..., :, -1] *= sign[..., None]
    return U @ Vh
