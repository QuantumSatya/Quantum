"""
Laser Polarization → Half-Wave Plate (HWP) Simulation (Jones Calculus)
----------------------------------------------------------------------
Offline simulation with a simple UI (matplotlib sliders) to vary:

  1) Input linear polarization angle α (deg)   (0° = Horizontal, 90° = Vertical)
  2) HWP fast-axis angle θ (deg)               (θ measured from Horizontal)

For each (α, θ) it displays:
  - HWP Jones matrix J_HWP(θ)
  - Input state |ψ_in> and output state |ψ_out> = J_HWP(θ)|ψ_in>
  - Whether the input state is an eigenstate of the HWP at that θ
  - If eigenstate: the corresponding eigenvalue (λ = +1 or λ = -1 in this ideal model)
  - The eigenvalues/eigenvectors of J_HWP(θ) for reference

Notes:
  - This uses the common ideal HWP model with retardance π:
        diag(1, -1)  in the fast/slow axes basis
    (Global phase is ignored; eigenvalues appear as ±1.)
  - A HWP does not absorb light, so intensity is conserved (||E||^2 stays constant)
    in this ideal model.

Requirements:
    python >= 3.8
    numpy
    matplotlib

Run:
    python laser_hwp_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def R(theta_rad: float) -> np.ndarray:
    """Rotation matrix that changes basis by angle theta (Jones calculus)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def hwp_matrix(theta_deg: float) -> np.ndarray:
    """
    Jones matrix for an ideal half-wave plate with fast axis at angle theta.
    J(θ) = R(θ) * diag(1, -1) * R(-θ)
    Equivalent to:
      [[ cos 2θ,  sin 2θ],
       [ sin 2θ, -cos 2θ]]
    """
    th = np.deg2rad(theta_deg)
    D = np.array([[1.0, 0.0],
                  [0.0, -1.0]], dtype=float)
    return R(th) @ D @ R(-th)


def linpol_state(alpha_deg: float) -> np.ndarray:
    """Jones vector for linear polarization at angle alpha (0°=H, 90°=V)."""
    a = np.deg2rad(alpha_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def eigen_test(J: np.ndarray, psi: np.ndarray, tol: float = 1e-9):
    """
    Test if psi is an eigenvector of J, i.e., J psi = λ psi for some scalar λ.
    Returns: (is_eigen, lambda_est or None)
    """
    out = J @ psi
    # If input is ~zero (should not happen here), return false
    if np.linalg.norm(psi) < 1e-12:
        return False, None

    # Check proportionality using 2D determinant |psi, out| = 0 for collinearity
    det = psi[0]*out[1] - psi[1]*out[0]
    if abs(det) > tol:
        return False, None

    # Estimate λ from a stable component
    lam = None
    if abs(psi[0]) >= abs(psi[1]) and abs(psi[0]) > tol:
        lam = out[0] / psi[0]
    elif abs(psi[1]) > tol:
        lam = out[1] / psi[1]

    if lam is None:
        return False, None

    return True, float(lam)


def fmt(x: float) -> str:
    return "0.0000" if abs(x) < 1e-12 else f"{x: .4f}"


def draw_vector(ax, v, color, label):
    v = unit(v)
    ax.annotate(
        "",
        xy=(v[0], v[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=3, color=color, shrinkA=0, shrinkB=0),
    )
    ax.text(v[0]*1.08, v[1]*1.08, label, color=color, ha="center", va="center", fontsize=10)


def main():
    fig = plt.figure(figsize=(7, 7))

    # Axes layout
    ax_geom = fig.add_axes([0.06, 0.24, 0.46, 0.70])
    ax_text = fig.add_axes([0.55, 0.24, 0.40, 0.70])
    ax_text.axis("off")

    # Slider axes
    ax_sl1 = fig.add_axes([0.3, 0.12, 0.50, 0.05])
    ax_sl2 = fig.add_axes([0.3, 0.06, 0.50, 0.05])

    alpha_slider = Slider(ax_sl1, r"Input linear polarization  $\alpha$ (deg)", 0.0, 360.0,
                          valinit=0.0, valstep=0.5)
    theta_slider = Slider(ax_sl2, r"HWP fast-axis angle  $\theta$ (deg)", 0.0, 180.0,
                          valinit=0.0, valstep=0.5)

    # Precompute circle
    ang = np.linspace(0, 2*np.pi, 400)
    txt = ax_text.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)

    def render(alpha_deg: float, theta_deg: float):
        # Geometry panel
        ax_geom.cla()
        ax_geom.set_title("Geometry view (Jones vectors on Ex–Ey plane)")
        ax_geom.set_xlabel("Ex")
        ax_geom.set_ylabel("Ey")
        ax_geom.set_aspect("equal", adjustable="box")
        ax_geom.set_xlim(-1.25, 1.25)
        ax_geom.set_ylim(-1.25, 1.25)
        for spine in ax_geom.spines.values():
            spine.set_visible(False)

        ax_geom.set_xticks([])
        ax_geom.set_yticks([])
        ax_geom.grid(True, alpha=0.1)
        ax_geom.plot(np.cos(ang), np.sin(ang), lw=1.8)
        ax_geom.axhline(0, lw=1.0)
        ax_geom.axvline(0, lw=1.0)

        psi_in = linpol_state(alpha_deg)
        J = hwp_matrix(theta_deg)
        psi_out = J @ psi_in

        I_in = float(psi_in @ psi_in)
        I_out = float(psi_out @ psi_out)

        # HWP axis directions
        th = np.deg2rad(theta_deg)
        fast_axis = np.array([np.cos(th), np.sin(th)])
        slow_axis = np.array([-np.sin(th), np.cos(th)])  # orthogonal

        # Draw vectors
        draw_vector(ax_geom, psi_in, "tab:blue", f"Input α={alpha_deg:.0f}°")
        draw_vector(ax_geom, fast_axis, "tab:green", f"Fast axis θ={theta_deg:.0f}°")
        draw_vector(ax_geom, psi_out, "tab:orange", "Output")

        # Eigen test for input state
        is_eig, lam = eigen_test(J, psi_in)

        if is_eig:
            # In the ideal model, λ should be close to +1 or -1
            # Snap for neat reporting
            lam_snap = 1.0 if lam >= 0 else -1.0
            eig_line = f"Input is an eigenstate? YES, eigenvalue λ ≈ {lam: .4f} (ideal: {lam_snap:+.0f})"
        else:
            eig_line = "Input is an eigenstate? NO (output is not proportional to input)"

        # Compute eigen-decomposition of J for reference
        evals, evecs = np.linalg.eig(J)
        # Normalize eigenvectors for nicer display
        evecs = np.array([unit(evecs[:,0].real), unit(evecs[:,1].real)]).T

        text_block = (
            f"Input |ψ_in> (linear α) = [cos α, sin α]^T\n"
            f"α = {alpha_deg:.0f}°\n"
            f"|ψ_in> = [ {fmt(psi_in[0])}, {fmt(psi_in[1])} ]^T\n\n"
            f"HWP angle θ = {theta_deg:.0f}°\n"
            f"J_HWP(θ) =\n"
            f"[ {fmt(J[0,0])}   {fmt(J[0,1])} ]\n"
            f"[ {fmt(J[1,0])}   {fmt(J[1,1])} ]\n\n"
            f"Output |ψ_out> = J|ψ_in> =\n"
            f"[ {fmt(psi_out[0])}, {fmt(psi_out[1])} ]^T\n\n"
            f"Intensity check (ideal HWP):\n"
            f"||ψ_in||^2  = {I_in:.4f}\n"
            f"||ψ_out||^2 = {I_out:.4f}\n\n"
            f"{eig_line}\n\n"
            f"Eigenvalues/eigenvectors of J_HWP(θ):\n"
            f"λ1 = {evals[0]: .4f}, v1 = [ {fmt(evecs[0,0])}, {fmt(evecs[1,0])} ]^T\n"
            f"λ2 = {evals[1]: .4f}, v2 = [ {fmt(evecs[0,1])}, {fmt(evecs[1,1])} ]^T\n\n"
            f"Interpretation:\n"
            f"- The eigenvectors are along the fast/slow axes (two orthogonal directions).\n"
            f"- If your input matches one of those axes, it is an eigenstate (only a sign/phase changes).\n"
        )
        txt.set_text(text_block)
        fig.canvas.draw_idle()

    # Initial render
    render(alpha_slider.val, theta_slider.val)

    def on_change(_):
        render(alpha_slider.val, theta_slider.val)

    alpha_slider.on_changed(on_change)
    theta_slider.on_changed(on_change)

    fig.suptitle("Laser Polarization → Half-Wave Plate (HWP): Eigenstates/Eigenvalues (Simulation)",
                 y=0.98, fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
