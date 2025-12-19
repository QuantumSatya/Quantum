
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

# ----------------------------
# Utilities
# ----------------------------

def normalize(v):
    v = np.array(v, dtype=float).reshape(3,)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / nrm


def bloch_from_theta_phi_deg(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([
        np.sin(th) * np.cos(ph),
        np.sin(th) * np.sin(ph),
        np.cos(th)
    ], dtype=float)


def add_3d_arrow(ax, start, end, arrow_length_ratio=0.12, **kwargs):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    vec = end - start
    return ax.quiver(
        start[0], start[1], start[2],
        vec[0], vec[1], vec[2],
        arrow_length_ratio=arrow_length_ratio,
        linewidth=1.4,
        **kwargs
    )


def add_label_near_point(ax, point, text, offset=0.10, **kwargs):
    p = np.array(point, dtype=float)
    p_off = p + offset * normalize(p)
    return ax.text(p_off[0], p_off[1], p_off[2], text, **kwargs)


def great_circle_arc(p_from, p_to, num=80):
    """
    Points on the shortest great-circle arc from p_from to p_to (both on unit sphere).
    Uses spherical linear interpolation (slerp).
    """
    p_from = normalize(p_from)
    p_to = normalize(p_to)
    dot = float(np.clip(np.dot(p_from, p_to), -1.0, 1.0))
    ang = np.arccos(dot)

    if ang < 1e-10:
        return np.repeat(p_from.reshape(1, 3), num, axis=0)

    t = np.linspace(0.0, 1.0, num)
    s1 = np.sin((1 - t) * ang) / np.sin(ang)
    s2 = np.sin(t * ang) / np.sin(ang)
    pts = (s1[:, None] * p_from[None, :]) + (s2[:, None] * p_to[None, :])
    # Numerical guard
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


# ----------------------------
# Initial state setup (degrees)
# ----------------------------

theta_deg = 60.0
phi_deg = 45.0
r0 = bloch_from_theta_phi_deg(theta_deg, phi_deg)

# ----------------------------
# Figure & axes
# ----------------------------

fig = plt.figure(figsize=(8.5, 6.5))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.22)

# ----------------------------
# Draw Bloch sphere (clean)
# ----------------------------

def draw_bloch_sphere(ax):
    u = np.linspace(0, np.pi, 90)
    v = np.linspace(0, 2 * np.pi, 90)
    x = np.outer(np.sin(u), np.cos(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    ax.plot_surface(x, y, z, alpha=0.12, edgecolor="none")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

draw_bloch_sphere(ax)

# ----------------------------
# Basis labels on axes
# ----------------------------

add_label_near_point(ax, [0, 0, 1], r"$\left|0\right\rangle$", fontsize=12)
add_label_near_point(ax, [0, 0, -1], r"$\left|1\right\rangle$", fontsize=12)

add_label_near_point(ax, [1, 0, 0], r"$\frac{\left|0\right\rangle+\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)
add_label_near_point(ax, [-1, 0, 0], r"$\frac{\left|0\right\rangle-\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)

add_label_near_point(ax, [0, 1, 0], r"$\frac{\left|0\right\rangle+i\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)
add_label_near_point(ax, [0, -1, 0], r"$\frac{\left|0\right\rangle-i\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)

# ----------------------------
# FULL axes with arrowheads at both ends
# ----------------------------

# X
ax.plot([-1, 1], [0, 0], [0, 0], lw=1.2, color="black")
add_3d_arrow(ax, [0.85, 0, 0], [1.0, 0, 0], arrow_length_ratio=0.6, color="black")
add_3d_arrow(ax, [-0.85, 0, 0], [-1.0, 0, 0], arrow_length_ratio=0.6, color="black")

# Y
ax.plot([0, 0], [-1, 1], [0, 0], lw=1.2, color="black")
add_3d_arrow(ax, [0, 0.85, 0], [0, 1.0, 0], arrow_length_ratio=0.6, color="black")
add_3d_arrow(ax, [0, -0.85, 0], [0, -1.0, 0], arrow_length_ratio=0.6, color="black")

# Z
ax.plot([0, 0], [0, 0], [-1, 1], lw=1.2, color="black")
add_3d_arrow(ax, [0, 0, 0.85], [0, 0, 1.0], arrow_length_ratio=0.6, color="black")
add_3d_arrow(ax, [0, 0, -0.85], [0, 0, -1.0], arrow_length_ratio=0.6, color="black")

# ----------------------------
# Dotted great-circle connections between ALL axis tips
# (+/-x, +/-y, +/-z) connect each-to-each on the sphere surface
# ----------------------------

axis_tips = [
    np.array([ 1.0, 0.0, 0.0]),  # +x
    np.array([-1.0, 0.0, 0.0]),  # -x
    np.array([ 0.0, 1.0, 0.0]),  # +y
    np.array([ 0.0,-1.0, 0.0]),  # -y
    np.array([ 0.0, 0.0, 1.0]),  # +z
    np.array([ 0.0, 0.0,-1.0]),  # -z
]

# Draw all pairwise great-circle arcs as dotted lines
for i in range(len(axis_tips)):
    for j in range(i + 1, len(axis_tips)):
        a = axis_tips[i]
        b = axis_tips[j]
        pts = great_circle_arc(a, b, num=70)
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            linestyle=":", lw=0.5, color="black", alpha=0.9
        )

# ----------------------------
# State vector arrow (green)
# ----------------------------

bloch_quiver = add_3d_arrow(ax, [0, 0, 0], r0, color="green")

# ----------------------------
# Keep animation scaffold (no rotation; UI updates redraw state)
# ----------------------------

frames = 200

def init_anim():
    return tuple()

def update(_frame):
    return tuple()

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init_anim,
    blit=False,
    interval=40,
    repeat=True,
)

# ----------------------------
# UI: theta, phi in DEGREES
# ----------------------------

axbox_theta = plt.axes([0.12, 0.10, 0.33, 0.07])
axbox_phi   = plt.axes([0.52, 0.10, 0.33, 0.07])

tb_theta = TextBox(axbox_theta, r"$\theta$ (deg)", initial=f"{theta_deg:.3f}")
tb_phi   = TextBox(axbox_phi,   r"$\phi$ (deg)",   initial=f"{phi_deg:.3f}")

axbtn = plt.axes([0.82, 0.02, 0.14, 0.07])
btn = Button(axbtn, "Apply")


def safe_eval_float(expr: str) -> float:
    allowed = {"np": np}
    return float(eval(expr, {"__builtins__": {}}, allowed))


def apply_theta_phi(_event):
    global theta_deg, phi_deg, r0, bloch_quiver

    try:
        new_theta = safe_eval_float(tb_theta.text.strip())
        new_phi = safe_eval_float(tb_phi.text.strip())
    except Exception:
        return

    # Conventional ranges:
    # theta in [0, 180], phi in (-180, 180]
    new_theta = float(new_theta % 360.0)
    if new_theta > 180.0:
        new_theta = 360.0 - new_theta
        new_phi += 180.0

    new_phi = float(((new_phi + 180.0) % 360.0) - 180.0)

    theta_deg, phi_deg = new_theta, new_phi
    r0 = bloch_from_theta_phi_deg(theta_deg, phi_deg)

    # Update Bloch arrow
    try:
        bloch_quiver.remove()
    except Exception:
        pass
    bloch_quiver = add_3d_arrow(ax, [0, 0, 0], r0, color="green")

    fig.canvas.draw_idle()


btn.on_clicked(apply_theta_phi)

plt.show()

