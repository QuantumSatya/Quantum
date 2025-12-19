
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

# ----------------------------
# Utilities
# ----------------------------

def normalize(v):
    v = np.array(v, dtype=float).reshape(3,)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / norm


def r_of_t(t, r0, n, omega):
    """
    Rotate r0 about axis n by angle theta = omega*t (Rodrigues).
    Ensures r(t) is unit-length (on Bloch sphere surface).
    """
    theta = omega * t
    ct = np.cos(theta)
    st = np.sin(theta)

    n_dot_r0 = float(np.dot(n, r0))
    n_cross_r0 = np.cross(n, r0)

    r = ct * r0 + st * n_cross_r0 + (1.0 - ct) * n_dot_r0 * n
    return r / np.linalg.norm(r)


def add_3d_arrow(ax, start, end, arrow_length_ratio=0.12, **kwargs):
    """
    Draw a 3D arrow (quiver) from start to end.
    Returns the quiver artist.
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    vec = end - start
    q = ax.quiver(
        start[0], start[1], start[2],
        vec[0], vec[1], vec[2],
        arrow_length_ratio=arrow_length_ratio,
        linewidth=1.4,
        **kwargs
    )
    return q


def add_label_near_point(ax, point, text, offset=0.10, **kwargs):
    """
    Place 3D text slightly offset from a given point (to avoid overlap).
    """
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
# Initial physics setup
# ----------------------------

omega = 5.0

# Default axis n-hat and initial state
n = normalize([0.2, 0.7, 0.68])
r0 = normalize([1.0, 1.0, 1.0])  # pure state on surface

# ----------------------------
# Figure & axes
# ----------------------------

fig = plt.figure(figsize=(8.5, 6.5))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.22)  # room for UI

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

    # Remove grid/scale/box
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

draw_bloch_sphere(ax)

# ----------------------------
# Basis labels on axes
# ----------------------------

# +Z and -Z
add_label_near_point(ax, [0, 0, 1], r"$\left|0\right\rangle$", fontsize=12)
add_label_near_point(ax, [0, 0, -1], r"$\left|1\right\rangle$", fontsize=12)

# +X and -X
add_label_near_point(ax, [1, 0, 0], r"$\frac{\left|0\right\rangle+\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)
add_label_near_point(ax, [-1, 0, 0], r"$\frac{\left|0\right\rangle-\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)

# +Y and -Y
add_label_near_point(ax, [0, 1, 0], r"$\frac{\left|0\right\rangle+i\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)
add_label_near_point(ax, [0, -1, 0], r"$\frac{\left|0\right\rangle-i\left|1\right\rangle}{\sqrt{2}}$", fontsize=12)

# ----------------------------
# FULL axes arrows from - to + (with arrowheads at + ends)
# ----------------------------

# Draw full axis lines from -1 to +1
# X axis
ax.plot([-1, 1], [0, 0], [0, 0], lw=1.2, color="black")
# +X arrowhead
add_3d_arrow(ax, [0.85, 0, 0], [1.0, 0, 0], arrow_length_ratio=0.6, color="black")
# -X arrowhead
add_3d_arrow(ax, [-0.85, 0, 0], [-1.0, 0, 0], arrow_length_ratio=0.6, color="black")

# Y axis
ax.plot([0, 0], [-1, 1], [0, 0], lw=1.2, color="black")
# +Y arrowhead
add_3d_arrow(ax, [0, 0.85, 0], [0, 1.0, 0], arrow_length_ratio=0.6, color="black")
# -Y arrowhead
add_3d_arrow(ax, [0, -0.85, 0], [0, -1.0, 0], arrow_length_ratio=0.6, color="black")

# Z axis
ax.plot([0, 0], [0, 0], [-1, 1], lw=1.2, color="black")
# +Z arrowhead
add_3d_arrow(ax, [0, 0, 0.85], [0, 0, 1.0], arrow_length_ratio=0.6, color="black")
# -Z arrowhead
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
# Artists for animation
# ----------------------------

# n-hat arrow (blue) and label at the tip
n_quiver = add_3d_arrow(ax, [0, 0, 0], n, color="blue")
n_label = ax.text(n[0] * 1.12, n[1] * 1.12, n[2] * 1.12, r"$\hat{n}$", fontsize=12)

# Bloch vector arrow (green)
bloch_quiver = add_3d_arrow(ax, [0, 0, 0], r0, color="green")

# Trajectory line (pink)
traj_line, = ax.plot([], [], [], lw=1, color="deeppink")
traj_x, traj_y, traj_z = [], [], []


def reset_trajectory():
    traj_x.clear()
    traj_y.clear()
    traj_z.clear()
    traj_line.set_data([], [])
    traj_line.set_3d_properties([])


# Animation timing
T = 2 * np.pi
frames = 220
ts = np.linspace(0, T, frames)


def init_anim():
    reset_trajectory()
    return (traj_line,)


def update(frame):
    global bloch_quiver

    r = r_of_t(ts[frame], r0=r0, n=n, omega=omega)

    # Update Bloch arrow
    try:
        bloch_quiver.remove()
    except Exception:
        pass
    bloch_quiver = add_3d_arrow(ax, [0, 0, 0], r, color="green")

    # Trajectory
    traj_x.append(r[0])
    traj_y.append(r[1])
    traj_z.append(r[2])
    traj_line.set_data(traj_x, traj_y)
    traj_line.set_3d_properties(traj_z)

    return (traj_line, bloch_quiver)


ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init_anim,
    blit=False,          # more reliable for 3D + quiver
    interval=40,
    repeat=True,
)


# ----------------------------
# UI: choose n-hat components (nx, ny, nz)
#     and initial state r0 (r0x, r0y, r0z)
# ----------------------------

# n-hat text boxes (lower row)
axbox_nx = plt.axes([0.12, 0.10, 0.18, 0.06])
axbox_ny = plt.axes([0.37, 0.10, 0.18, 0.06])
axbox_nz = plt.axes([0.62, 0.10, 0.18, 0.06])

# r0 text boxes (upper row)
axbox_r0x = plt.axes([0.12, 0.20, 0.18, 0.06])
axbox_r0y = plt.axes([0.37, 0.20, 0.18, 0.06])
axbox_r0z = plt.axes([0.62, 0.20, 0.18, 0.06])

# TextBoxes for n-hat
tb_nx = TextBox(axbox_nx, "n_x", initial=f"{n[0]:.3f}")
tb_ny = TextBox(axbox_ny, "n_y", initial=f"{n[1]:.3f}")
tb_nz = TextBox(axbox_nz, "n_z", initial=f"{n[2]:.3f}")

# TextBoxes for r0
tb_r0x = TextBox(axbox_r0x, "r_x", initial=f"{r0[0]:.3f}")
tb_r0y = TextBox(axbox_r0y, "r_y", initial=f"{r0[1]:.3f}")
tb_r0z = TextBox(axbox_r0z, "r_z", initial=f"{r0[2]:.3f}")

# Apply button
axbtn = plt.axes([0.85, 0.16, 0.05, 0.05])
btn = Button(axbtn, "Go!")


def apply_nhat_and_r0(_event):
    global n, r0, n_quiver, bloch_quiver

    try:
        nx = float(tb_nx.text.strip())
        ny = float(tb_ny.text.strip())
        nz = float(tb_nz.text.strip())

        r0x = float(tb_r0x.text.strip())
        r0y = float(tb_r0y.text.strip())
        r0z = float(tb_r0z.text.strip())
    except ValueError:
        return

    # Normalize vectors
    n = normalize([nx, ny, nz])
    r0 = normalize([r0x, r0y, r0z])

    # Update n-hat arrow
    try:
        n_quiver.remove()
    except Exception:
        pass
    n_quiver = add_3d_arrow(ax, [0, 0, 0], n, color="blue")
    n_label.set_position((n[0] * 1.12, n[1] * 1.12))
    n_label.set_3d_properties(n[2] * 1.12, zdir=None)

    # Update initial Bloch vector arrow
    try:
        bloch_quiver.remove()
    except Exception:
        pass
    bloch_quiver = add_3d_arrow(ax, [0, 0, 0], r0, color="green")

    # Reset trajectory since dynamics changed
    reset_trajectory()
    fig.canvas.draw_idle()


btn.on_clicked(apply_nhat_and_r0)


plt.show()

