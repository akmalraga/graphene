import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Patch

# Styling defaults
rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.usetex": False,  # Set True if using LaTeX renderer
    "font.family": "serif"
})

# Critical values
Wc_onsite = 6.0
Wc_hopping = 5.0

# Meshgrid
x = np.linspace(1, 10, 400)
y = np.linspace(1, 10, 400)
X, Y = np.meshgrid(x, y)

# Phase map
Z = np.zeros_like(X)
Z[(X > Wc_onsite) | (Y > Wc_hopping)] = 1  # Anderson
Z[(X <= Wc_onsite) & (Y <= Wc_hopping)] = 0  # Topological

# Plot
fig, ax = plt.subplots(figsize=(6.0, 3.7))
c = ax.contourf(X, Y, Z, levels=[-0.1, 0.5, 1.1],
                colors=['brown', 'C1'], alpha=0.9)

# Boundary line
ax.contour(X, Y, Z, levels=[0.5], colors='k', linewidths=1.5)

# Labels and title
ax.set_xlabel(r"Onsite Disorder $W_\mathrm{on}$", labelpad=6)
ax.set_ylabel(r"Hopping Disorder $W_\mathrm{hop}$", labelpad=6)
ax.set_title(r"Phase Diagram: Topological $\leftrightarrow$ Anderson Insulator", pad=12)

# Custom legend
legend_elements = [
    Patch(facecolor='brown', edgecolor='k', label='Topological Phase'),
    Patch(facecolor='C1', edgecolor='k', label='Anderson Insulator')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True)

# Grid
ax.grid(True, linestyle='--', alpha=0.2)

# Axis range (optional)
ax.set_xlim(1, 10)
ax.set_ylim(1, 10)

plt.tight_layout()
plt.savefig("phase_diagram_topo_anderson.png", dpi=300)
plt.show()

