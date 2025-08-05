from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model, wf_array 


# Disorder strength
W = 0.0  # Adjust W as needed
delta=0.7     # site energy
t=-1.0        # spin-independent first-neighbor hop
#theta = 30.0
soc=0.154      # spin-dependent second-neighbor hop
rashba=0.05   # spin-flip first-neighbor hop
soc_list=np.array([-0.054,-0.24]) # spin-dependent second-neighbor hop
zoc = 0.3*soc


def set_model(t, soc, rashba, delta):
    # Set up Kane-Mele model
    lat = [[1, 0, 0], [0.5,np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    
    model = tb_model(3, 3, lat, orb, nspin=2)
    
    # Generate disorder values for each site
    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))  
    
    # Set onsite energy with disorder
    onsite_energies = [delta + disorder_values[i] if i % 2 == 0 else -delta + disorder_values[i] for i in range(len(orb))]
    model.set_onsite(onsite_energies)
    
    # Definitions of Pauli matrices
    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])
    sigma_z = np.array([0., 0., 0., 1])
    r3h = np.sqrt(3.0)/2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x

    # Spin-independent first-neighbor hops
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        model.set_hop(t, 0, 1, lvec)

    # Spin-dependent second-neighbor hops
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)

    model.set_hop(0.3 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])
    model.set_hop(-0.3 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])

    # Spin-flip first-neighbor hops
    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")

    return model

model = set_model(t, soc, rashba, delta)

# Initialize wf_array untuk grid 2D (k_x, k_y) dengan k_z tetap
nk = 41
dk = 1.0 / (nk - 1)
wf = wf_array(model, [nk, nk])  # Grid 2D (asumsi sistem 2D efektif)

# Isi wf_array dengan vektor eigen
for k0 in range(nk):
    for k1 in range(nk):
        kx = k0 * dk
        ky = k1 * dk
        kvec = [kx, ky, 0.0]  # Fix k_z=0 untuk sistem 2D
        (eval, evec) = model.solve_one(kvec, eig_vectors=True)
        wf[k0, k1] = evec

# Impose periodic boundary conditions
wf.impose_pbc(mesh_dir=0, k_dir=0)
wf.impose_pbc(mesh_dir=1, k_dir=1)

# Hitung HWF centers (Berry phase)
hwfc = wf.berry_phase([0,1], dir=1, contin=True, berry_evals=True)/(2*np.pi)

# Plot Wannier flow
fig, ax = plt.subplots(figsize=(6.0, 3.71))
kx_vals = np.linspace(0, 1, nk)
colors = ['C1', 'brown']
labels = ['Orbital A', 'Orbital B']

for n in range(2):
    for shift in [-1.0, 0.0, 1.0]:
        ax.plot(kx_vals, hwfc[:,n] + shift, 
                color=colors[n], 
                linestyle='--',
                marker='x',
                linewidth=3.5, 
                alpha=0.9)

ax.set_xlim([0.,1.])
ax.set_xticks([0.,0.5,1.])
ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"HWF center $\bar{s}_y$")

ax.grid(True, which='both', linestyle='--', alpha=0.3)
plt.title(f"Wannier Centers (SOC={soc}, Rashba={rashba})", fontsize=11)
plt.tight_layout()
plt.savefig("hwf_centers.png", dpi=300, bbox_inches='tight')
plt.show()

