from pythtb import tb_model
import numpy as np
import matplotlib.pyplot as plt

def get_my_model(phase):
    """
    Mengembalikan model TB anda.
    phase: 'topological' (delta kecil) atau 'trivial' (delta besar)
    """
    
    # --- Parameter Asal Anda ---
    t = -1.0
    soc_val = 0.25  # 1/4 seperti dalam kod anda
    rashba = 0.05
    W = 10  # Kita matikan disorder untuk plot band structure yang bersih
    
    # Logik pertukaran fasa (seperti hwc.py)
    # Jika delta dominan, sistem menjadi trivial insulator
    if phase == "trivial":
        delta = 3.0  
    else:
        delta = 0.7  # Nilai asal anda (kemungkinan topologi)

    # --- Definisi Matriks Pauli (2x2) ---
    sigma_0 = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Vektor Rashba
    r3h = np.sqrt(3.0) / 2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x

    # --- Definisi Kekisi (Lattice) ---
    # Model anda adalah 3D
    lat_vecs = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb_vecs = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    
    # Inisialisasi Model (dim_k=3, dim_r=3, nspin=2)
    my_model = tb_model(3, 3, lat_vecs, orb_vecs, nspin=2)

    # --- Onsite Energies ---
    onsite_energies = [delta, -delta]
    my_model.set_onsite(onsite_energies)

    # --- Hopping Terms (Mengikut vis_mymodel.py) ---
    
    # 1. Nearest Neighbor (t)
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        my_model.set_hop(t * sigma_0, 0, 1, lvec)

    # 2. Next-Nearest Neighbor (SOC - Kane Mele Type)
    # Sublattice 0 -> 0
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        my_model.set_hop(soc_val * 1j * sigma_z, 0, 0, lvec)
    # Sublattice 1 -> 1
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        my_model.set_hop(soc_val * 1j * sigma_z, 1, 1, lvec)

    # 3. Interlayer / Vertical Hopping (Ciri khas model anda)
    my_model.set_hop(0.1 * soc_val * 1j * sigma_z, 1, 1, [0, 0, 1])
    my_model.set_hop(-0.1 * soc_val * 1j * sigma_z, 0, 0, [0, 0, 1])

    # 4. Rashba Coupling
    # Perhatikan mode="add"
    my_model.set_hop(1j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    my_model.set_hop(1j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    my_model.set_hop(1j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")

    return my_model

# --- Bahagian Plotting (Diadaptasi daripada hwc.py) ---

# Buat dua model: satu trivial, satu topologi (asal)
model_triv = get_my_model("trivial")
model_topo = get_my_model("topological")

# Inisialisasi rajah
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Laluan K-Space (High Symmetry Points)
# Perlu 3 koordinat kerana model anda 3D (x, y, z)
# Gamma -> K -> M -> K' -> Gamma (pada satah kz=0)
k_nodes = [
    [0.0, 0.0, 0.0],          # Gamma
    [2./3., 1./3., 0.0],      # K
    [0.5, 0.5, 0.0],          # M
    [1./3., 2./3., 0.0],      # K'
    [0.0, 0.0, 0.0],          # Gamma
]

# Label untuk paksi
label_k = (r"$\Gamma$", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma$")

# Selesaikan dan plot
# proj_orb_idx=[0] bermaksud kita mewarnakan band berdasarkan unjuran pada orbital 0
model_triv.plot_bands(
    k_nodes=k_nodes, nk=201, k_node_labels=label_k, fig=fig, ax=ax1, proj_orb_idx=[0]
)
model_topo.plot_bands(
    k_nodes=k_nodes, nk=201, k_node_labels=label_k, fig=fig, ax=ax2, proj_orb_idx=[0]
)

# Tajuk
ax1.set_title(r"Model Anda: Fasa Trivial ($\Delta$ Besar)")
ax2.set_title(r"Model Anda: Fasa Topologi? ($\Delta=0.7$)")

# Hadkan paksi Y jika perlu agar plot lebih jelas
ax1.set_ylim(-6, 6)
ax2.set_ylim(-6, 6)

plt.tight_layout()
plt.show()
