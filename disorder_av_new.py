import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model
import pandas as pd

# Konstanta dan parameter
delta = 0.7
t = -1.0
soc_list = np.array([0.06, 0.24])
rashba = 0.05
width = 10
nkr = 101
n_avg = 100
W = 10 * soc_list

# Matriks Pauli
sigma_z = np.array([0., 0., 0., 1.])
sigma_x = np.array([0., 1., 0., 0])
sigma_y = np.array([0., 0., 1., 0])
r3h = np.sqrt(3.0) / 2.0
sigma_a = 0.5 * sigma_x - r3h * sigma_y
sigma_b = 0.5 * sigma_x + r3h * sigma_y
sigma_c = -1.0 * sigma_x

data_csv =[]

def set_model(t, soc, rashba, delta, W):
    lat = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    model = tb_model(3, 3, lat, orb, nspin=2)

    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))
    onsite_energies = [
        delta + disorder_values[i] if i % 2 == 0 else -delta + disorder_values[i]
        for i in range(len(orb))
    ]
    model.set_onsite(onsite_energies)

    # Hopping terms
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        model.set_hop(t, 0, 1, lvec)

    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)

    model.set_hop(0.3 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])
    model.set_hop(-0.3 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])

    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")

    return model

fig, ax = plt.subplots(1, 2, figsize=(8, 5)) # Ukuran sedikit diperbesar
path = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
label = (r'$\Gamma$', r'$X$', r'$\Gamma$')

# List untuk menyimpan data CSV jika masih dibutuhkan
data_csv = []

for je, soc_val in enumerate(soc_list):
    ax1 = ax[je]
    ax1.set_title(f"SOC = {soc_val:.3f}, W = {W[je]:.2f}")
    
    # Setup sumbu sekali saja di awal
    # Kita perlu dummy model sekali untuk mendapatkan k_node yang benar
    dummy_model = set_model(t, soc_val, rashba, delta, 0)
    rib_dummy = dummy_model.cut_piece(width, fin_dir=1, glue_edgs=False)
    (k_vec, k_dist, k_node) = rib_dummy.k_path(path, nkr, report=False)
    
    ax1.set_xlim([0, k_node[-1]])
    ax1.set_xticks(k_node)
    ax1.set_xticklabels(label)
    ax1.set_ylim(-4, 4) # Zoom sedikit agar detail gap terlihat
    ax1.set_ylabel("Energy (eV)")

    print(f"Sedang memproses SOC={soc_val}... Mohon tunggu.")

    # --- LOOP SAMPEL DISORDER ---
    for sample_idx in range(n_avg):
        # 1. Bangun Model & Solve untuk sampel INI
        my_model = set_model(t, soc_val, rashba, delta, W[je])
        ribbon_model = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)
        
        # Solve
        (k_vec, k_dist, k_node) = ribbon_model.k_path(path, nkr, report=False)
        rib_eval, rib_evec = ribbon_model.solve_all(k_vec, eig_vectors=True)
        nbands = rib_eval.shape[0]

        # 2. Plot LANGSUNG sampel ini (Accumulative Plotting)
        #    Jangan dirata-rata!
        for i in range(len(k_vec)):
            # Hitung bobot tepi hanya untuk sampel ini
            pos_exp = ribbon_model.position_expectation(rib_evec[:, i], dir=1)
            
            # Logika pewarnaan:
            # Jika di tepi (<1 atau >width-1) -> Bobot 1 (Jelas)
            # Jika di tengah -> Bobot 0.1 (Sangat transparan)
            edge_weight = np.where((pos_exp < 1.0) | (pos_exp > width - 1.0), 1.0, 0.0)
            
            # Trik Visualisasi:
            # Gunakan alpha rendah (0.05). 
            # Jika 100 garis menumpuk di tempat yang sama -> Jadi tebal/gelap.
            # Jika garis acak (disorder) -> Tetap tipis/samar.
            
            # Plot titik bulk (warna abu-abu, sangat transparan)
            # Kita pisah agar visualnya lebih bagus: Bulk (Grey), Edge (Orange)
            
            # Plot semua titik dulu dengan alpha sangat rendah
            ax1.scatter([k_dist[i]] * nbands, rib_eval[:, i],
                        s=0.5, 
                        c='gray', # Warna dasar abu-abu
                        alpha=0.6, # <--- KUNCI: Transparansi tinggi
                        marker='x', linewidths=0)

            # Timpa titik edge (jika ada) dengan warna oranye
            # Cari indeks pita yang merupakan edge state
            edge_indices = np.where(edge_weight > 0.5)[0]
            if len(edge_indices) > 0:
                ax1.scatter([k_dist[i]] * len(edge_indices), rib_eval[edge_indices, i],
                            s=0.8, 
                            c='C1', # Warna oranye untuk edge
                            alpha=1.0, # Sedikit lebih jelas dari bulk
                            marker='.', linewidths=0)

        # (Opsional) Print progress setiap 10 sampel agar tahu jalan
        if sample_idx % 10 == 0:
            print(f"  Sampel {sample_idx}/{n_avg} selesai.")

print("Plotting selesai. Menyimpan gambar...")
fig.tight_layout()
plt.savefig("disordered_band_structure_accumulative.pdf") 
plt.show()
