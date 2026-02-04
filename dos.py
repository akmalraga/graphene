from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model
from joblib import Parallel, delayed

# ==========================================
# PARAMETER
# ==========================================
delta = 0.7
t = -1.0
soc = 1.24       # Pastikan ini konsisten dengan studi Anda (0.06 atau 0.24?)
rashba = 0.4     # Hati-hati, nilai ini cukup besar
width = 30
nkr = 100        # Jumlah k-point
n_samples = 50   # Jumlah sampel disorder (bisa dinaikkan jika laptop kuat)
sigma_broadening = 0.05 # Lebar Gaussian (smearing)

# Range Disorder yang mau dicek
W_values = np.linspace(0, 50, 26) * soc 
dos_at_zero_list = []

# ==========================================
# FUNGSI MODEL
# ==========================================
def set_model(t, soc, rashba, delta, W):
    lat = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    model = tb_model(3, 3, lat, orb, nspin=2)

    # Disorder acak
    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))
    onsite_energies = [
        delta + disorder_values[i] if i % 2 == 0 else -delta + disorder_values[i]
        for i in range(len(orb))
    ]
    model.set_onsite(onsite_energies)

    # Hopping
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        model.set_hop(t, 0, 1, lvec)

    # SOC
    sigma_z = np.array([0., 0., 0., 1.])
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)
        
    # Interlayer & Rashba (Sesuai kode asli Anda)
    model.set_hop(0.3 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])
    model.set_hop(-0.3 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])

    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])
    r3h = np.sqrt(3.0) / 2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x
    
    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")
    
    return model

# ==========================================
# FUNGSI SIMULASI PER W
# ==========================================
def get_eigenvalues_for_sample(W):
    """Fungsi helper untuk dijalankan satu sampel (untuk parallel)"""
    my_model = set_model(t, soc, rashba, delta, W)
    ribbon = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)
    
    # PERBAIKAN: Gunakan lintasan 1D uniform (0 sampai 1)
    # Ini memberikan sampling DOS yang lebih akurat daripada path 2D berbelok
    k_vec = np.linspace(0.0, 1.0, nkr).reshape((nkr, 1)) 
    
    vals = ribbon.solve_all(k_vec)
    return vals.flatten()

# ==========================================
# LOOP UTAMA
# ==========================================
print("Mulai perhitungan DOS vs W...")

for val_W in W_values:
    # Jalankan Parallel untuk n_samples
    # Backend 'loky' biasanya paling stabil untuk numpy
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(get_eigenvalues_for_sample)(val_W) for _ in range(n_samples)
    )
    
    # Gabungkan semua eigenvalues dari 100 sampel ke satu array raksasa
    all_energies = np.concatenate(results)
    
    # --- METODE GAUSSIAN BROADENING (LEBIH HALUS DARI HISTOGRAM) ---
    # Kita ingin hitung DOS tepat di E = 0.0
    # Rumus: Sum (1 / (sigma * sqrt(2pi))) * exp(-(E - E_n)^2 / 2sigma^2)
    # Karena kita cuma butuh nilai di E=0, rumusnya jadi sederhana:
    # exp(-E_n^2 / 2sigma^2)
    
    gauss_contrib = np.exp(-(all_energies**2) / (2 * sigma_broadening**2))
    dos_val = np.sum(gauss_contrib) / (np.sqrt(2 * np.pi) * sigma_broadening)
    
    # Normalisasi (dibagi total k-point dan total sampel)
    dos_norm = dos_val / (nkr * n_samples * width) # Opsional: bagi width agar per atom
    
    dos_at_zero_list.append(dos_norm)
    print(f"W = {val_W:.2f} | DOS(E=0) = {dos_norm:.4f}")

# ==========================================
# PLOTTING
# ==========================================
plt.figure(figsize=(7, 5))
plt.plot(W_values / soc, dos_at_zero_list, 'o-', linewidth=2, color='darkblue')

plt.xlabel(r"Disorder Strength $W/SOC$", fontsize=12)
plt.ylabel(r"Density of States at $E=0$ (a.u.)", fontsize=12)
plt.title(f"Evolution of DOS at Fermi Level (Width={width})")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("dos_vs_W_gaussian.pdf")
plt.show()
