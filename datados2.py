from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# Parameter
delta = 0.7     # Onsite energy
t = -1.0        # Hopping
soc = 0.06      # Spin-orbit coupling (SOC)
rashba = 0.05   # Rashba coupling
width = 10      # Lebar ribbon
nkr = 101       # Jumlah k-points
n_samples = 10  # Jumlah realisasi disorder
W_values = np.linspace(0, 30.0, 100) * soc  # Rentang W yang diuji

def set_model(t, soc, rashba, delta, W):
    # Set up model Kane-Mele dengan disorder
    lat = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    
    model = tb_model(3, 3, lat, orb, nspin=2)
    
    # Generate disorder
    np.random.seed()  # Seed berbeda untuk setiap realisasi
    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))
    
    # Set onsite energy dengan disorder
    onsite_energies = [
        delta + disorder_values[i] if i % 2 == 0 
        else -delta + disorder_values[i] 
        for i in range(len(orb))
    ]
    model.set_onsite(onsite_energies)
    
    # Spin-independent first-neighbor hopping
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        model.set_hop(t, 0, 1, lvec)
    
    # Spin-dependent second-neighbor hopping (SOC)
    sigma_z = np.array([0., 0., 0., 1])  # Matriks Pauli z
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)
    
    # Spin-flip first-neighbor hopping (Rashba)
    sigma_a = 0.5 * np.array([0., 1., 0., 0]) - (np.sqrt(3)/2) * np.array([0., 0., 1., 0])
    sigma_b = 0.5 * np.array([0., 1., 0., 0]) + (np.sqrt(3)/2) * np.array([0., 0., 1., 0])
    sigma_c = -1.0 * np.array([0., 1., 0., 0])
    
    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")
    
    return model

# List untuk menyimpan DOS pada E=0
dos_at_zero = []

for W in W_values:
    all_eigenvalues = []
    for _ in range(n_samples):
        # Bangun model dengan disorder W
        my_model = set_model(t, soc, rashba, delta, W)
        
        # Potong model menjadi ribbon
        ribbon_model = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)
        
        # Hitung eigenenergi
        (k_vec, k_dist, k_node) = ribbon_model.k_path(
            [[0.,0.], [2./3.,1./3.], [.5,.5], [1./3.,2./3.], [0.,0.]],
            nkr, report=False
        )
        rib_eval = ribbon_model.solve_all(k_vec)
        all_eigenvalues.append(rib_eval.flatten())
    
    # Gabungkan semua eigenenergi
    combined_eval = np.concatenate(all_eigenvalues)
    
    # Hitung histogram dan ekstrak DOS pada E=0
    hist, bin_edges = np.histogram(combined_eval, bins=50, range=(-4., 4.), density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    idx_zero = np.argmin(np.abs(bin_centers))  # Indeks bin terdekat E=0
    dos_at_zero.append(hist[idx_zero])

# Plot hasil
plt.plot(W_values/soc, dos_at_zero, 'o-', label='Simulasi')
plt.axvline(x=2.5, c='r', ls='--', label='Prediksi $W_c=2.5\lambda_{SO}$')
plt.xlabel("$W/\lambda_{SO}$")
plt.ylabel("DOS pada $E=0$")
plt.legend()
plt.show()
