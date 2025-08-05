import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

# Konstanta dan parameter
delta = 0.7
t = -1.0
soc_list = np.array([0.0, 0.24])
rashba = 0.4
width = 10
nkr = 101
n_avg = 100
W = 6 * soc_list

# Matriks Pauli
sigma_z = np.array([0., 0., 0., 1.])
sigma_x = np.array([0., 1., 0., 0])
sigma_y = np.array([0., 0., 1., 0])
r3h = np.sqrt(3.0) / 2.0
sigma_a = 0.5 * sigma_x - r3h * sigma_y
sigma_b = 0.5 * sigma_x + r3h * sigma_y
sigma_c = -1.0 * sigma_x

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

# Plotting setup
fig, ax = plt.subplots(1, 2, figsize=(6, 3.7))
path = [[0., 0.], [2./3., 1./3.], [0.5, 0.5], [1./3., 2./3.], [0., 0.]]
label = [r'$\Gamma$', r'$K$', r'$M$', r"$K'$", r'$\Gamma$']

for je, soc_val in enumerate(soc_list):
    eval_total = None
    evec_total = None
    for _ in range(n_avg):
        my_model = set_model(t, soc_val, rashba, delta, W[je])
        ribbon_model = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)
        k_vec, k_dist, k_node = ribbon_model.k_path(path, nkr, report=False)
        rib_eval, rib_evec = ribbon_model.solve_all(k_vec, eig_vectors=True)

        if eval_total is None:
            eval_total = np.array(rib_eval)
            evec_total = np.array(rib_evec)
        else:
            eval_total += np.array(rib_eval)
            evec_total += np.array(rib_evec)

    rib_eval_avg = eval_total / n_avg
    rib_evec_avg = evec_total / n_avg
    nbands = rib_eval_avg.shape[0]

    ax1 = ax[je]
    ax1.set_xlim([0, k_node[-1]])
    ax1.set_xticks(k_node)
    ax1.set_xticklabels(label)
    ax1.set_ylim(-5, 5)
    ax1.set_ylabel("Edge band structure (010)")
    ax1.set_title(f"SOC = {soc_val:.3f}")

    for i in range(len(k_vec)):
        pos_exp = ribbon_model.position_expectation(rib_evec_avg[:, i], dir=1)
        weight = np.minimum(3.0 * pos_exp / width, 1.0)
        ax1.scatter([k_dist[i]] * nbands, rib_eval_avg[:, i],
                    s=0.6 + 2.5 * weight, c='C1', marker='x')

fig.tight_layout()
plt.savefig("disordered_band_structure_matched.pdf")
plt.show()

