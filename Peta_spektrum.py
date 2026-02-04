from pythtb import TBModel, Lattice
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

delta = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

t = 1.0

soc_val = 0.25

rashba =  np.array([0.0, 0.05, 0.1, 0.2 , 0.3])
W = 0 * soc_val


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

    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...)

    model = TBModel(lattice=lattice, spinful=True)


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


    model.set_hop(0.1 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])

    model.set_hop(-0.1 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])


    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")


    return model

k_nodes = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0], [0, 0.5]]
k_labels = [
    r"$\bar{\Gamma}$",
    r"$\bar{X}$",
    r"$\bar{M}$",
    r"$\bar{\Gamma}$",
    r"$\bar{Y}$",
]

fig, ax = plt.subplots(len(delta), len(rashba), figsize=(12,12))
ax = np.atleast_2d(ax)

for a,i in enumerate(delta):
    for b,j in enumerate(rashba):

        my_model = set_model(t, soc_val, j, i, W)
        fin_model = my_model.make_finite(periodic_dirs=[0], num_cells=[20])

        # buat figure sementara
        fig_tmp, ax_tmp = fin_model.plot_bands(
            nk=201,
            k_nodes=k_nodes,
            k_node_labels=k_labels,
            lw=1
            )

        # ambil axes tunggal
        if isinstance(ax_tmp, (list, tuple, np.ndarray)):
            ax_tmp = np.asarray(ax_tmp).ravel()[0]

        # copy semua garis
        for line in ax_tmp.get_lines():
            ax[a,b].plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                lw=line.get_linewidth()
            )

        ax[a,b].set_title(f"Δ={i}, R={j}")
        ax[a,b].set_xlim(ax_tmp.get_xlim())
        ax[a,b].set_ylim(ax_tmp.get_ylim())

        plt.close(fig_tmp)   

plt.tight_layout()
plt.show()
