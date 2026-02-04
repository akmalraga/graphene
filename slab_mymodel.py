from pythtb import TBModel, Lattice
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
delta = 0.0
t = 1.0
soc_val = 0.125
rashba = 0.05
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


my_model = set_model(t, soc_val, rashba, delta, W)


print(my_model)


my_model.info()

fin_model = my_model.make_finite(periodic_dirs=[1], num_cells=[20])

k_nodes = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0], [0, 0.5]]
k_labels = [
    r"$\bar{\Gamma}$",
    r"$\bar{X}$",
    r"$\bar{M}$",
    r"$\bar{\Gamma}$",
    r"$\bar{Y}$",
]

fin_model.visualize_3d(draw_hoppings=True)

fig, ax = fin_model.plot_bands(
    nk=500, k_nodes=k_nodes, k_node_labels=k_labels, proj_orb_idx=[1]
)

plt.show()
