from pythtb import TBModel, Lattice

import numpy as np

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


delta = 0.7

t = -1.0

soc_list = np.array([0.06, 0.24])

soc_val = 1/4

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


    model.set_hop(0.3 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])

    model.set_hop(-0.3 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])


    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")


    return model


my_model = set_model(t, soc_val, rashba, delta, 0)


print(my_model)


my_model.info()

fig = my_model.visualize_3d(draw_hoppings=True)
plt.show()
