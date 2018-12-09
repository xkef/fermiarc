import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh


def constructarray(y, z, a, mat_shape):
    y, z, a = complex(y), complex(z), complex(a)
    m = np.zeros((mat_shape, mat_shape), dtype=complex)
    # Diag
    np.fill_diagonal(m,
                     np.tile([(-a + np.cos(y) + np.cos(z)),
                              (a - np.cos(y) - np.cos(z))],
                             mat_shape))
    # Frist of-diag
    np.fill_diagonal(m[:, 1:],
                     -np.tile([np.sin(y) * 1j,
                               -1j * np.real(np.sqrt(np.pi / 2))],
                              mat_shape - 1))
    np.fill_diagonal(m[1:],
                     np.tile([np.sin(y) * 1j,
                              -1j * np.real(np.sqrt(np.pi / 2))],
                             mat_shape - 1))
    # Second of-diag
    np.fill_diagonal(m[:, 2:],
                     np.tile([1 + 0j, -1 + 0j],
                             mat_shape - 2))
    np.fill_diagonal(m[2:],
                     np.tile([1 + 0j, -1 + 0j],
                             mat_shape - 2))
    # Third of-diag
    np.fill_diagonal(m[:, 3:],
                     np.tile([1j * np.real(np.sqrt(np.pi / 2)), 0],
                             mat_shape - 3))
    np.fill_diagonal(m[3:],
                     -np.tile([1j * np.real(np.sqrt(np.pi / 2)), 0],
                              mat_shape - 3))
    return m


if __name__ == '__main__':
    # init
    nr_eigs = 500
    grid = np.linspace(0, 10, nr_eigs)
    eig = np.zeros((nr_eigs, nr_eigs))

    # timeit
    print("timing...")
    start_time = time.time()
    # calc eigs
    for index, z_grid in np.ndenumerate(grid):
        eig[index, :] = eigvalsh(constructarray(y=0,
                                                z=z_grid,
                                                a=0,
                                                mat_shape=nr_eigs))
    print(nr_eigs, ": %s seconds" % (time.time() - start_time))

    # plot
    plt.clf()
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    fig, ax = plt.subplots()

    ax.plot(eig, linewidth=1)

    ax.set(xlabel='??',
           ylabel='Energy Levels',
           title='Fermi Something')

    plt.savefig(
        'plots/eigvals.pdf',
        dpi=None,
        facecolor='w',
        edgecolor='w',
        papertype='a4',
        format='pdf',
        transparent=True,
        bbox_inches=None,
        pad_inches=0.1,
        frameon=None)
