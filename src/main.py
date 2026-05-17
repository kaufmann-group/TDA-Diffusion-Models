
from msep.msep_cpp import MultiSpeciesExclusionProcess

import numpy as np
import matplotlib.pyplot as plt

def plot_path_2d(path):
        assert len(path[0]) == 2, "can only plot with dimension = 4"

        plt.figure(figsize=(6, 6))
        plt.plot(path[:, 0], path[:, 1], "-o", markersize=2)
        plt.axis("equal")
        plt.xlabel(r"$h_1$")
        plt.ylabel(r"$h_2$")
        plt.title("projected directed polymer path, d = 3")

        plt.savefig("figures/projected_directed_polymer_3d.png", dpi=300)
        plt.show()

def plot_path_3d(path):
    assert len(path[0]) == 3, "can only plot with dimension = 4"

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o", markersize=2)
    ax.set_xlabel(r"$h_1$")
    ax.set_ylabel(r"$h_2$")
    ax.set_zlabel(r"$h_3$")
    ax.set_title("projected directed polymer path, d = 4")
    ax.set_box_aspect([1, 1, 1])

    plt.savefig("figures/projected_directed_polymer_4d.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    dimension = 3
    density = [1 / 3, 1 / 3, 1 / 3]
    length = 300

    rates_matrix_3d = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    asym_diffusion_2d = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_3d, length=length, seed=2504, shuffle=True, check_pairwise_balance=True)

    asym_diffusion_2d.simulate(steps=100000)
    plot_path_2d(asym_diffusion_2d.get_path())

    """
    Dimension 4 case: projects onto a 3D hyperplane.
    """
    dimension = 4
    density = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    length = 400

    rates_matrix_4d = np.array(
        [
            [0.0, 1.6, 2.3, 1.8],
            [2.4, 0.0, 2.7, 2.2],
            [1.7, 1.3, 0.0, 1.5],
            [2.2, 1.8, 2.5, 0.0],
        ],
        dtype=np.float64,
    )

    asym_diffusion_3d = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_4d, length=length, seed=2504, shuffle=True, check_pairwise_balance=True)

    asym_diffusion_3d.simulate(steps=100000)
    plot_path_3d(asym_diffusion_3d.get_path())
