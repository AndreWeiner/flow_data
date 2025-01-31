"""
    compute the convergence behavior of the U mean field for the cylinder3D simulation case wrt
    time
"""
import torch as pt
import matplotlib.pyplot as plt

from os import makedirs
from typing import Tuple
from os.path import join, exists

from flowtorch.data import FOAMDataloader


def compute_norm_of_fields(load_path: str, time_boundaries: list = None,
                           field: str = "UMean") -> Tuple[pt.Tensor, list]:
    loader = FOAMDataloader(load_path)

    # get the defined boundaries for start and end time to use if provided
    if time_boundaries is not None:
        idx = sorted([i for i, t in enumerate(loader.write_times) if t in time_boundaries])
        write_times = loader.write_times[idx[0]:idx[1]+1]

    # else use all times steps but zero
    else:
        write_times = loader.write_times[1:]

    # check for the time steps in which the target filed is present
    write_times = [t for t in write_times if field in loader.field_names[t]]

    # compute the norm of the field in the last time step
    norm_first_field = loader.load_snapshot(field, write_times[0]).norm()

    # now compute the difference of the norm between two consecutive time steps
    all_norms, last_snapshot = [1], 0
    for i in range(len(write_times)):
        if i == 0:
            last_snapshot = loader.load_snapshot(field, write_times[i])
            continue
        new_snapshot = loader.load_snapshot(field, write_times[i])
        dt = float(write_times[i]) - float(write_times[i-1])
        all_norms.append(((new_snapshot-last_snapshot) / dt).norm() / norm_first_field)
        last_snapshot = new_snapshot

    # don't return the norm of the last field, since the difference is zero
    return pt.tensor(list(map(float, write_times))), all_norms


if __name__ == "__main__":
    # define load and save path
    load_dir = join("/media", "janis", "Elements", "Janis", "cylinder_3D_Re3900_tests")
    save_dir = join("run", "cylinder_3D_Re3900", "plots_final")

    # cases to compare
    cases = ["cylinder_3D_Re3900"]

    # flow quantities and cylinder properties
    u_inf = 39
    d = 0.1

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # load the U mean fields, compute the Frobenius norm between the difference of two consecutive time steps scaled
    # wrt the norm of the field from the last time step
    results = [compute_norm_of_fields(join(load_dir, c)) for c in cases]

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for times, res in results:
        ax.plot(times * u_inf / d, res, marker="x")
    ax.set_xlim(results[0][0][0] * u_inf / d, results[0][0][-1] * u_inf / d)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t \frac{U_{\infty}}{d}$")
    ax.set_ylabel(r"$\left| \left| \frac{\overline{U}_{t+1} - \overline{U}_{t}}{\Delta t}\right| \right|_F \times "
                  r"\frac{1}{||\overline{U}_{t_0} ||_F}$")
    ax.grid(visible=True, which="major", linestyle="-", alpha=0.45, color="black", axis="y")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.35, color="black", axis="y")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_dir, f"convergence_Umean_field_vs_time.png"))
    plt.close("all")
