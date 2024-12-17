"""
    test plots, etc. for cylinder3D_Re3900 case
    large parts of this code are copied from:

     https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_mesh_for_CFD_applications/tree/main/post_processing

"""
import regex as re
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os import makedirs
from typing import Union
from os.path import join, exists
from matplotlib.patches import Circle, Rectangle

from simutils import compute_friction_velocity, load_force_coeffs


def get_cfl_number(load_path: str) -> dict:
    """
    gets the avg. and max. Courant numbers from the solver's log file

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: dict containing the mean and max. Courant numbers
    """
    # check if we have multiple log files, if so sort them
    try:
        logs = sorted(glob(join(load_path, f"log.pimpleFoam*")), key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        logs = glob(join(load_path, f"log.pimpleFoam*"))

    data = {"cfl_mean": [], "cfl_max": []}
    for log in logs:
        with open(log, "r") as f:
            logfile = f.readlines()

        """
        solver log file looks something like this:
    
            Courant Number mean: 0.00156147 max: 0.860588
        """
        start_line = False
        for line in logfile:
            # omit the initial Courant number (prior starting the time loop)
            if line.startswith("Starting time loop"):
                start_line = True
            if line.startswith("Courant Number mean") and start_line:
                data["cfl_mean"].append(float(line.split(" ")[3]))
                data["cfl_max"].append(float(line.split(" ")[-1].strip("\n")))
            else:
                continue

    return data


def load_residuals(load_path: str):
    dirs = sorted(glob(join(load_path, "postProcessing", "residuals", "*")), key=lambda x: float(x.split("/")[-1]))
    res = [pd.read_csv(join(p, "solverInfo.dat"), delimiter=r"\s+", skiprows=2, header=None,
                       usecols=[0, 2, 3, 5, 6, 8, 9, 13, 14],
                       names=["t", "Ux_start", "Ux_end", "Uy_start", "Uy_end", "Uz_start", "Uz_end", "p_start",
                              "p_end"]) for p in dirs]
    return res[0] if len(res) == 1 else pd.concat(res)


def get_pimple_iterations(load_path: str) -> list:
    """
    gets the number of PIMPLE iterations (p-U-couplings)

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: dict containing the mean and max. Courant numbers, and if present the mean and max. CFL from the interface
    """
    pattern = [r"PIMPLE: not converged within ", r"PIMPLE: converged in "]

    # check if we have multiple log files, if so sort them
    try:
        logs = sorted(glob(join(load_path, f"log.pimpleFoam*")), key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        logs = glob(join(load_path, f"log.pimpleFoam*"))

    data = []
    for log in logs:
        with open(log, "r") as f:
            logfile = f.readlines()

        for line in logfile:
            if line.startswith(pattern[0]) or line.startswith(pattern[1]):
                data.append(int(line.split(" ")[-2]))
    return data


def get_probe_locations(load_path: str) -> pt.Tensor:
    pattern = r"-?\d.\d+ -?\d.\d+ -?\d.\d+"
    with open(join(load_path, "system", "controlDict"), "r") as f:
        loc = f.readlines()

    # avoid finding other coordinate tuples, which may be present in the controlDict
    idx = [k for k, line in enumerate(loc) if "probeLocations" in line][0]

    # get coordinates of probes, omit appending empty lists and map strings to float
    coords = [re.findall(pattern, line) for k, line in enumerate(loc) if re.findall(pattern, line) and k > idx]
    return pt.tensor([list(map(float, c[0].split())) for c in coords])


def load_probes(load_path: str, num_probes: int, filename: str = "p", skip_n_points: int = 0) -> pd.DataFrame:
    """
    load the data of the probes written out during the simulation

    :param load_path: path to the top-level directory of the simulation
    :param num_probes: amount of probes placed in the flow field
    :param filename: name of the field written out in the probes directory, e.g. 'p', 'p_rgh' or 'U'
    :param skip_n_points: offset, in case we don't want to read in the 1st N time steps of the values
    :return: dataframe containing the values for each probe
    """
    dirs = sorted(glob(join(load_path, "postProcessing", "probes", "*")), key=lambda x: float(x.split("/")[-1]))
    _probes = []

    for d in dirs:
        # skip header, header = n_probes + 2 lines containing probe no. and time header
        if filename.startswith("p"):
            names = ["time"] + [f"{filename}_probe_{pb}" for pb in range(num_probes)]
            probe = pd.read_table(join(d, filename), sep=r"\s+", skiprows=(num_probes + 2) + skip_n_points, header=None,
                                  names=names)
        else:
            names = ["time"]
            for pb in range(num_probes):
                names += [f"{k}_probe_{pb}" for k in ["ux", "uy", "uz"]]

            probe = pd.read_table(join(d, filename), sep=r"\s+", skiprows=(num_probes + 2) + skip_n_points, header=None,
                                  names=names)

            # replace all parentheses, because (ux u_y uz) is separated since all columns are separated with white space
            # as well
            for k in names:
                if k.startswith("ux"):
                    probe[k] = probe[k].str.replace("(", "", regex=True).astype(float)
                elif k.startswith("uz"):
                    probe[k] = probe[k].str.replace(")", "", regex=True).astype(float)
                else:
                    continue
        _probes.append(probe)

    return _probes[0] if len(_probes) == 1 else pd.concat(_probes)


def load_line_samples(load_path: str, loc: list):
    # TODO: order Re stresses unknown, this is just a guess
    names = ["coord", "p", "pMean", "pPrime2Mean", "Ux", "Uy", "Uz", "UxMean", "UyMean", "UzMean", "UxxP2Mean",
             "UxyP2Mean", "UyxP2Mean", "UyyP2Mean", "UzxP2Mean", "UzzP2Mean"]

    all_lines, coords = [], []
    for l in loc:
        # use the last time step
        files = sorted(glob(join(load_path, "postProcessing", "sample_lines", "*", f"*_{l}_*.csv")),
                       key=lambda x: float(x.split("/")[-2]))[-1]

        # load the coordinates
        coords.append(pt.tensor(pd.read_csv(files, skiprows=1, header=None, names=["x", "y", "z"],
                                            usecols=range(0, 3)).values))

        # load the file containing the data
        all_lines.append(pt.tensor(pd.read_csv(files, names=names, header=None, sep=",", skiprows=1,
                                               usecols=range(len(names))).values).unsqueeze(-1))

    return coords, all_lines


def load_sampling_planes(load_path: str, name: str):
    planes = []

    # now loop over the time folders and load the data
    files = glob(join(load_path, "postProcessing", "sample_planes", "*", name))

    # we only need to load the coordinates once since they're not changing
    coords = pd.read_csv(files[0], skiprows=2, header=None, names=["x", "y", "z"], usecols=range(0, 3), sep=r"\s+")

    for file in sorted(files, key=lambda x: float(x.split("/")[-2])):
        if name.startswith("U"):
            planes.append(pt.tensor(pd.read_csv(file, skiprows=2, header=None, names=["Ux", "Uy", "Uz"],
                                                usecols=range(3, 6), sep=r"\s+").values).unsqueeze(-1))
        else:
            planes.append(pt.tensor(pd.read_csv(file, skiprows=2, header=None, names=["p"],
                                                usecols=[3], sep=r"\s+").values).unsqueeze(-1))

    return pt.tensor(coords.values), pt.cat(planes, dim=-1)


def load_surface_coordinates(load_path: str, name: str, cylinder_pos: Union[tuple, list]):
    coords, _ = load_sampling_planes(load_path, name)

    # account for the shift wrt cylinder's origin
    coords[:, 0] -= cylinder_pos[0]
    coords[:, 1] -= cylinder_pos[1]

    return coords


def compute_phi(coords: pt.Tensor, param):
    # compute phi
    angle = pt.atan2(coords[:, 1], coords[:, 0]).rad2deg()

    # sort angles and avg. over z
    phi_sorted = angle[pt.argsort(angle)].unique()
    if len(param.size()) > 1:
        _avg_over_z = pt.cat([param[pt.where(angle == k)[0], :, :].mean(0).unsqueeze(-1) for k in phi_sorted], dim=-1)
    else:
        _avg_over_z = pt.cat([param[pt.where(angle == k)].mean().unsqueeze(-1) for k in phi_sorted], dim=-1)

    return phi_sorted, _avg_over_z


def plot_probes(save_path: str, data: list, num_probes: int = 10, title: str = "", param: str = "p",
                legend_list: list = None, share_y: bool = True, xlabel: str = r"$t \qquad [s]$",
                scaling_factor: Union[int, float] = 1) -> None:
    """
    plot the values of the probes wrt time

    :param save_path: name of the top-level directory where the plots should be saved
    :param data: the probe data loaded using the 'load_probes' function of this script
    :param num_probes: number of probes placed in the flow field
    :param title: title of the plot (if wanted)
    :param param: parameter, either 'p', 'p_rgh'; or, if U was loaded: 'ux', 'uy', 'uz'
    :param legend_list: legend entries for the plot (if wanted)
    :param share_y: flag if all probes should have the same scaling for the y-axis
    :param xlabel: label for the x-axis
    :param scaling_factor: factor for making the time dimensionless if wanted
    :return: None
    """
    min_x = min([k.time.min() * scaling_factor for k in data])
    max_x = max([k.time.max() * scaling_factor for k in data])
    if share_y:
        fig, ax = plt.subplots(nrows=num_probes, ncols=1, figsize=(8, 8), sharex="all")
    else:
        fig, ax = plt.subplots(nrows=num_probes, ncols=1, figsize=(8, 8), sharex="all")

    for j in range(len(data)):
        for k in range(num_probes):
            ax[k].plot(data[j]["time"] * scaling_factor, data[j][f"{param}_probe_{k}"])
            ax[k].set_ylabel(f"$probe$ ${k + 1}$", rotation="horizontal", labelpad=35)
    ax[-1].set_xlabel(xlabel)
    ax[-1].set_xlim(min_x, max_x)
    fig.suptitle(title)
    fig.tight_layout()
    if legend_list:
        fig.subplots_adjust(bottom=0.12)
        fig.legend(legend_list, loc="lower center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_path, f"probes_vs_time_{param}.png"))
    plt.close("all")


if __name__ == "__main__":
    # define load and save path
    load_dir = join("/media", "janis", "Elements1", "Janis", "cylinder_3D_Re3900_tests")
    # load_dir = join("run", "cylinder_3D_Re3900")
    save_dir = join("run", "cylinder_3D_Re3900", "plots_final")

    # cases to compare
    cases = ["cylinder_3D_Re3900"]

    # flow quantities
    u_inf = 39
    rho = 1
    nu = 1.0e-3

    # cylinder properties
    d = 0.1
    cylinder_position = (0.8, 1.0)

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

    # """
    # load and plot the residuals
    residuals = [load_residuals(join(load_dir, c)) for c in cases]

    for i in range(len(cases)):
        fig, ax = plt.subplots(2, 1, sharex="col", figsize=(6, 4))
        ax[0].plot(residuals[i].t * u_inf / d, residuals[i].Ux_start, label="$U_{x}$")
        ax[0].plot(residuals[i].t * u_inf / d, residuals[i].Uy_start, label="$U_{y}$")
        ax[0].plot(residuals[i].t * u_inf / d, residuals[i].Uz_start, label="$U_{z}$")
        ax[0].plot(residuals[i].t * u_inf / d, residuals[i].p_start, label="$p$")

        ax[1].plot(residuals[i].t * u_inf / d, residuals[i].Ux_end)
        ax[1].plot(residuals[i].t * u_inf / d, residuals[i].Uy_end)
        ax[1].plot(residuals[i].t * u_inf / d, residuals[i].Uz_end)
        ax[1].plot(residuals[i].t * u_inf / d, residuals[i].p_end)

        ax[-1].set_xlabel(r"$t \frac{U_{\infty}}{d}$")
        ax[0].set_ylabel("$initial$ $residual$")
        ax[1].set_ylabel("$final$ $residual$")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[-1].set_xlim(residuals[i].t.iloc[0], residuals[i].t.iloc[-1] * u_inf / d)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.legend(loc="upper center", ncols=4)
        plt.savefig(join(save_dir, f"residuals_vs_t_mesh_{i}.png"))
        plt.close("all")

    # get the Courant number from solver log and plot it
    cfl = [get_cfl_number(join(load_dir, c)) for c in cases]

    # we write out the residuals every time step
    for i in range(len(cases)):
        fig, ax = plt.subplots(1, 1, sharex="col", figsize=(6, 3))
        t = pt.linspace(residuals[i].t.iloc[0], residuals[i].t.iloc[-1], len(cfl[i]["cfl_mean"]))
        ax.plot(t * u_inf / d, cfl[i]["cfl_mean"], label=r"$\mu\left(CFL\right)$")
        ax.plot(t * u_inf / d, cfl[i]["cfl_max"], label=r"$max\left(CFL\right)$")
        ax.set_xlabel(r"$t \frac{U_{\infty}}{d}$")
        ax.set_ylabel("$Co$")
        ax.set_xlim(0, residuals[i].t.iloc[-1] * u_inf / d)
        fig.tight_layout()
        fig.subplots_adjust()
        ax.legend(loc="upper right", ncols=4)
        plt.savefig(join(save_dir, f"courant_vs_t_mesh_{i}.png"))
        plt.close("all")

    # check PIMPLE iterations
    n_iter = [get_pimple_iterations(join(load_dir, c)) for c in cases]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i in range(len(cases)):
        t = pt.linspace(residuals[i].t.iloc[0], residuals[i].t.iloc[-1], len(n_iter[i])) * u_inf / d
        ax.plot(t, n_iter[i], label=f"mesh {i}")
    ax.set_xlabel(r"$t \frac{U_{\infty}}{d}$")
    ax.set_ylabel("$N_{PIMPLE}$")
    ax.set_xlim(residuals[0].t.iloc[0], residuals[0].t.iloc[-1] * u_inf / d)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, "n_pimple_iter_vs_t.png"))
    plt.close("all")

    # load and plot the force coefficients
    forces = [load_force_coeffs(join(load_dir, c)) for c in cases]

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex="col")

    for i in range(len(cases)):
        ax[0].plot(forces[i].t * u_inf / d, forces[i].cx, label=f"mesh {i}")
        ax[1].plot(forces[i].t * u_inf / d, forces[i].cy)
        ax[i].set_ylim(0.85, 1.05)
    ax[0].set_ylabel(r"$c_d$")
    ax[1].set_ylabel(r"$c_l$")
    ax[-1].set_xlabel(r"$t \frac{U_{\infty}}{d}$")
    ax[-1].set_xlim(0, forces[0].t.iloc[-1] * u_inf / d)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_dir, "coefficients_vs_t.png"))
    plt.close("all")

    # load and plot the probes for p and U
    n_probes = 7
    for p in ["U", "p"]:
        probes = [load_probes(join(load_dir, c), num_probes=n_probes, filename=p) for c in cases]
        if p == "U":
            for i in ["ux", "uy", "uz"]:
                plot_probes(save_dir, probes, param=i, scaling_factor=u_inf / d, xlabel=r"$t \frac{U_{\infty}}{d}$",
                            num_probes=n_probes, legend_list=[f"mesh {i}" for i in range(len(cases))])
        else:
            plot_probes(save_dir, probes, param=p, scaling_factor=u_inf / d, xlabel=r"$t \frac{U_{\infty}}{d}$",
                        num_probes=n_probes, legend_list=[f"mesh {i}" for i in range(len(cases))])
                        
    # compute u_tau
    t_start = 0.19225           # start after 75 CTU
    u_tau, grad_u = [], []
    for c in cases:
        _, _, _, _, u_tau_tmp, grad_u_tmp = compute_friction_velocity(join(load_dir, c, "postProcessing",
                                                                           "sample_planes"), "gradU_cylinder.raw",
                                                                      t_start, nu)
        u_tau.append(u_tau_tmp)
        grad_u.append(grad_u_tmp)
    del _, u_tau_tmp, grad_u_tmp

    # load the coordinates of the cylinder surface
    coordinates = [load_surface_coordinates(join(load_dir, c), "gradU_cylinder.raw", cylinder_position) for c in cases]

    # compute mean and std. dev. of u_tau
    phi, u_tau_mean, u_tau_std, grad_u_mean, grad_u_std = [], [], [], [], []

    # estimation of the normal distance to the first cell center (done in paraview) for computing approx. y+
    # y0 = [0.0000805, 0.0000675, 0.000054]
    for i in range(len(cases)):
        u_tau_tmp = pt.cat([u.unsqueeze(-1) for u in u_tau[i]], dim=-1)
        grad_u_tmp = pt.cat([u.unsqueeze(-1) for u in grad_u[i]], dim=-1)

        # estimate min. / max. y+
        # print(f"y+ (min. / max.) for case {i}: {round((rho * y0[i] * u_tau_tmp / nu).min().item(), 4)}, "
        #       f"{round((rho * y0[i] * u_tau_tmp / nu).max().item(), 4)}")

        # compute avg. u_tau and grad_u wrt z-coordinate for temporal mean and std. deviation
        phi_tmp, u_tau_mean_tmp = compute_phi(coordinates[i], u_tau_tmp.mean(-1))
        _, u_tau_std_tmp = compute_phi(coordinates[i], u_tau_tmp.std(-1))
        _, grad_u_mean_tmp = compute_phi(coordinates[i], grad_u_tmp.mean(-1))
        _, grad_u_std_tmp = compute_phi(coordinates[i], grad_u_tmp.std(-1))

        phi.append(phi_tmp)
        u_tau_mean.append(u_tau_mean_tmp)
        u_tau_std.append(u_tau_std_tmp)
        grad_u_mean.append(grad_u_mean_tmp)
        grad_u_std.append(grad_u_std_tmp)

    del _, u_tau_mean_tmp, u_tau_std_tmp, phi_tmp, grad_u_mean_tmp, grad_u_std_tmp, grad_u, u_tau, grad_u_tmp

    # plot u_tau and avg. grad_u
    fig, ax = plt.subplots(5, 2, figsize=(6, 5), sharex="col")
    for i in range(len(cases)):
        ax[0][0].plot(phi[i] + 180, u_tau_mean[i] / u_inf, label=f"mesh {i}")
        ax[0][1].plot(phi[i] + 180, u_tau_std[i] / u_inf)

        # du/dx
        ax[1][0].plot(phi[i] + 180, grad_u_mean[i][0, 0, :] * d / u_inf)
        ax[1][1].plot(phi[i] + 180, grad_u_std[i][0, 0, :] * d / u_inf)
        ax[2][0].plot(phi[i] + 180, grad_u_mean[i][1, 1, :] * d / u_inf)
        ax[2][1].plot(phi[i] + 180, grad_u_std[i][1, 1, :] * d / u_inf)
        ax[3][0].plot(phi[i] + 180, grad_u_mean[i][0, 1, :] * d / u_inf)
        ax[3][1].plot(phi[i] + 180, grad_u_std[i][0, 1, :] * d / u_inf)
        ax[4][0].plot(phi[i] + 180, grad_u_mean[i][1, 0, :] * d / u_inf)
        ax[4][1].plot(phi[i] + 180, grad_u_std[i][1, 0, :] * d / u_inf)

    # set all labels
    ax[-1][0].set_xlabel(r"$\phi$ $[^\circ]$")
    ax[-1][1].set_xlabel(r"$\phi$ $[^\circ]$")
    ax[0][0].set_ylabel(r"$\overline{u}_{\tau} / U_{\infty}$")
    ax[1][0].set_ylabel(r"$\partial_{\tilde{x}} \tilde{u}$")
    ax[2][0].set_ylabel(r"$\partial_{\tilde{y}} \tilde{v}$")
    ax[3][0].set_ylabel(r"$\partial_{\tilde{y}} \tilde{u}$")
    ax[4][0].set_ylabel(r"$\partial_{\tilde{x}} \tilde{v}$")

    # remaining mesh
    ax[0][0].set_title("mean")
    ax[0][1].set_title("std.")
    ax[-1][0].set_xlim(0, 360)
    ax[-1][1].set_xlim(0, 360)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, "u_tau_mean_vs_phi.png"))
    plt.close("all")

    # plot cp vs. angle phi, the cylinder coordinates remain the same (fig. 7)
    phi, cp_avg_over_z = [], []
    for i in range(len(cases)):
        _, p_tmp = load_sampling_planes(join(load_dir, cases[i]), "p_cylinder.raw")

        # compute mean cp distribution, cp and transform cartesian coordinates to angle, compare:
        # https://www.openfoam.com/documentation/guides/latest/doc/guide-fos-field-pressure.html
        cp_tmp = 2 * p_tmp.mean(-1).squeeze() / (rho * u_inf ** 2)
        phi_tmp, cp_avg_over_z_tmp = compute_phi(coordinates[i], cp_tmp)
        phi.append(phi_tmp)
        cp_avg_over_z.append(cp_avg_over_z_tmp)

    del _, cp_avg_over_z_tmp, phi_tmp, cp_tmp

    # plot avg. cp vs. cylinder angle
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, case in enumerate(cases):
        ax.plot(phi[i] + 180, cp_avg_over_z[i], label=f"mesh {i}")
    ax.set_xlabel(r"$\phi$ $[^\circ]$")
    ax.set_ylabel(r"$\overline{c}_p$")
    ax.set_xlim(0, 360)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, "cp_vs_phi.png"))
    plt.close("all")

    # plot UMean line for the wake in x-direction along the wake (fig. 8)
    line_samples_mean = []
    for c in cases:
        _, line_samples_mean_tmp = load_line_samples(join(load_dir, c), ["1"])
        line_samples_mean.append(line_samples_mean_tmp)
    del _, line_samples_mean_tmp

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex="col")
    for j in range(len(cases)):
        for i in range(len(line_samples_mean[j])):
            if i == 0:
                ax[0].plot(line_samples_mean[j][i][:, 0] / d, line_samples_mean[j][i][:, 7] / u_inf, label=f"mesh {j}")
            else:
                ax[0].plot(line_samples_mean[j][i][:, 0] / d, line_samples_mean[j][i][:, 7] / u_inf)
            ax[1].plot(line_samples_mean[j][i][:, 0] / d, line_samples_mean[j][i][:, -1] / u_inf ** 2)
    ax[0].set_ylabel(r"$\overline{u} / U_{\infty}$")
    ax[1].set_ylabel(r"$\overline{u^{\prime} u^{\prime}} / U_{\infty}^2$")
    ax[-1].set_xlabel(r"$x / D$")
    ax[-1].set_xlim(line_samples_mean[0][0][:, 0].min() / d, line_samples_mean[0][0][:, 0].max() / d)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_dir, f"U_wake_yd1.png"))
    plt.close("all")

    # plot UMean (fig. 9)
    coordinates, line_samples_mean = [], []
    for c in cases:
        coord, line_samples_mean_tmp = load_line_samples(join(load_dir, c), ["5", "7", "10"])
        coordinates.append(coord)
        line_samples_mean.append(line_samples_mean_tmp)
    del line_samples_mean_tmp, coord

    fig, ax = plt.subplots(1, 3, figsize=(6, 4), sharey="row")
    for col in range(len(line_samples_mean[0])):
        for i in range(len(cases)):
            ax[col].plot(line_samples_mean[i][col][:, 7] / u_inf, line_samples_mean[i][col][:, 0] / d,
                         label=f"mesh {i}")
        ax[col].set_xlabel(r"$\overline{u} / U_{\infty}$")
    ax[0].set_ylabel(r"$y / D$")
    for i, l in enumerate(["5", "7", "10"]):
        ax[i].set_title(rf"$x / D = {l}$")
    ax[0].set_ylim(0, 20)
    fig.tight_layout()
    ax[0].legend(ncol=1, loc="upper left")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, f"U_xd.png"))
    plt.close("all")

    # load line samples for Re stresses
    coordinates, line_samples_mean = [], []
    for c in cases:
        coord, line_samples_mean_tmp = load_line_samples(join(load_dir, c), ["106", "154", "202"])
        line_samples_mean.append(line_samples_mean_tmp)
        coordinates.append(coord)
    del line_samples_mean_tmp, coord

    # plot UMean and UPrime2Mean (fig. 10)      TODO: check assignment of Re stresses
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="col", figsize=(6, 6))
    for row in range(len(line_samples_mean[0])):
        for i in range(len(cases)):
            if row == 0:
                ax[row][0].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 7] / u_inf,
                                label=f"mesh {i}")
            else:
                ax[row][0].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 7] / u_inf)
            ax[row][1].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 10] / u_inf ** 2)

        ax[row][0].set_ylabel(r"$\overline{u} / U_{\infty}$")
        ax[row][1].set_ylabel(r"$\overline{u^{\prime} u^{\prime}} / U_{\infty}^2$")
    ax[-1][-1].set_xlabel(r"$y / D$")
    ax[-1][0].set_xlabel(r"$y / D$")
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.94)
    plt.savefig(join(save_dir, "U_xd_Uprime2Mean.png"))
    plt.close("all")

    # VPrime2Mean (fig. 11)
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="col", figsize=(6, 6))
    for row in range(len(line_samples_mean[0])):
        for i in range(len(cases)):
            if row == 0:
                ax[row][0].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 8] / u_inf,
                                label=f"mesh {i}")
            else:
                ax[row][0].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 8] / u_inf)
            ax[row][1].plot(coordinates[i][row][:, 0] / d, line_samples_mean[i][row][:, 13] / u_inf ** 2)

        ax[row][0].set_ylabel(r"$\overline{v} / U_{\infty}$")
        ax[row][1].set_ylabel(r"$\overline{v^{\prime} v^{\prime}} / U_{\infty}^2$")
    ax[-1][-1].set_xlabel(r"$y / D$")
    ax[-1][0].set_xlabel(r"$y / D$")
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.94)
    plt.savefig(join(save_dir, "V_xd_Vprime2Mean.png"))
    plt.close("all")

    # """
    # load and plot plane samplings (not present in paper)
    for c in range(len(cases)):
        coordinates, plane = load_sampling_planes(join(load_dir, cases[c]), "U_plane_xy.raw")

        # plot mean and std. deviation for all components
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(6, 6))
        for row in range(2):
            ax[row][0].tricontourf(coordinates[:, 0] / d, coordinates[:, 1] / d, plane[:, row, :].mean(-1))
            ax[row][1].tricontourf(coordinates[:, 0] / d, coordinates[:, 1] / d, plane[:, row, :].std(-1))
            ax[row][0].add_patch(Circle((cylinder_position[0] / d, cylinder_position[1] / d), radius=1 / 2,
                                        color="white"))
            ax[row][1].add_patch(Circle((cylinder_position[0] / d, cylinder_position[1] / d), radius=1 / 2,
                                        color="white"))
            ax[row][0].set_aspect("equal")
            ax[row][1].set_aspect("equal")
            ax[row][0].set_ylabel("$y/D$")
        ax[-1][0].set_xlabel("$x/D$")
        ax[-1][1].set_xlabel("$x/D$")
        ax[0][0].set_title(r"$\overline{u}$")
        ax[0][1].set_title(r"$\sigma\left(u\right)$")
        ax[1][0].set_title(r"$\overline{v}$")
        ax[1][1].set_title(r"$\sigma\left(v\right)$")
        fig.tight_layout()
        fig.subplots_adjust()
        plt.savefig(join(save_dir, f"mean_std_U_xy_plane_mesh{c}.png"))
        plt.close("all")

        # same for x-z-plane
        coordinates, plane = load_sampling_planes(join(load_dir, cases[c]), "U_plane_xz.raw")
        dim = [0, 2]
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(6, 4))
        for row in range(2):
            ax[row][0].tricontourf(coordinates[:, 0] / d, coordinates[:, 2] / d, plane[:, dim[row], :].mean(-1))
            ax[row][1].tricontourf(coordinates[:, 0] / d, coordinates[:, 2] / d, plane[:, dim[row], :].std(-1))
            ax[row][0].add_patch(Rectangle((cylinder_position[0] / d - 1/2, 0), width=1, height=pt.pi,
                                           color="white"))
            ax[row][1].add_patch(Rectangle((cylinder_position[0] / d - 1/2, 0), width=1, height=pt.pi,
                                           color="white"))
            ax[row][0].set_ylabel("$z/D$")
        ax[-1][0].set_xlabel("$x/D$")
        ax[-1][1].set_xlabel("$x/D$")
        ax[-1][0].set_ylim(0, pt.pi)
        ax[-1][1].set_ylim(0, pt.pi)
        ax[0][0].set_title(r"$\overline{u}$")
        ax[0][1].set_title(r"$\sigma\left(u\right)$")
        ax[1][0].set_title(r"$\overline{w}$")
        ax[1][1].set_title(r"$\sigma\left(w\right)$")
        fig.tight_layout()
        fig.subplots_adjust()
        plt.savefig(join(save_dir, f"mean_std_U_xz_plane_mesh{c}.png"))
        plt.close("all")
    # """

    """
    # plot probe positions as sanity check
    probes = get_probe_locations(load_dir)
    domain_xy = [[0, 2.4], [0, 2.0]]        # [[xmin, xmax], [ymin, ymax]]
    cylinder_pos = (0.8, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(probes[:, 0], probes[:, 1], color="red", zorder=10)
    ax.add_patch(Circle(cylinder_pos, radius=d/2, color="gray"))
    ax.add_patch(Rectangle((domain_xy[0][0], domain_xy[1][0]), width=domain_xy[0][1], height=domain_xy[1][1],
                            edgecolor="black", linewidth=2, facecolor="none"))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    """
