"""Helper functions for loading and processing simulation data.
"""

import sys
from os import environ
import torch as pt
from pandas import read_csv
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, environ.get("FLOWTORCH_INSTALL_DIR"))
from flowtorch.data import CSVDataloader


def compute_friction_velocity(path: str, filename: str, t_start: float, nu: float) -> tuple:
    loader = CSVDataloader.from_foam_surface(path, filename)
    times = loader.write_times
    times = [t for t in times if float(t) >= t_start]
    vertices = loader.vertices
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    area_normal = -pt.vstack(loader.load_snapshot(["area_x", "area_y", "area_z"], times[0])).T
    area = area_normal.norm(dim=1)
    normal = area_normal / area.unsqueeze(-1)
    u_tau, grad_u = [], []
    for t in times:
        grad = pt.vstack(
            loader.load_snapshot(["gradU_xx", "gradU_xy", "gradU_xz", "gradU_yx", "gradU_yy", "gradU_yz", "gradU_zx", "gradU_zy", "gradU_zz"], t)
        ).T
        grad_u.append(grad.reshape((grad.shape[0], 3, 3)))
        shear = nu * (grad_u[-1] + grad_u[-1].transpose(2, 1))
        projection_norm = (shear @ normal.unsqueeze(-1)).squeeze().norm(dim=1)
        u_tau.append(projection_norm.sqrt())
    return x, y, z, area, u_tau, grad_u


def load_force_coeffs(path, usecols=[0, 1, 4], names=["t", "cx", "cy"]):
    return read_csv(path, sep=r"\s+", comment="#", header=None, usecols=usecols, names=names)


if __name__ == '__main__':
    pass