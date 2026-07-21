# -*- coding: utf-8 -*-
"""Forward simulation only. Run this file before shurusange_inverse.py."""

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def simulate_forward_pde(
    L=10.0,
    Nz=400,
    D0=0.8,
    alpha=0.8,
    z1=4.0,
    z2=5.0,
    E=3e10,
    rho=2500.0,
    P0=1e5,
    Tp=1e-3,
    T_max=0.008,
    CFL=0.9,
    obs_z_list=None,
):
    """
    使用显式中心差分近似：
        rho A(z) u_tt = (E A(z) u_z)_z

    参数
    ----
    obs_z_list : list[float]
        需要输出速度时程的深度位置。本研究的反演数据仅使用 [0.0]。
    """
    if obs_z_list is None:
        obs_z_list = [0.0]

    dz = L / Nz
    z_nodes = np.linspace(0.0, L, Nz + 1)
    z_seg_centers = z_nodes[:-1] + 0.5 * dz

    A0 = math.pi * D0**2 / 4.0
    A_seg = np.full(Nz, A0, dtype=float)
    mask_def = (z_seg_centers >= z1) & (z_seg_centers <= z2)
    A_seg[mask_def] = alpha * A0

    # 节点等效截面积
    A_node = np.empty(Nz + 1, dtype=float)
    A_node[0] = A_seg[0]
    A_node[-1] = A_seg[-1]
    A_node[1:-1] = 0.5 * (A_seg[:-1] + A_seg[1:])

    # 集中质量和段刚度
    m_node = rho * A_node * dz
    k_seg = E * A_seg / dz

    c = math.sqrt(E / rho)
    dt = CFL * dz / c
    Nt = int(T_max / dt)
    t = np.arange(Nt + 1, dtype=float) * dt

    obs_idx = {}
    for z_obs in obs_z_list:
        j = int(round(z_obs / dz))
        j = max(0, min(Nz, j))
        obs_idx[float(z_obs)] = j

    v_dict = {
        float(z_obs): np.zeros(Nt + 1, dtype=float)
        for z_obs in obs_z_list
    }

    def impact_force(t_phys):
        if 0.0 <= t_phys <= Tp:
            return P0 * math.sin(math.pi * t_phys / Tp)
        return 0.0

    u_prev = np.zeros(Nz + 1, dtype=float)
    u_curr = np.zeros(Nz + 1, dtype=float)
    force_vec = np.zeros(Nz + 1, dtype=float)

    # 初始加速度
    force_vec[1:-1] = (
        k_seg[:-1] * (u_prev[:-2] - u_prev[1:-1])
        + k_seg[1:] * (u_prev[2:] - u_prev[1:-1])
    )
    force_vec[0] = k_seg[0] * (u_prev[1] - u_prev[0]) + impact_force(0.0)
    force_vec[-1] = k_seg[-1] * (u_prev[-2] - u_prev[-1])

    a0 = force_vec / m_node
    u_curr = u_prev + 0.5 * dt**2 * a0

    for z_obs, j in obs_idx.items():
        v_dict[z_obs][0] = 0.0
        v_dict[z_obs][1] = (u_curr[j] - u_prev[j]) / dt

    for n in range(1, Nt):
        tn = t[n]

        force_vec[1:-1] = (
            k_seg[:-1] * (u_curr[:-2] - u_curr[1:-1])
            + k_seg[1:] * (u_curr[2:] - u_curr[1:-1])
        )
        force_vec[0] = (
            k_seg[0] * (u_curr[1] - u_curr[0])
            + impact_force(tn)
        )
        force_vec[-1] = k_seg[-1] * (u_curr[-2] - u_curr[-1])

        acceleration = force_vec / m_node
        u_next = 2.0 * u_curr - u_prev + dt**2 * acceleration

        for z_obs, j in obs_idx.items():
            v_dict[z_obs][n + 1] = (u_next[j] - u_curr[j]) / dt

        u_prev, u_curr = u_curr, u_next

    return t, v_dict


def main():
    L, D0 = 10.0, 0.8
    E, rho = 3.0e10, 2500.0
    alpha_true, z1_true, z2_true = 0.8, 4.0, 5.0
    T_max, Nz, CFL = 0.008, 400, 0.9

    excitation_list = [
        {"name": "Impact 1", "P0": 5000, "Tp": 0.3e-3},
        {"name": "Impact 2", "P0": 5000, "Tp": 0.6e-3},
        {"name": "Impact 3", "P0": 5000, "Tp": 1.0e-3},
        {"name": "Impact 4", "P0": 5000, "Tp": 1.4e-3},
    ]
    tp_array = np.array([item["Tp"] for item in excitation_list])
    tp_min, tp_max = float(tp_array.min()), float(tp_array.max())
    if tp_max <= tp_min:
        raise ValueError("At least two distinct impact durations are required.")
    for item in excitation_list:
        item["Tp_cond"] = 2.0 * (item["Tp"] - tp_min) / (tp_max - tp_min) - 1.0

    P0_ref = 5000
    time_reference = None
    velocity_rows = []
    for item in excitation_list:
        time_values, velocity_dict = simulate_forward_pde(
            L=L, Nz=Nz, D0=D0,
            alpha=alpha_true, z1=z1_true, z2=z2_true,
            E=E, rho=rho, P0=item["P0"], Tp=item["Tp"],
            T_max=T_max, CFL=CFL, obs_z_list=[0.0],
        )
        if time_reference is None:
            time_reference = time_values
        elif not np.allclose(time_values, time_reference):
            raise RuntimeError("Time grids differ between impacts.")
        velocity_rows.append(velocity_dict[0.0])
        print(
            f"{item['name']}: P0={item['P0']:.1f} N, "
            f"Tp={item['Tp'] * 1e3:.1f} ms, Nt={len(time_values)-1}"
        )

    velocity_matrix = np.stack(velocity_rows, axis=0)
    output_path = Path(__file__).resolve().with_name("pile_forward_data.npz")
    np.savez_compressed(
        output_path,
        schema_version=np.array(1, dtype=np.int64),
        time=time_reference,
        velocity_top=velocity_matrix,
        impact_names=np.array([x["name"] for x in excitation_list], dtype="U32"),
        P0_values=np.array([x["P0"] for x in excitation_list]),
        Tp_values=np.array([x["Tp"] for x in excitation_list]),
        Tp_cond_values=np.array([x["Tp_cond"] for x in excitation_list]),
        L=np.array(L), D0=np.array(D0), E=np.array(E), rho=np.array(rho),
        T_max=np.array(T_max), Nz=np.array(Nz, dtype=np.int64),
        CFL=np.array(CFL), P0_ref=np.array(P0_ref),
        alpha_true=np.array(alpha_true),
        z1_true=np.array(z1_true), z2_true=np.array(z2_true),
    )
    print(f"\nForward data saved to: {output_path}")

    time_ms = time_reference * 1e3
    fig, axes = plt.subplots(len(excitation_list), 1, figsize=(9, 8.5), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (axis, item) in enumerate(zip(axes, excitation_list)):
        axis.plot(time_ms, velocity_matrix[index], linewidth=1.3)
        axis.set_ylabel(r"$v_{top}$ (m/s)")
        axis.set_title(f"{item['name']}: Tp={item['Tp']*1e3:.1f} ms")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Forward pile-head velocity responses")
    fig.tight_layout()
    figure_path = Path(__file__).resolve().with_name("pile_forward_responses.png")
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Forward figure saved to: {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
