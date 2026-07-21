# 
"""PINN inversion only. Reads pile_forward_data.npz; does not run forward simulation."""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# ============================================================
# 2. PINN：三子网 + 无量纲 PDE 残差
# ============================================================

class SubNet(nn.Module):
    """
    条件子网络：输入 (xi, tau, Tp_cond)，输出无量纲位移
        w(xi, tau; Tp_cond)。

    Tp_cond 为归一化后的锤击持续时间，取值范围为 [-1, 1]。
    """

    def __init__(self, in_dim=3, hidden_dim=64, n_hidden=6):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, xi, tau, tp_cond):
        if tp_cond.ndim == 0:
            tp_cond = tp_cond.reshape(1, 1).expand_as(xi)
        elif tp_cond.shape != xi.shape:
            tp_cond = tp_cond.expand_as(xi)

        x = torch.cat([xi, tau, tp_cond], dim=1)
        return self.model(x)


def pde_residual_nd(net, xi, tau, tp_cond):
    """
    给定激励条件 Tp_cond 时的无量纲 PDE 残差：
        r = w_tau_tau - w_xi_xi

    自动微分仅对 xi 和 tau 求导，Tp_cond 作为已知条件量保持不变。
    """
    w = net(xi, tau, tp_cond)

    w_tau = autograd.grad(
        w,
        tau,
        grad_outputs=torch.ones_like(w),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_tautau = autograd.grad(
        w_tau,
        tau,
        grad_outputs=torch.ones_like(w_tau),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_xi = autograd.grad(
        w,
        xi,
        grad_outputs=torch.ones_like(w),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_xixi = autograd.grad(
        w_xi,
        xi,
        grad_outputs=torch.ones_like(w_xi),
        create_graph=True,
        retain_graph=True,
    )[0]

    return w_tautau - w_xixi


def moving_average(values, window=500):
    """对一维序列进行移动平均。"""
    values = np.asarray(values, dtype=float)

    if window <= 1 or len(values) < window:
        return values.copy()

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


# ============================================================
# 3. 主程序
# ============================================================

def main():
    torch.set_default_dtype(torch.float32)

    data_path = Path(__file__).resolve().with_name("pile_forward_data.npz")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Forward data not found: {data_path}\n"
            "Run shurusange_forward.py first."
        )

    required = {
        "time", "velocity_top", "impact_names", "P0_values",
        "Tp_values", "Tp_cond_values", "L", "D0", "E", "rho",
        "P0_ref", "alpha_true", "z1_true", "z2_true",
    }
    with np.load(data_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError("Missing forward-data fields: " + ", ".join(missing))
        t_np = np.asarray(data["time"], dtype=float)
        velocity_top = np.asarray(data["velocity_top"], dtype=float)
        impact_names = np.asarray(data["impact_names"]).astype(str)
        p0_values = np.asarray(data["P0_values"], dtype=float)
        tp_values = np.asarray(data["Tp_values"], dtype=float)
        tp_cond_values = np.asarray(data["Tp_cond_values"], dtype=float)
        L = float(data["L"].item())
        D0 = float(data["D0"].item())
        E = float(data["E"].item())
        rho = float(data["rho"].item())
        P0_ref = float(data["P0_ref"].item())
        alpha_true = float(data["alpha_true"].item())
        z1_true = float(data["z1_true"].item())
        z2_true = float(data["z2_true"].item())

    n_impacts = len(impact_names)
    expected_shape = (n_impacts, len(t_np))
    if velocity_top.shape != expected_shape:
        raise ValueError(
            f"velocity_top shape {velocity_top.shape}; expected {expected_shape}."
        )
    if not (len(p0_values) == len(tp_values) == len(tp_cond_values) == n_impacts):
        raise ValueError("The numbers of responses and impact parameters differ.")
    if len(t_np) < 2 or not np.all(np.diff(t_np) > 0.0):
        raise ValueError("The time vector must be strictly increasing.")

    A0 = math.pi * D0**2 / 4.0
    c = math.sqrt(E / rho)
    xi1_true, xi2_true = z1_true / L, z2_true / L
    xi_c_true = 0.5 * (xi1_true + xi2_true)
    lam_true = xi2_true - xi1_true
    T0 = L / c
    U0 = L * P0_ref / (E * A0)
    v_scale = U0 * c / L
    tau_np = t_np / T0
    tau_max = float(tau_np[-1])

    forward_results = [
        {
            "name": str(impact_names[k]),
            "P0": float(p0_values[k]),
            "Tp": float(tp_values[k]),
            "Tp_cond": float(tp_cond_values[k]),
            "t_np": t_np,
            "v_top_np": velocity_top[k],
        }
        for k in range(n_impacts)
    ]

    print(f"Loaded forward data: {data_path}")
    print(f"Number of impacts = {n_impacts}")
    print(f"Wave speed c = {c:.2f} m/s, tau_max={tau_max:.3f}")
    print(
        f"True parameters: alpha={alpha_true:.3f}, "
        f"xi1={xi1_true:.3f}, xi2={xi2_true:.3f}, "
        f"xi_c={xi_c_true:.3f}, lam={lam_true:.3f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    impact_data_list = []
    for impact_index, result in enumerate(forward_results):
        impact_data_list.append({
            "impact_index": impact_index,
            "name": result["name"],
            "P0": result["P0"],
            "Tp": result["Tp"],
            "Tp_cond": result["Tp_cond"],
            "tau": torch.tensor(
                tau_np, dtype=torch.float32, device=device
            ).view(-1, 1),
            "v_star": torch.tensor(
                result["v_top_np"] / v_scale,
                dtype=torch.float32, device=device,
            ).view(-1, 1),
            "v_top_np": result["v_top_np"],
        })

    # --------------------------------------------------------
    # 3.6 所有激励共享一套条件三子网
    # 三个网络均以 (xi, tau, Tp_cond) 为输入。
    conditional_nets = nn.ModuleList([
        SubNet(in_dim=3).to(device),
        SubNet(in_dim=3).to(device),
        SubNet(in_dim=3).to(device),
    ])

    # --------------------------------------------------------
    # 3.7 所有激励共享的反演参数
    # --------------------------------------------------------
    logit_alpha = nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    theta_c = nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    theta_l = nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32, device=device)
    )

    def get_alpha_xi1_xi2():
        alpha_hat = 0.5 + 0.45 * torch.sigmoid(logit_alpha)
        lam_hat = 0.05 + 0.1 * torch.sigmoid(theta_l)
        xi_c_hat = 0.3 + 0.3 * torch.sigmoid(theta_c)

        xi1_hat = xi_c_hat - 0.5 * lam_hat
        xi2_hat = xi_c_hat + 0.5 * lam_hat

        xi1_hat = torch.clamp(xi1_hat, 0.02, 0.98)
        xi2_hat = torch.clamp(xi2_hat, 0.02, 0.98)
        xi2_hat = torch.maximum(xi2_hat, xi1_hat + 1e-3)

        return alpha_hat, xi1_hat, xi2_hat, xi_c_hat, lam_hat

    params = (
        list(conditional_nets.parameters())
        + [logit_alpha, theta_c, theta_l]
    )

    mse = nn.MSELoss()

    # --------------------------------------------------------
    # 3.8 损失权重和缺陷反射时间窗
    # --------------------------------------------------------
    w_pde = 50.0
    w_data = 200.0
    w_if_u = 5.0
    w_if_N = 200.0
    w_bc_bottom = 1.0
    w_bc_top = 1.0
    w_ic = 1.0

    # 桩顶缺陷反射宽时间窗加权
    # 该窗口覆盖较宽的缺陷反射波段，不针对某个精确缺陷位置。
    use_defect_window_weight = True
    defect_window_start = 2.0e-3
    defect_window_end = 4.2e-3
    defect_window_edge = 0.20e-3
    defect_window_gain = 5.0

    def P_star_tau(tau_nd, P0_k, Tp_k):
        """
        第 k 次锤击的无量纲桩顶荷载：
            P*_k = P_k / P0_ref
        """
        t_phys = tau_nd * T0
        amplitude_ratio = P0_k / P0_ref

        active_force = (
            amplitude_ratio
            * torch.sin(math.pi * t_phys / Tp_k)
        )

        return torch.where(
            (t_phys >= 0.0) & (t_phys <= Tp_k),
            active_force,
            torch.zeros_like(t_phys),
        )

    def defect_window_weight(t_phys):
        """
        平滑矩形时间窗：
            2.0 ms ~ 4.2 ms 内权重提高，
        避免使用精确反射峰位置作为强先验。
        """
        if not use_defect_window_weight:
            return torch.ones_like(t_phys)

        left_gate = torch.sigmoid(
            (t_phys - defect_window_start) / defect_window_edge
        )
        right_gate = torch.sigmoid(
            (defect_window_end - t_phys) / defect_window_edge
        )
        window = left_gate * right_gate

        return 1.0 + defect_window_gain * window

    # ========================================================
    # 3.9 通用：计算单次激励的各项损失
    # ========================================================

    def compute_single_impact_losses(
        net_group,
        impact_data,
        fixed_points,
        data_indices=None,
    ):
        """
        使用共享条件三子网，计算某一次激励对应的 PDE、数据、边界、
        接口和初始损失。

        net_group:
            所有激励共享的 [net1, net2, net3]
        impact_data:
            当前激励的桩顶观测、P0、Tp 和 Tp_cond
        fixed_points:
            固定采样点字典
        data_indices:
            Adam 阶段的固定数据索引；None 表示使用全数据。
        """
        net1, net2, net3 = net_group

        def make_condition(reference_tensor):
            return torch.full_like(
                reference_tensor,
                fill_value=impact_data["Tp_cond"],
            )

        (
            alpha_hat,
            xi1_hat,
            xi2_hat,
            _,
            _,
        ) = get_alpha_xi1_xi2()

        # ----------------------------------------------------
        # (1) PDE 残差：三个子域采用独立的可微局部坐标映射
        # ----------------------------------------------------
        # s_fj 始终固定在 (0, 1)；实际坐标 xi_fj 随当前 xi1_hat、
        # xi2_hat 连续移动。不能对 xi_fj 再执行 detach，否则会再次
        # 切断 PDE 损失到界面位置参数的梯度。
        s_f1 = fixed_points["s_f1"].clone().detach()
        s_f2 = fixed_points["s_f2"].clone().detach()
        s_f3 = fixed_points["s_f3"].clone().detach()

        tau_f1 = (
            fixed_points["tau_f1"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        tau_f2 = (
            fixed_points["tau_f2"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        tau_f3 = (
            fixed_points["tau_f3"]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        # 子域1: [0, xi1_hat]
        # 子域2: [xi1_hat, xi2_hat]
        # 子域3: [xi2_hat, 1]
        xi_f1 = s_f1 * xi1_hat
        xi_f2 = xi1_hat + s_f2 * (xi2_hat - xi1_hat)
        xi_f3 = xi2_hat + s_f3 * (1.0 - xi2_hat)

        r1 = pde_residual_nd(
            net1,
            xi_f1,
            tau_f1,
            make_condition(xi_f1),
        )
        r2 = pde_residual_nd(
            net2,
            xi_f2,
            tau_f2,
            make_condition(xi_f2),
        )
        r3 = pde_residual_nd(
            net3,
            xi_f3,
            tau_f3,
            make_condition(xi_f3),
        )

        # 三个子域等权，避免优化器通过压缩某个子域长度来人为降低
        # 该子域对 PDE 损失的贡献。
        loss_pde = (
            r1.pow(2).mean()
            + r2.pow(2).mean()
            + r3.pow(2).mean()
        ) / 3.0

        # ----------------------------------------------------
        # (2) 桩顶速度数据损失
        # ----------------------------------------------------
        if data_indices is None:
            tau_data = impact_data["tau"]
            v_star_data = impact_data["v_star"]
        else:
            tau_data = impact_data["tau"][data_indices]
            v_star_data = impact_data["v_star"][data_indices]

        tau_data = (
            tau_data
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_data = torch.zeros_like(tau_data)

        w_top_data = net1(
            xi_top_data,
            tau_data,
            make_condition(xi_top_data),
        )
        w_top_tau = autograd.grad(
            w_top_data,
            tau_data,
            grad_outputs=torch.ones_like(w_top_data),
            create_graph=True,
        )[0]

        data_residual = w_top_tau - v_star_data
        t_data_phys = tau_data * T0
        time_weight = defect_window_weight(t_data_phys)

        loss_data = (
            time_weight * data_residual.pow(2)
        ).mean()

        # ----------------------------------------------------
        # (3) 桩底自由边界
        # ----------------------------------------------------
        tau_bottom = (
            fixed_points["tau_bottom"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_bottom = torch.ones_like(
            tau_bottom,
            requires_grad=True,
        )

        w_bottom = net3(
            xi_bottom,
            tau_bottom,
            make_condition(xi_bottom),
        )
        w_bottom_xi = autograd.grad(
            w_bottom,
            xi_bottom,
            grad_outputs=torch.ones_like(w_bottom),
            create_graph=True,
        )[0]

        loss_bc_bottom = mse(
            w_bottom_xi,
            torch.zeros_like(w_bottom_xi),
        )

        # ----------------------------------------------------
        # (4) 桩顶锤击边界
        # ----------------------------------------------------
        tau_top = (
            fixed_points["tau_top_list"][
                impact_data["impact_index"]
            ]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_bc = torch.zeros_like(
            tau_top,
            requires_grad=True,
        )

        w_top_bc = net1(
            xi_top_bc,
            tau_top,
            make_condition(xi_top_bc),
        )
        w_top_xi = autograd.grad(
            w_top_bc,
            xi_top_bc,
            grad_outputs=torch.ones_like(w_top_bc),
            create_graph=True,
        )[0]

        p_star = P_star_tau(
            tau_top,
            impact_data["P0"],
            impact_data["Tp"],
        )
        loss_bc_top = mse(
            w_top_xi + p_star,
            torch.zeros_like(w_top_xi),
        )

        # ----------------------------------------------------
        # (5) 两处接口：位移连续 + 轴力连续
        # ----------------------------------------------------
        tau_interface_base = fixed_points["tau_interface"]

        # z = z1
        tau_if1 = (
            tau_interface_base
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi1_interface = xi1_hat.expand_as(tau_if1)

        w1_at_xi1 = net1(
            xi1_interface,
            tau_if1,
            make_condition(xi1_interface),
        )
        w2_at_xi1 = net2(
            xi1_interface,
            tau_if1,
            make_condition(xi1_interface),
        )

        w1_xi_at_xi1 = autograd.grad(
            w1_at_xi1,
            xi1_interface,
            grad_outputs=torch.ones_like(w1_at_xi1),
            create_graph=True,
        )[0]
        w2_xi_at_xi1 = autograd.grad(
            w2_at_xi1,
            xi1_interface,
            grad_outputs=torch.ones_like(w2_at_xi1),
            create_graph=True,
        )[0]

        loss_if_u1 = mse(w1_at_xi1, w2_at_xi1)
        loss_if_N1 = mse(
            w1_xi_at_xi1,
            alpha_hat * w2_xi_at_xi1,
        )

        # z = z2
        tau_if2 = (
            tau_interface_base
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi2_interface = xi2_hat.expand_as(tau_if2)

        w2_at_xi2 = net2(
            xi2_interface,
            tau_if2,
            make_condition(xi2_interface),
        )
        w3_at_xi2 = net3(
            xi2_interface,
            tau_if2,
            make_condition(xi2_interface),
        )

        w2_xi_at_xi2 = autograd.grad(
            w2_at_xi2,
            xi2_interface,
            grad_outputs=torch.ones_like(w2_at_xi2),
            create_graph=True,
        )[0]
        w3_xi_at_xi2 = autograd.grad(
            w3_at_xi2,
            xi2_interface,
            grad_outputs=torch.ones_like(w3_at_xi2),
            create_graph=True,
        )[0]

        loss_if_u2 = mse(w2_at_xi2, w3_at_xi2)
        loss_if_N2 = mse(
            alpha_hat * w2_xi_at_xi2,
            w3_xi_at_xi2,
        )

        loss_if_u = 0.5 * (loss_if_u1 + loss_if_u2)
        loss_if_N = 0.5 * (loss_if_N1 + loss_if_N2)

        # ----------------------------------------------------
        # (6) 初始条件：w=0, w_tau=0
        #     与 PDE 配点相同，采用三个子域的可微局部坐标映射。
        # ----------------------------------------------------
        s_ic1 = fixed_points["s_ic1"].clone().detach()
        s_ic2 = fixed_points["s_ic2"].clone().detach()
        s_ic3 = fixed_points["s_ic3"].clone().detach()

        xi_ic1 = s_ic1 * xi1_hat
        xi_ic2 = xi1_hat + s_ic2 * (xi2_hat - xi1_hat)
        xi_ic3 = xi2_hat + s_ic3 * (1.0 - xi2_hat)

        tau_ic1 = torch.zeros_like(s_ic1, requires_grad=True)
        tau_ic2 = torch.zeros_like(s_ic2, requires_grad=True)
        tau_ic3 = torch.zeros_like(s_ic3, requires_grad=True)

        w_ic1 = net1(
            xi_ic1,
            tau_ic1,
            make_condition(xi_ic1),
        )
        w_ic2 = net2(
            xi_ic2,
            tau_ic2,
            make_condition(xi_ic2),
        )
        w_ic3 = net3(
            xi_ic3,
            tau_ic3,
            make_condition(xi_ic3),
        )

        w_ic1_tau = autograd.grad(
            w_ic1,
            tau_ic1,
            grad_outputs=torch.ones_like(w_ic1),
            create_graph=True,
        )[0]
        w_ic2_tau = autograd.grad(
            w_ic2,
            tau_ic2,
            grad_outputs=torch.ones_like(w_ic2),
            create_graph=True,
        )[0]
        w_ic3_tau = autograd.grad(
            w_ic3,
            tau_ic3,
            grad_outputs=torch.ones_like(w_ic3),
            create_graph=True,
        )[0]

        loss_ic1 = 0.5 * (
            w_ic1.pow(2).mean()
            + w_ic1_tau.pow(2).mean()
        )
        loss_ic2 = 0.5 * (
            w_ic2.pow(2).mean()
            + w_ic2_tau.pow(2).mean()
        )
        loss_ic3 = 0.5 * (
            w_ic3.pow(2).mean()
            + w_ic3_tau.pow(2).mean()
        )

        loss_ic = (loss_ic1 + loss_ic2 + loss_ic3) / 3.0

        return {
            "pde": loss_pde,
            "data": loss_data,
            "bc_bottom": loss_bc_bottom,
            "bc_top": loss_bc_top,
            "if_u": loss_if_u,
            "if_N": loss_if_N,
            "ic": loss_ic,
        }

    def weighted_total_loss(loss_components):
        return (
            w_pde * loss_components["pde"]
            + w_data * loss_components["data"]
            + w_if_u * loss_components["if_u"]
            + w_if_N * loss_components["if_N"]
            + w_bc_bottom * loss_components["bc_bottom"]
            + w_bc_top * loss_components["bc_top"]
            + w_ic * loss_components["ic"]
        )

    def empty_loss_accumulator():
        return {
            "loss": 0.0,
            "loss_pde": 0.0,
            "loss_data": 0.0,
            "loss_bc_bottom": 0.0,
            "loss_bc_top": 0.0,
            "loss_if_u": 0.0,
            "loss_if_N": 0.0,
            "loss_ic": 0.0,
        }

    def add_detached_losses(accumulator, total_loss, components, scale):
        accumulator["loss"] += (
            float(total_loss.detach().item()) * scale
        )
        accumulator["loss_pde"] += (
            float(components["pde"].detach().item()) * scale
        )
        accumulator["loss_data"] += (
            float(components["data"].detach().item()) * scale
        )
        accumulator["loss_bc_bottom"] += (
            float(components["bc_bottom"].detach().item()) * scale
        )
        accumulator["loss_bc_top"] += (
            float(components["bc_top"].detach().item()) * scale
        )
        accumulator["loss_if_u"] += (
            float(components["if_u"].detach().item()) * scale
        )
        accumulator["loss_if_N"] += (
            float(components["if_N"].detach().item()) * scale
        )
        accumulator["loss_ic"] += (
            float(components["ic"].detach().item()) * scale
        )

    def append_parameter_info(info):
        with torch.no_grad():
            (
                alpha_hat,
                xi1_hat,
                xi2_hat,
                xi_c_hat,
                lam_hat,
            ) = get_alpha_xi1_xi2()

        info.update({
            "alpha_hat": float(alpha_hat.detach().item()),
            "xi1_hat": float(xi1_hat.detach().item()),
            "xi2_hat": float(xi2_hat.detach().item()),
            "xi_c_hat": float(xi_c_hat.detach().item()),
            "lam_hat": float(lam_hat.detach().item()),
        })

        return info

    # ========================================================
    # 3.10 Adam 阶段采样=====================================
    def split_count_three(total_count):
        """将总配点数尽可能均匀地分配给三个子域。"""
        base = total_count // 3
        return base, base, total_count - 2 * base

    # --------------------------------------------------------
    # 自适应采样参数
    # --------------------------------------------------------
    adaptive_warmup_epochs = 5000
    adaptive_update_interval = 5000
    adaptive_candidate_factor = 3
    adaptive_high_fraction = 0.60
    adaptive_batch_size = 512

    # 桩顶边界分层比例：锤击作用段、缺陷反射段、完整时域。
    top_force_fraction = 0.30
    top_reflection_fraction = 0.40

    def allocate_top_counts(total_count):
        n_force = int(round(total_count * top_force_fraction))
        n_reflection = int(
            round(total_count * top_reflection_fraction)
        )
        n_global = total_count - n_force - n_reflection
        return n_force, n_reflection, n_global

    def sample_tau_interval(n_points, t_start, t_end):
        """在给定物理时间区间均匀采样，并转换为无量纲时间。"""
        if n_points <= 0:
            return torch.empty(0, 1, device=device)

        t_final = float(t_np[-1])
        t_start = max(0.0, min(float(t_start), t_final))
        t_end = max(t_start, min(float(t_end), t_final))

        if t_end <= t_start:
            t_phys = torch.full(
                (n_points, 1),
                fill_value=t_start,
                device=device,
            )
        else:
            t_phys = (
                t_start
                + torch.rand(n_points, 1, device=device)
                * (t_end - t_start)
            )

        return t_phys / T0

    def shuffled_concat(tensor_list):
        result = torch.cat(tensor_list, dim=0)
        if result.shape[0] > 1:
            permutation = torch.randperm(
                result.shape[0], device=device
            )
            result = result[permutation]
        return result.detach()

    def make_stratified_top_times(impact_data, total_count):
        """生成桩顶边界的初始分层时间点。"""
        n_force, n_reflection, n_global = allocate_top_counts(
            total_count
        )

        return shuffled_concat([
            sample_tau_interval(
                n_force, 0.0, impact_data["Tp"]
            ),
            sample_tau_interval(
                n_reflection,
                defect_window_start,
                defect_window_end,
            ),
            sample_tau_interval(
                n_global, 0.0, float(t_np[-1])
            ),
        ])

    def select_mixed_indices(scores, n_select):
        """选择高残差点，同时保留均匀随机点保证全域覆盖。"""
        scores = scores.reshape(-1)
        n_candidates = int(scores.numel())
        n_select = min(int(n_select), n_candidates)

        if n_select <= 0:
            return torch.empty(
                0, dtype=torch.long, device=device
            )

        n_high = int(round(n_select * adaptive_high_fraction))
        n_high = max(0, min(n_high, n_select))
        n_uniform = n_select - n_high

        if n_high > 0:
            high_indices = torch.topk(
                scores, k=n_high, largest=True
            ).indices
        else:
            high_indices = torch.empty(
                0, dtype=torch.long, device=device
            )

        if n_uniform > 0:
            available_mask = torch.ones(
                n_candidates, dtype=torch.bool, device=device
            )
            available_mask[high_indices] = False
            available = torch.nonzero(
                available_mask, as_tuple=False
            ).reshape(-1)
            random_order = torch.randperm(
                available.numel(), device=device
            )
            uniform_indices = available[
                random_order[:n_uniform]
            ]
        else:
            uniform_indices = torch.empty(
                0, dtype=torch.long, device=device
            )

        selected = torch.cat(
            [high_indices, uniform_indices], dim=0
        )
        if selected.numel() > 1:
            selected = selected[
                torch.randperm(selected.numel(), device=device)
            ]
        return selected

    def map_reference_to_subdomain(s_value, domain_index, xi1, xi2):
        if domain_index == 1:
            return s_value * xi1
        if domain_index == 2:
            return xi1 + s_value * (xi2 - xi1)
        if domain_index == 3:
            return xi2 + s_value * (1.0 - xi2)
        raise ValueError("domain_index 必须为 1、2 或 3。")

    def score_pde_candidates(domain_index, s_candidates, tau_candidates):
        """计算四种激励下的平均 PDE 点残差。"""
        with torch.no_grad():
            _, xi1_now, xi2_now, _, _ = get_alpha_xi1_xi2()
            xi1_value = xi1_now.detach()
            xi2_value = xi2_now.detach()

        net = conditional_nets[domain_index - 1]
        score_chunks = []

        for start in range(0, s_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                s_candidates.shape[0],
            )
            s_batch = s_candidates[start:end]
            tau_base = tau_candidates[start:end]
            batch_score = torch.zeros_like(s_batch)

            for impact_data in impact_data_list:
                # 候选点评分不参与参数优化，因此此处可以把实际坐标
                # 变为叶子张量；正式训练使用的映射仍保持可微。
                xi_batch = map_reference_to_subdomain(
                    s_batch,
                    domain_index,
                    xi1_value,
                    xi2_value,
                ).detach().requires_grad_(True)
                tau_batch = (
                    tau_base.clone().detach().requires_grad_(True)
                )
                tp_batch = torch.full_like(
                    xi_batch,
                    fill_value=impact_data["Tp_cond"],
                )

                residual = pde_residual_nd(
                    net, xi_batch, tau_batch, tp_batch
                )
                batch_score = (
                    batch_score + residual.detach().pow(2)
                )

            batch_score = batch_score / n_impacts
            score_chunks.append(batch_score)

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def score_top_bc_candidates(impact_data, tau_candidates):
        """计算某次激励的桩顶力边界残差。"""
        score_chunks = []
        net1 = conditional_nets[0]

        for start in range(0, tau_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                tau_candidates.shape[0],
            )
            tau_batch = tau_candidates[start:end].detach()
            xi_batch = torch.zeros_like(
                tau_batch, requires_grad=True
            )
            tp_batch = torch.full_like(
                tau_batch,
                fill_value=impact_data["Tp_cond"],
            )

            w_top = net1(xi_batch, tau_batch, tp_batch)
            w_top_xi = autograd.grad(
                w_top,
                xi_batch,
                grad_outputs=torch.ones_like(w_top),
                create_graph=False,
            )[0]
            p_star = P_star_tau(
                tau_batch,
                impact_data["P0"],
                impact_data["Tp"],
            )
            score_chunks.append(
                (w_top_xi + p_star).detach().pow(2)
            )

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def score_interface_candidates(tau_candidates):
        """计算四种激励、两个界面的综合连续条件残差。"""
        with torch.no_grad():
            alpha_now, xi1_now, xi2_now, _, _ = (
                get_alpha_xi1_xi2()
            )
            alpha_value = alpha_now.detach()
            xi1_value = float(xi1_now.detach().cpu())
            xi2_value = float(xi2_now.detach().cpu())

        net1, net2, net3 = conditional_nets
        score_chunks = []

        for start in range(0, tau_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                tau_candidates.shape[0],
            )
            tau_batch = tau_candidates[start:end].detach()
            batch_score = torch.zeros_like(tau_batch)

            for impact_data in impact_data_list:
                tp_batch = torch.full_like(
                    tau_batch,
                    fill_value=impact_data["Tp_cond"],
                )
                xi1_batch = torch.full_like(
                    tau_batch,
                    fill_value=xi1_value,
                    requires_grad=True,
                )
                xi2_batch = torch.full_like(
                    tau_batch,
                    fill_value=xi2_value,
                    requires_grad=True,
                )

                w1_xi1 = net1(xi1_batch, tau_batch, tp_batch)
                w2_xi1 = net2(xi1_batch, tau_batch, tp_batch)
                dw1_xi1 = autograd.grad(
                    w1_xi1,
                    xi1_batch,
                    grad_outputs=torch.ones_like(w1_xi1),
                    create_graph=False,
                )[0]
                dw2_xi1 = autograd.grad(
                    w2_xi1,
                    xi1_batch,
                    grad_outputs=torch.ones_like(w2_xi1),
                    create_graph=False,
                )[0]

                w2_xi2 = net2(xi2_batch, tau_batch, tp_batch)
                w3_xi2 = net3(xi2_batch, tau_batch, tp_batch)
                dw2_xi2 = autograd.grad(
                    w2_xi2,
                    xi2_batch,
                    grad_outputs=torch.ones_like(w2_xi2),
                    create_graph=False,
                )[0]
                dw3_xi2 = autograd.grad(
                    w3_xi2,
                    xi2_batch,
                    grad_outputs=torch.ones_like(w3_xi2),
                    create_graph=False,
                )[0]

                # 用正式损失中的界面权重进行候选点评分，避免位移
                # 连续残差与轴力连续残差的重要性在选点阶段失配。
                displacement_score = 0.5 * (
                    (w1_xi1 - w2_xi1).detach().pow(2)
                    + (w2_xi2 - w3_xi2).detach().pow(2)
                )
                force_score = 0.5 * (
                    (dw1_xi1 - alpha_value * dw2_xi1)
                    .detach()
                    .pow(2)
                    + (alpha_value * dw2_xi2 - dw3_xi2)
                    .detach()
                    .pow(2)
                )
                residual_score = (
                    w_if_u * displacement_score
                    + w_if_N * force_score
                )
                batch_score = batch_score + residual_score

            batch_score = batch_score / n_impacts
            score_chunks.append(batch_score)

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def adapt_pde_points(fixed_points, domain_counts):
        """更新三个子域的 PDE 参考点。"""
        for domain_index, n_target in enumerate(
            domain_counts, start=1
        ):
            n_candidates = max(
                n_target,
                adaptive_candidate_factor * n_target,
            )
            s_candidates = torch.rand(
                n_candidates, 1, device=device
            )
            tau_candidates = torch.rand(
                n_candidates, 1, device=device
            ) * tau_max
            scores = score_pde_candidates(
                domain_index, s_candidates, tau_candidates
            )
            selected = select_mixed_indices(scores, n_target)
            fixed_points[f"s_f{domain_index}"] = (
                s_candidates[selected].detach()
            )
            fixed_points[f"tau_f{domain_index}"] = (
                tau_candidates[selected].detach()
            )

    def adapt_top_boundary_points(fixed_points, total_count):
        """按三层比例分别选择每次激励的高残差桩顶时间点。"""
        target_counts = allocate_top_counts(total_count)
        tau_top_list = []

        for impact_data in impact_data_list:
            intervals = [
                (0.0, impact_data["Tp"]),
                (defect_window_start, defect_window_end),
                (0.0, float(t_np[-1])),
            ]
            selected_layers = []

            for n_target, (t_start, t_end) in zip(
                target_counts, intervals
            ):
                n_candidates = max(
                    n_target,
                    adaptive_candidate_factor * n_target,
                )
                tau_candidates = sample_tau_interval(
                    n_candidates, t_start, t_end
                )
                scores = score_top_bc_candidates(
                    impact_data, tau_candidates
                )
                selected = select_mixed_indices(
                    scores, n_target
                )
                selected_layers.append(
                    tau_candidates[selected].detach()
                )

            tau_top_list.append(
                shuffled_concat(selected_layers)
            )

        fixed_points["tau_top_list"] = tau_top_list

    def adapt_interface_points(fixed_points, n_target):
        """更新界面条件的时间点。"""
        n_candidates = max(
            n_target,
            adaptive_candidate_factor * n_target,
        )
        tau_candidates = torch.rand(
            n_candidates, 1, device=device
        ) * tau_max
        scores = score_interface_candidates(tau_candidates)
        selected = select_mixed_indices(scores, n_target)
        fixed_points["tau_interface"] = (
            tau_candidates[selected].detach()
        )

    def refresh_adaptive_points(
        fixed_points,
        domain_counts,
        n_top,
        n_interface,
        phase_label,
    ):
        """依次更新 PDE、桩顶边界和界面采样点。"""
        adapt_pde_points(fixed_points, domain_counts)
        adapt_top_boundary_points(fixed_points, n_top)
        adapt_interface_points(fixed_points, n_interface)
        print(
            f"[Adaptive sampling] {phase_label}: "
            "PDE、桩顶边界和界面点已更新并固定。"
        )

    N_f_adam = 8000
    N_bc_adam = 2000
    N_if_adam = 5000
    N_ic_adam = 2000
    N_data_adam = 5000

    # 分别持有自己的固定参考坐标 s∈(0,1)。
    N_f1_adam, N_f2_adam, N_f3_adam = split_count_three(
        N_f_adam
    )
    N_ic1_adam, N_ic2_adam, N_ic3_adam = split_count_three(
        N_ic_adam
    )

    with torch.no_grad():
        adam_fixed_points = {
            "s_f1": torch.rand(
                N_f1_adam, 1, device=device
            ),
            "s_f2": torch.rand(
                N_f2_adam, 1, device=device
            ),
            "s_f3": torch.rand(
                N_f3_adam, 1, device=device
            ),
            "tau_f1": torch.rand(
                N_f1_adam, 1, device=device
            ) * tau_max,
            "tau_f2": torch.rand(
                N_f2_adam, 1, device=device
            ) * tau_max,
            "tau_f3": torch.rand(
                N_f3_adam, 1, device=device
            ) * tau_max,
            "tau_bottom": torch.rand(
                N_bc_adam, 1, device=device
            ) * tau_max,
            # 桩顶边界按每次激励分别分层采样：
            # 锤击作用段 + 缺陷反射时间窗 + 全时间域。
            "tau_top_list": [
                make_stratified_top_times(
                    impact_data,
                    N_bc_adam,
                )
                for impact_data in impact_data_list
            ],
            "tau_interface": torch.rand(
                N_if_adam, 1, device=device
            ) * tau_max,
            "s_ic1": torch.rand(
                N_ic1_adam, 1, device=device
            ),
            "s_ic2": torch.rand(
                N_ic2_adam, 1, device=device
            ),
            "s_ic3": torch.rand(
                N_ic3_adam, 1, device=device
            ),
        }

        adam_data_indices = []
        for impact_data in impact_data_list:
            total_points = impact_data["tau"].shape[0]

            if N_data_adam >= total_points:
                indices = torch.arange(
                    total_points,
                    device=device,
                )
            else:
                # 不放回抽样，避免同一个时间点重复过多
                indices = torch.randperm(
                    total_points,
                    device=device,
                )[:N_data_adam]

            adam_data_indices.append(indices)

    optimizer_adam = optim.Adam(params, lr=1e-3)
    history = []

    def train_step_adam():
        optimizer_adam.zero_grad()
        accumulator = empty_loss_accumulator()
        impact_data_loss_values = []

        # 逐次反传，降低多组波场同时保留计算图导致的显存压力
        for k in range(n_impacts):
            components = compute_single_impact_losses(
                conditional_nets,
                impact_data_list[k],
                adam_fixed_points,
                data_indices=adam_data_indices[k],
            )
            total_loss_k = weighted_total_loss(components)

            # 所有激励损失取平均
            scaled_loss = total_loss_k / n_impacts
            scaled_loss.backward()

            add_detached_losses(
                accumulator,
                total_loss_k,
                components,
                scale=1.0 / n_impacts,
            )
            impact_data_loss_values.append(
                float(components["data"].detach().item())
            )

        optimizer_adam.step()

        for k, value in enumerate(impact_data_loss_values, start=1):
            accumulator[f"loss_data_impact_{k}"] = value

        return append_parameter_info(accumulator)

    # 正式训练轮数。如需先检查程序，可临时改小。
    n_epochs_adam = 60000

    for epoch in range(1, n_epochs_adam + 1):
        info = train_step_adam()
        info["epoch"] = epoch
        info["phase"] = "adam"
        history.append(info)

        if epoch % 1000 == 0:
            print(
                f"[Adam] Epoch {epoch:6d} | "
                f"loss={info['loss']:.3e}, "
                f"pde={info['loss_pde']:.3e}, "
                f"data={info['loss_data']:.3e}, "
                f"bc_top={info['loss_bc_top']:.3e}, "
                f"if_u={info['loss_if_u']:.3e}, "
                f"if_N={info['loss_if_N']:.3e}, "
                f"ic={info['loss_ic']:.3e}, "
                f"alpha={info['alpha_hat']:.4f}, "
                f"xi1={info['xi1_hat']:.4f}, "
                f"xi2={info['xi2_hat']:.4f}"
            )

        # Adam 训练若干轮后，根据当前网络残差重新选择三类关键点。
        # 更新之后的点在下一个更新周期内保持固定，避免每轮随机换点。
        if (
            epoch >= adaptive_warmup_epochs
            and epoch % adaptive_update_interval == 0
            and epoch < n_epochs_adam
        ):
            refresh_adaptive_points(
                adam_fixed_points,
                (
                    N_f1_adam,
                    N_f2_adam,
                    N_f3_adam,
                ),
                N_bc_adam,
                N_if_adam,
                phase_label=f"Adam epoch {epoch}",
            )

    # ========================================================
    # 3.11 L-BFGS 阶段采样
    # ========================================================
    Nf_fix = 5000
    Nbc_fix = 2000
    Nif_fix = 2000
    Nic_fix = 2000

    Nf1_fix, Nf2_fix, Nf3_fix = split_count_three(Nf_fix)
    Nic1_fix, Nic2_fix, Nic3_fix = split_count_three(Nic_fix)

    with torch.no_grad():
        lbfgs_fixed_points = {
            "s_f1": torch.rand(
                Nf1_fix, 1, device=device
            ),
            "s_f2": torch.rand(
                Nf2_fix, 1, device=device
            ),
            "s_f3": torch.rand(
                Nf3_fix, 1, device=device
            ),
            "tau_f1": torch.rand(
                Nf1_fix, 1, device=device
            ) * tau_max,
            "tau_f2": torch.rand(
                Nf2_fix, 1, device=device
            ) * tau_max,
            "tau_f3": torch.rand(
                Nf3_fix, 1, device=device
            ) * tau_max,
            "tau_bottom": torch.rand(
                Nbc_fix, 1, device=device
            ) * tau_max,
            "tau_top_list": [
                make_stratified_top_times(
                    impact_data,
                    Nbc_fix,
                )
                for impact_data in impact_data_list
            ],
            "tau_interface": torch.rand(
                Nif_fix, 1, device=device
            ) * tau_max,
            "s_ic1": torch.rand(
                Nic1_fix, 1, device=device
            ),
            "s_ic2": torch.rand(
                Nic2_fix, 1, device=device
            ),
            "s_ic3": torch.rand(
                Nic3_fix, 1, device=device
            ),
        }

    # 进入 L-BFGS 前，依据 Adam 结束时的网络状态做最后一次自适应选点。
    # 此后整个 L-BFGS 阶段固定这些点，满足确定性闭包的要求。
    refresh_adaptive_points(
        lbfgs_fixed_points,
        (Nf1_fix, Nf2_fix, Nf3_fix),
        Nbc_fix,
        Nif_fix,
        phase_label="before L-BFGS",
    )

    def compute_aggregate_fixed(backward=False):
        accumulator = empty_loss_accumulator()
        impact_data_loss_values = []
        total_value = torch.zeros((), device=device)

        for k in range(n_impacts):
            components = compute_single_impact_losses(
                conditional_nets,
                impact_data_list[k],
                lbfgs_fixed_points,
                data_indices=None,  # L-BFGS 使用全部桩顶时程
            )
            total_loss_k = weighted_total_loss(components)
            scaled_loss = total_loss_k / n_impacts

            if backward:
                scaled_loss.backward()

            total_value = (
                total_value
                + scaled_loss.detach()
            )

            add_detached_losses(
                accumulator,
                total_loss_k,
                components,
                scale=1.0 / n_impacts,
            )
            impact_data_loss_values.append(
                float(components["data"].detach().item())
            )

        for k, value in enumerate(impact_data_loss_values, start=1):
            accumulator[f"loss_data_impact_{k}"] = value

        accumulator = append_parameter_info(accumulator)
        return total_value, accumulator

    optimizer_lbfgs = optim.LBFGS(
        params,
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # 正式 L-BFGS 精修步数。如需先检查程序，可临时改小。
    n_lbfgs_steps = 2000
    start_epoch = n_epochs_adam
    info_holder = {}

    for step in range(1, n_lbfgs_steps + 1):

        def closure():
            optimizer_lbfgs.zero_grad()

            loss_value, closure_info = compute_aggregate_fixed(
                backward=True
            )
            info_holder["info"] = closure_info

            # LBFGS 已从参数中取得梯度；返回当前标量损失即可
            return loss_value

        optimizer_lbfgs.step(closure)

        # step 完成后重新评估当前参数状态
        _, info = compute_aggregate_fixed(backward=False)

        epoch = start_epoch + step
        info["epoch"] = epoch
        info["phase"] = "lbfgs"
        history.append(info)

        if step % 10 == 0:
            print(
                f"[L-BFGS] Step {step:5d} "
                f"(epoch~{epoch}) | "
                f"loss={info['loss']:.3e}, "
                f"pde={info['loss_pde']:.3e}, "
                f"data={info['loss_data']:.3e}, "
                f"bc_top={info['loss_bc_top']:.3e}, "
                f"if_u={info['loss_if_u']:.3e}, "
                f"if_N={info['loss_if_N']:.3e}, "
                f"ic={info['loss_ic']:.3e}, "
                f"alpha={info['alpha_hat']:.4f}, "
                f"xi1={info['xi1_hat']:.4f}, "
                f"xi2={info['xi2_hat']:.4f}"
            )

    # ========================================================
    # 4. 结果与可视化
    # ========================================================
    last = history[-1]

    alpha_est = last["alpha_hat"]
    xi1_est = last["xi1_hat"]
    xi2_est = last["xi2_hat"]
    xi_c_est = last["xi_c_hat"]
    lam_est = last["lam_hat"]

    print("\n===== Inversion result =====")
    print(
        f"True alpha={alpha_true:.3f}, "
        f"Estimated alpha={alpha_est:.4f}"
    )
    print(
        f"True xi1={xi1_true:.3f}, "
        f"Estimated xi1={xi1_est:.4f}"
    )
    print(
        f"True xi2={xi2_true:.3f}, "
        f"Estimated xi2={xi2_est:.4f}"
    )
    print(
        f"True xi_c={xi_c_true:.3f}, "
        f"Estimated xi_c={xi_c_est:.4f}"
    )
    print(
        f"True lam={lam_true:.3f}, "
        f"Estimated lam={lam_est:.4f}"
    )
    print(
        f"True z1,z2=({z1_true:.2f},{z2_true:.2f}) m"
    )
    print(
        f"Estimated z1,z2="
        f"({xi1_est*L:.2f},{xi2_est*L:.2f}) m"
    )

    epochs = np.asarray(
        [item["epoch"] for item in history],
        dtype=int,
    )

    def history_array(key):
        return np.asarray(
            [item[key] for item in history],
            dtype=float,
        )

    total_losses = history_array("loss")
    pde_losses = history_array("loss_pde")
    data_losses = history_array("loss_data")
    bc_top_losses = history_array("loss_bc_top")
    bc_bottom_losses = history_array("loss_bc_bottom")
    if_u_losses = history_array("loss_if_u")
    if_N_losses = history_array("loss_if_N")
    ic_losses = history_array("loss_ic")

    eps_plot = 1e-16
    total_losses = np.maximum(total_losses, eps_plot)
    pde_losses = np.maximum(pde_losses, eps_plot)
    data_losses = np.maximum(data_losses, eps_plot)
    bc_top_losses = np.maximum(bc_top_losses, eps_plot)
    bc_bottom_losses = np.maximum(
        bc_bottom_losses,
        eps_plot,
    )
    if_u_losses = np.maximum(if_u_losses, eps_plot)
    if_N_losses = np.maximum(if_N_losses, eps_plot)
    ic_losses = np.maximum(ic_losses, eps_plot)

    # --------------------------------------------------------
    # 图1：未加权损失分量
    # --------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.semilogy(
        epochs,
        total_losses,
        label="total",
        linewidth=1.8,
    )
    plt.semilogy(epochs, pde_losses, label="pde")
    plt.semilogy(epochs, data_losses, label="data")
    plt.semilogy(epochs, bc_top_losses, label="bc_top")
    plt.semilogy(
        epochs,
        bc_bottom_losses,
        label="bc_bottom",
    )
    plt.semilogy(
        epochs,
        if_u_losses,
        label="interface_u",
    )
    plt.semilogy(
        epochs,
        if_N_losses,
        label="interface_N",
    )
    plt.semilogy(epochs, ic_losses, label="initial")
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training loss components "
        "(conditional network, mean over impacts)"
    )
    plt.legend(ncol=2)
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图2：加权损失分量
    # --------------------------------------------------------
    weighted_pde = w_pde * pde_losses
    weighted_data = w_data * data_losses
    weighted_bc_top = w_bc_top * bc_top_losses
    weighted_bc_bottom = (
        w_bc_bottom * bc_bottom_losses
    )
    weighted_if_u = w_if_u * if_u_losses
    weighted_if_N = w_if_N * if_N_losses
    weighted_ic = w_ic * ic_losses

    plt.figure(figsize=(12, 5))
    plt.semilogy(
        epochs,
        total_losses,
        label="total",
        linewidth=1.8,
    )
    plt.semilogy(
        epochs,
        weighted_pde,
        label="weighted pde",
    )
    plt.semilogy(
        epochs,
        weighted_data,
        label="weighted data",
    )
    plt.semilogy(
        epochs,
        weighted_bc_top,
        label="weighted bc_top",
    )
    plt.semilogy(
        epochs,
        weighted_bc_bottom,
        label="weighted bc_bottom",
    )
    plt.semilogy(
        epochs,
        weighted_if_u,
        label="weighted interface_u",
    )
    plt.semilogy(
        epochs,
        weighted_if_N,
        label="weighted interface_N",
    )
    plt.semilogy(
        epochs,
        weighted_ic,
        label="weighted initial",
    )
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Weighted loss")
    plt.title(
        "Weighted training loss components "
        "(conditional network, mean over impacts)"
    )
    plt.legend(ncol=2)
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图3：每次激励的数据损失
    # --------------------------------------------------------
    plt.figure(figsize=(10, 4))
    for k in range(1, n_impacts + 1):
        key = f"loss_data_impact_{k}"
        impact_loss = np.maximum(
            history_array(key),
            eps_plot,
        )
        plt.semilogy(
            epochs,
            impact_loss,
            label=impact_data_list[k-1]["name"],
        )

    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Pile-head data loss")
    plt.title("Data losses under different impacts")
    plt.legend()
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图4：移动平均总损失
    # --------------------------------------------------------
    ma_window = 500
    total_ma = moving_average(total_losses, ma_window)

    if len(total_losses) >= ma_window:
        epochs_ma = epochs[ma_window - 1:]
    else:
        epochs_ma = epochs

    plt.figure(figsize=(12, 5))
    plt.semilogy(
        epochs,
        total_losses,
        linewidth=0.6,
        alpha=0.40,
        label="Raw total loss",
    )
    plt.semilogy(
        epochs_ma,
        total_ma,
        linewidth=2.0,
        label=f"Moving average, window={ma_window}",
    )
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Raw and moving-average total loss")
    plt.legend()
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图5：反演参数演化
    # --------------------------------------------------------
    alpha_history = history_array("alpha_hat")
    xi1_history = history_array("xi1_hat")
    xi2_history = history_array("xi2_hat")

    plt.figure(figsize=(8, 3))
    plt.plot(
        epochs,
        alpha_history,
        label="alpha_hat",
        linewidth=1.2,
    )
    plt.axhline(
        alpha_true,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label="true alpha",
    )
    plt.xlabel("Epoch")
    plt.ylabel("alpha_hat")
    plt.title("Evolution of alpha_hat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 3))
    plt.plot(
        epochs,
        xi1_history,
        label="xi1_hat",
        linewidth=1.2,
    )
    plt.plot(
        epochs,
        xi2_history,
        label="xi2_hat",
        linewidth=1.2,
    )
    plt.axhline(
        xi1_true,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label="xi1_true",
    )
    plt.axhline(
        xi2_true,
        color="k",
        linestyle=":",
        linewidth=1.0,
        label="xi2_true",
    )
    plt.xlabel("Epoch")
    plt.ylabel("xi")
    plt.title("Evolution of xi1_hat and xi2_hat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------------
    # 图6：多次锤击的桩顶速度响应
    # --------------------------------------------------------
    for net in conditional_nets:
        net.eval()

    t_ms = t_np * 1000.0
    velocity_predictions = []
    net1 = conditional_nets[0]

    for k in range(n_impacts):
        tau_eval = (
            impact_data_list[k]["tau"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_eval = torch.zeros_like(tau_eval)
        tp_eval = torch.full_like(
            tau_eval,
            fill_value=impact_data_list[k]["Tp_cond"],
        )

        w_top = net1(xi_top_eval, tau_eval, tp_eval)
        w_top_tau = autograd.grad(
            w_top,
            tau_eval,
            grad_outputs=torch.ones_like(w_top),
            create_graph=False,
        )[0]

        v_prediction = (
            w_top_tau
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            * v_scale
        )
        velocity_predictions.append(v_prediction)

    # 组合图：多次桩顶响应
    fig, axes = plt.subplots(
        n_impacts,
        1,
        figsize=(9, 3.0 * n_impacts),
        sharex=True,
    )

    if n_impacts == 1:
        axes = [axes]

    for k, axis in enumerate(axes):
        forward_velocity = impact_data_list[k]["v_top_np"]
        predicted_velocity = velocity_predictions[k]

        axis.plot(
            t_ms,
            forward_velocity,
            label="Forward",
            linewidth=1.5,
        )
        axis.plot(
            t_ms,
            predicted_velocity,
            "--",
            label="PINN",
            linewidth=1.3,
        )
        axis.set_ylabel(r"$v_{\mathrm{top}}$ (m/s)")
        axis.set_title(
            f"{impact_data_list[k]['name']}: "
            f"Tp={impact_data_list[k]['Tp']*1e3:.1f} ms"
        )
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        "Pile-head velocity responses under different impacts"
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(
        "multi_impact_pile_head_velocity.png",
        dpi=600,
        bbox_inches="tight",
    )

    # 每次激励单独输出带局部放大窗口的图
    x1_ms = 2.5
    x2_ms = 4.0

    for k in range(n_impacts):
        forward_velocity = impact_data_list[k]["v_top_np"]
        predicted_velocity = velocity_predictions[k]

        fig, axis = plt.subplots(figsize=(8.5, 3.4))

        axis.plot(
            t_ms,
            forward_velocity,
            label="Forward (z=0 m)",
            linewidth=1.6,
        )
        axis.plot(
            t_ms,
            predicted_velocity,
            "--",
            label="PINN (z=0 m)",
            linewidth=1.4,
        )

        axis.set_xlabel("Time (ms)")
        axis.set_ylabel(r"$v_{\mathrm{top}}$ (m/s)")
        axis.set_title(
            f"{impact_data_list[k]['name']}: "
            "pile-head velocity"
        )
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(loc="upper left")

        mask = (t_ms >= x1_ms) & (t_ms <= x2_ms)

        # 只有当时间范围内确实存在数据点时才绘制局部放大图
        if np.any(mask):
            # 手动定位到中上部，避免遮挡后期大反射峰
            inset_axis = axis.inset_axes([
                0.42,
                0.58,
                0.34,
                0.34,
            ])

            inset_axis.plot(
                t_ms,
                forward_velocity,
                linewidth=1.2,
            )
            inset_axis.plot(
                t_ms,
                predicted_velocity,
                "--",
                linewidth=1.1,
            )
            inset_axis.set_xlim(x1_ms, x2_ms)

            y_min = min(
                forward_velocity[mask].min(),
                predicted_velocity[mask].min(),
            )
            y_max = max(
                forward_velocity[mask].max(),
                predicted_velocity[mask].max(),
            )
            delta_y = max(y_max - y_min, 1e-12)

            inset_axis.set_ylim(
                y_min - 0.18 * delta_y,
                y_max + 0.18 * delta_y,
            )
            inset_axis.grid(
                True,
                linestyle="--",
                alpha=0.35,
            )
            inset_axis.tick_params(labelsize=8)

            mark_inset(
                axis,
                inset_axis,
                loc1=1,
                loc2=3,
                fc="none",
                ec="0.35",
                linewidth=0.8,
            )

        fig.tight_layout()
        fig.savefig(
            f"pile_head_velocity_impact_{k+1}_with_inset.png",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


if __name__ == "__main__":
    main()
