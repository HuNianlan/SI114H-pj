import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("SimpleAdam does not support sparse gradients")

                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # 权重衰减
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # 更新一阶与二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                p.data = p.data.addcdiv(exp_avg, denom, value=-step_size)

        return loss


class LineSearchSolver:
    def __init__(self, energy_fn, constraint_fn=None, alpha_init=1.0, c=1e-4, tau=0.5):
        """
        energy_fn: a callable that takes (Verts, Faces) and returns energy
        constraint_fn: optional constraint penalty, also a callable
        alpha_init: initial step size
        c: Armijo condition parameter
        tau: step size reduction factor (0 < tau < 1)
        """
        self.energy_fn = energy_fn
        self.constraint_fn = constraint_fn
        self.alpha_init = alpha_init
        self.c = c
        self.tau = tau

    def step(self, Verts: torch.Tensor, Faces: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_init
        E_current = self.energy_fn(Verts, Faces)
        descent_dir = -grad

        with torch.no_grad():
            while True:
                V_new = Verts + alpha * descent_dir
                E_new = self.energy_fn(V_new, Faces)

                # Armijo condition
                if E_new <= E_current + self.c * alpha * torch.sum(grad * descent_dir):
                    break

                alpha *= self.tau  # reduce step size

                if alpha < 1e-6:
                    break  # prevent infinite loop

            Verts += alpha * descent_dir  # update Verts
        return Verts
