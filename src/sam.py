from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional
import torch

class BaseSAM(torch.optim.Optimizer, metaclass=ABCMeta):
    def __init__(
        self, 
        params, 
        base_optimizer: torch.optim.Optimizer, 
        rho: float, 
        adaptive: bool = False,
        **kwargs
    ) -> None:
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.closure = None # [修复] 初始化属性，让外部可以挂载

    @abstractmethod
    def first_step(self, zero_grad: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def second_step(self, zero_grad: bool = False) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        # === 修复核心：优先使用传入的 closure，如果没有，检查是否有挂载的 closure ===
        if closure is None:
            if hasattr(self, "closure"):
                closure = self.closure
                self.closure = None # 用完即焚，防止污染下一次迭代
        
        # 如果依然没有 closure（比如 Trainer 调用 step 时），直接返回，避免报错
        if closure is None:
            return
        # ===================================================================

        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        # 获取第一个参数所在的设备，确保 norm 计算在正确的设备上
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

        if not torch.isfinite(norm):
            print(f"Warning: Grad norm is {norm}, skipping SAM step to prevent collapse.")
            return None # 返回 None 表示异常

        return norm

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAM(BaseSAM):
    def __init__(
        self, 
        params, 
        base_optimizer: torch.optim.Optimizer, 
        rho: float,
        adaptive: bool = False,
        **kwargs
    ) -> None:
        super().__init__(params, base_optimizer, rho, adaptive, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()

        # === 新增安全检查 ===
        if grad_norm is None: 
            return # 跳过这一步
        # ==================
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()