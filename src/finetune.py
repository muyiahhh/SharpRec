import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Union, Any
from sam import SAM

class SAFTTrainer(Trainer):
    def __init__(self, use_sam=False, rho=0.05, **kwargs):
        """
        Args:
            use_sam (bool): 是否使用 SAM 优化器。如果为 False，则执行普通 SFT。
            rho (float): SAM 的扰动半径。
        """
        super().__init__(**kwargs)
        self.use_sam = use_sam
        self.rho = rho

    def create_optimizer(self):
        """
        重写优化器初始化：
        - 如果 use_sam=False: 调用父类方法 (默认 AdamW 或 TrainingArguments 指定的优化器)。
        - 如果 use_sam=True: 初始化 SAM 优化器。
        """
        # === 分支 1: 不使用 SAM，回归标准 Trainer 逻辑 ===
        if not self.use_sam:
            return super().create_optimizer()

        # === 分支 2: 使用 SAM ===
        if self.optimizer is None:
            params_to_optimize = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            
            if not params_to_optimize:
                raise ValueError("No trainable parameters found. Please ensure LoRA is correctly applied.")

            print(f"SAM Optimizer initialized. Optimizing {len(params_to_optimize)} tensors. Rho={self.rho}")

            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }

            self.optimizer = SAM(
                params=params_to_optimize,
                base_optimizer=torch.optim.AdamW,
                rho=self.rho,
                **optimizer_kwargs
            )
            
        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        - 如果 use_sam=False: 调用父类标准 training_step (Forward -> Backward -> Step 自动处理)。
        - 如果 use_sam=True: 执行手动 Backward 并挂载 closure。
        """
        
        # === 分支 1: 不使用 SAM，回归标准 SFT 逻辑 ===
        if not self.use_sam:
            return super().training_step(model, inputs, num_items_in_batch)

        # === 分支 2: 使用 SAM 的特殊逻辑 ===
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 定义闭包
        def closure():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                
            self.accelerator.backward(loss)
            return loss

        # 1. 计算梯度 (Backward)
        loss = closure() 

        # 2. 如果到了梯度累积的这一步，挂载 closure 到优化器
        if self.accelerator.sync_gradients:
            opt = self.optimizer
            # 递归找到底层 SAM optimizer
            while hasattr(opt, "optimizer"):
                opt = opt.optimizer
            
            # 确保它是我们自定义的 SAM，才挂载 closure
            if hasattr(opt, "closure"):
                opt.closure = closure

        return loss.detach()