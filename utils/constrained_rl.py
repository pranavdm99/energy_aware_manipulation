
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from utils.constants import LAGRANGIAN_CLIP_MAX

class LagrangianRegistry:
    """Shared registry for Lagrangian multipliers."""
    weights = {} # descriptor -> alpha

    @classmethod
    def get_weight(cls, descriptor, default=0.0):
        return cls.weights.get(descriptor, default)

    @classmethod
    def set_weight(cls, descriptor, value):
        cls.weights[descriptor] = value

class LagrangianCallback(BaseCallback):
    """
    A callback for Energy-Constrained Optimization (ECO).
    Dynamically adjusts the energy penalty (Lagrange multiplier lambda)
    per descriptor to keep average energy consumption below target budgets.
    """
    def __init__(
        self, 
        learning_rate: float = 0.05,
        initial_lambda: float = 0.01,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.learning_rate = learning_rate
        self.initial_lambda = initial_lambda
        
        # We store lambda parameters per descriptor
        self.lambdas = {}  # descriptor -> log_lambda Parameter
        self.optimizers = {} # descriptor -> Optimizer
        self.history = {} # descriptor -> list of energies

    def _get_lagrangian_params(self, descriptor):
        if descriptor not in self.lambdas:
            # log-space for positivity
            log_lambda = torch.nn.Parameter(torch.tensor(np.log(self.initial_lambda + 1e-6)))
            self.lambdas[descriptor] = log_lambda
            self.optimizers[descriptor] = torch.optim.Adam([log_lambda], lr=self.learning_rate)
            self.history[descriptor] = []
            LagrangianRegistry.set_weight(descriptor, self.initial_lambda)
        return self.lambdas[descriptor], self.optimizers[descriptor]

    def _on_step(self) -> bool:
        # 1. Collect results from finished episodes
        for info in self.locals.get("infos", []):
            if "energy" in info and "episode_summary" in info["energy"]:
                summary = info["energy"]["episode_summary"]
                energy = summary["total_energy"]
                
                lang_info = info.get("language", {})
                descriptor = lang_info.get("descriptor", "default")
                
                _, _ = self._get_lagrangian_params(descriptor) 
                self.history[descriptor].append(energy)
                
                if len(self.history[descriptor]) > 100:
                    self.history[descriptor].pop(0)

        # 2. Periodically update lambdas
        if self.n_calls % 100 == 0:
            for descriptor, energies in self.history.items():
                if len(energies) < 3: 
                    continue
                    
                # Find budget in current infos (fallback to normally budget)
                budget = 150.0
                for info in self.locals.get("infos", []):
                    if info.get("language", {}).get("descriptor") == descriptor:
                        budget = info["language"].get("energy_budget", 150.0)
                        break
                
                avg_energy = np.mean(energies)
                violation = avg_energy - budget
                
                log_lambda, optimizer = self._get_lagrangian_params(descriptor)
                
                if log_lambda.grad is not None:
                    log_lambda.grad.zero_()
                
                # Dual objective: maximize lambda * (avg_energy - budget)
                # Loss = -(lambda * violation)
                current_lambda_tensor = torch.exp(log_lambda)
                loss = -(current_lambda_tensor * violation)
                loss.backward()
                optimizer.step()

                # --- ECO Stability Fix: Lagrangian Clipping ---
                # Prevents alpha/lambda from exploding to astronomical values (e.g. 1e153)
                # when the budget is difficult to satisfy.
                with torch.no_grad():
                    max_log_lambda = np.log(LAGRANGIAN_CLIP_MAX)  # Bounded by constant
                    log_lambda.clamp_(max=max_log_lambda)
                
                updated_lambda = torch.exp(log_lambda).item()
                LagrangianRegistry.set_weight(descriptor, updated_lambda)
                
                # Log to WandB
                prefix = f"eco/{descriptor}"
                self.logger.record(f"{prefix}/lambda", updated_lambda)
                self.logger.record(f"{prefix}/avg_energy", avg_energy)
                self.logger.record(f"{prefix}/budget", budget)
                self.logger.record(f"{prefix}/violation", violation)

        return True
