from .abstract_loss_funciton import AbstractLossFunction
from typing import List, Optional
import torch.nn.functional as F

class MultiLossFunction(AbstractLossFunction):
    def __init__(self, loss_functions: List[AbstractLossFunction], weights: Optional[List[float]] = None, normalize_losses_by_main_loss: bool = False):
        self.loss_functions = loss_functions
        self.weights = weights
        self.normalize_losses_by_main_loss = normalize_losses_by_main_loss
    
    def wandb_log(self, x_0_hat):
        wandb_logs = [loss_function.wandb_log(x_0_hat) for loss_function in self.loss_functions]
        common_loss = sum(
            [loss_function.wandb_log(x_0_hat)["loss"] * weight 
            for loss_function, weight in zip(self.loss_functions,self.weights) 
            if "loss" in loss_function.wandb_log(x_0_hat) and loss_function.wandb_log(x_0_hat).get("loss") is not None]
        )
        wandb_log = {}
        for log in wandb_logs:
            wandb_log.update(log)
        wandb_log["common_loss"] = common_loss
        return wandb_log
    
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        loss_function_values = [loss_function(x_0_hat, time, structures=structures, i=i, step=step) for loss_function in self.loss_functions]
        losses = [v[0] for v in loss_function_values]
        # the first weight is not taken into account. only the rest of the weights since the first is the main loss...!

        main_loss = losses[0]
        loss_sum = main_loss
        if len(losses) > 1:
            if self.normalize_losses_by_main_loss: 
                loss_sum = loss_sum + sum([
                    l / l.clone().detach().abs() * main_loss.clone().detach().abs() * w 
                    for l, w in zip(losses[1:], self.weights[1:]) # Normalizing the losses by bringing them first to the magnitude of the main loss and then scaling by the weight
                ])
            else:
                loss_sum = loss_sum + sum([
                    l * w 
                    for l, w in zip(losses[1:], self.weights[1:]) # Standard weighted sum of the losses
                ])


        new_x_0_hats = [loss_values[1] for loss_values in loss_function_values]
        new_x_0_hats = [item for item in new_x_0_hats if item is not None]
        assert len(new_x_0_hats) <= 1
        return loss_sum, losses, new_x_0_hats[0] if len(new_x_0_hats) > 0 else None
    
    def post_optimization_step(self):
        for loss_function in self.loss_functions:
            loss_function.post_optimization_step()
    
    def save_state(self, structures, folder_path, **kwargs):
        for loss_function in self.loss_functions:
            loss_function.save_state(structures, folder_path, **kwargs)