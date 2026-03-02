from .abstract_loss_funciton import AbstractLossFunction
from typing import List, Optional
import torch.nn.functional as F



class MultiLossFunction(AbstractLossFunction):
    def __init__(
        self, 
        loss_functions: List[AbstractLossFunction], 
        weights: Optional[List[float]] = None, 
        normalize_losses_by_main_loss: Optional[bool] = False,
        loss_labels: Optional[List[str]] = None,
    ):
        self.loss_functions = loss_functions
        self.weights = weights
        self.normalize_losses_by_main_loss = normalize_losses_by_main_loss
    
    def wandb_log(self, x_0_hat):
        wandb_logs = [loss_function.wandb_log(x_0_hat) for loss_function in self.loss_functions]
        common_loss = sum(
            log["loss"] * weight
            for log, weight in zip(wandb_logs, self.weights)
            if "loss" in log and log.get("loss") is not None
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
                    l / (l.clone().detach().abs() + 1e-8) * main_loss.clone().detach().abs() * w 
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
    
    def pre_optimization_step(self, x_0_hat, i=None, step=None):
        # Either only the main loss function can change the x_0_hat or the only one that changed x_0_hat is being chosen.
        # Mainly used for purposes like concatenation of frozen atoms / inprinting etc.
        
        main_loss = self.loss_functions[0] # Assumes the first loss is the main loss
        new_x_0_hat = main_loss.pre_optimization_step(x_0_hat, i=i, step=step) # Normally just contains new concatenated parts etc.
        
        if len(self.loss_functions) > 1:
            for loss_function in self.loss_functions[1:]:
                loss_function.pre_optimization_step(new_x_0_hat, i=i, step=step) # Perform other pre-optimizations if there's any.
        
        return new_x_0_hat
    
    def post_optimization_step(self, x_0_hat):
        main_loss = self.loss_functions[0]
        new_x_0_hat = main_loss.post_optimization_step(x_0_hat) 
        # An important element is removing any concatenated bits. However, some other optimizations like b-factor tuning can be present. 

        if len(self.loss_functions) > 1:
            for loss_function in self.loss_functions[1:]:
                loss_function.post_optimization_step(new_x_0_hat) # Perform other post-optimizations if there's any.
        
        return new_x_0_hat
    
    def save_state(self, structures, folder_path, **kwargs):
        for loss_function in self.loss_functions:
            loss_function.save_state(structures, folder_path, **kwargs)
