from .abstract_loss_funciton import AbstractLossFunction
from typing import List, Optional
import torch.nn.functional as F

class MultiLossFunction(AbstractLossFunction):
    def __init__(self, loss_functions: List[AbstractLossFunction], weights: Optional[List[float]] = None):
        self.loss_functions = loss_functions
        self.weights = weights
    
    def wandb_log(self, x_0_hat):
        wandb_logs = [loss_function.wandb_log(x_0_hat) for loss_function in self.loss_functions]
        wandb_log = {}
        for log in wandb_logs:
            wandb_log.update(log)
        return wandb_log
    
    def __call__(self, x_0_hat, time, structures=None, i=None):
        # TODO: possibly make the x_0_hat go from one loss function to the other, might be usefull
        loss_function_values = [loss_function(x_0_hat, time) for loss_function in self.loss_functions]
        loss_sum = sum([loss_values[0] * weight for loss_values, weight in zip(loss_function_values, self.weights)])
        new_x_0_hats = [loss_values[1] for loss_values in loss_function_values]
        new_x_0_hats = [item for item in new_x_0_hats if item is not None]
        assert len(new_x_0_hats) <= 1
        return loss_sum, new_x_0_hats[0] if len(new_x_0_hats) > 0 else None
    
    def post_optimization_step(self):
        for loss_function in self.loss_functions:
            loss_function.post_optimization_step()
    
    def save_state(self, structures, folder_path):
        for loss_function in self.loss_functions:
            loss_function.save_state(structures, folder_path)