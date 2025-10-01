from .abstract_loss_funciton import AbstractLossFunction
from ..utils.openfold_violations.violations import find_structural_violations, get_atom14_positions


class ViolationLossFunction(AbstractLossFunction):
    def __init__(self, atom_array):
        self.atom_array = atom_array
        self._last_loss = None

    def get_violations_loss(self, x_0_hat):
        x_0_hat_14atoms = get_atom14_positions(self.atom_array, x_0_hat)
        violations_dict = find_structural_violations(x_0_hat_14atoms, self.atom_array)
        violation_loss = violations_dict["between_residues"]["clashes_mean_loss"] + violations_dict["between_residues"]["connections_per_residue_loss_sum"].mean() + violations_dict["within_residues"]["per_atom_loss_sum"].mean()
        return violation_loss
    
    def __call__(self, x_0_hat, time):
        loss = self.get_violations_loss(x_0_hat)
        self._last_loss = loss.item()
        return loss, None
    
    def wandb_log(self, x_0_hat):
        return {"violation loss": self._last_loss}