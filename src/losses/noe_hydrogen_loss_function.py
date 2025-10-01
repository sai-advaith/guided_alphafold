
from matplotlib import pyplot as plt
from tqdm import tqdm

from .s_2_loss_function import S2LossFunction

from ..protenix.metrics.rmsd import self_aligned_rmsd

from ..utils.io import load_pdb_atom_locations
from ..utils.hydrogen_addition import FragmentLibrary, AtomNameLibrary, get_hydrogen_names
from .abstract_loss_funciton import AbstractLossFunction
import wandb
import torch
import pandas as pd
import numpy as np

EPSILON=1e-5


def calculate_within_chain_clash(
    coordinates: torch.Tensor,
    threshold: float = 1.1
) -> torch.Tensor:
    # Get pairwise distances between all atoms
    distances = torch.cdist(coordinates, coordinates)
    # Pick only the upper triangular part of the distance matrix ignoring the diagonal
    distances = distances.triu(diagonal=1)
    # Slice out upper triangular part
    distances = distances[distances > 0]
    # Penalize heavily if the distance is below the threshold
    loss = (torch.relu(threshold - distances)/0.25).exp()
    num_violations = (distances < threshold).sum()
    return loss.mean(), num_violations

def methyl_group_names(nmr_name, residue_name):
    if residue_name == "LEU" and nmr_name == "MD1":
        return ["HD11", "HD12", "HD13"]
    if residue_name == "LEU" and nmr_name == "MD2":
        return ["HD21", "HD22", "HD23"]
    if residue_name == "ALA" and nmr_name == "MB":
        return ["HB1","HB2","HB3"]
    if residue_name == "MET" and nmr_name == "ME":
        return ["HE1", "HE2", "HE3"]
    if residue_name == "ILE" and nmr_name == "MG":
        return ["HG21","HG22","HG23"]
    if residue_name == "ILE" and nmr_name == "MD":
        return ["HD11","HD12","HD13"]
    if residue_name == "VAL" and nmr_name == "MG1":
        return ["HG11","HG12","HG13"]
    if residue_name == "VAL" and nmr_name == "MG2":
        return ["HG21","HG22","HG23"]
    if residue_name == "THR" and nmr_name == "MG":
        return ["HG21","HG22","HG23"]
    raise ValueError("Unknown methyl hydrogen to pdb conversion")

def q_group_names(nmr_name, residue_name):
    if residue_name == "TYR" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TYR" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "PHE" and nmr_name == "QD":
        return ["HD1","HD2"]
    if residue_name == "PHE" and nmr_name == "QE":
        return ["HE1", "HE2"]
    if residue_name == "TRP" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TRP" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "TRP" and nmr_name == "QZ":
        return ["HZ2"]
    if residue_name == "LYS" and nmr_name == "QZ":
        return ["HZ1","HZ2","HZ3"]
    raise ValueError("Unknown q hydrogen to pdb conversion")

def distance_to_intensity(val):
    return 1/(val**6+EPSILON)

def intensity_to_distance(val):
    return 1/(val**(1/6)+EPSILON)
   
   
class NOEHydrogenLossFunction(AbstractLossFunction):
    def __init__(self, restraint_file, pdb_file, atom_array=None, device="cpu", iid_loss=False,
                 methyl_relax_file=None, methyl_relax_scale=0.0, 
                 methyl_rdc_file=None, methyl_rdc_scale=0.0, 
                 amide_rdc_file=None, amide_rdc_scale=0.0, 
                 amide_relax_file=None, amide_relax_scale=0.0, 
                 noe_scale=1.0, op_n_bootstrap=10, average_intensity=False):
        self.nmr_data = pd.read_csv(restraint_file)
        self.nmr_data = self.nmr_data[self.nmr_data["type"]=="NOE"]
        self.pdb_id = pdb_file.split("/")[1]
       
        self.atom_array = atom_array
        
        self.iid_loss = iid_loss
        self.op_n_bootstrap = op_n_bootstrap
        self.noe_scale = noe_scale
        self.average_intensity = average_intensity
        
        self._initialize_log_values()

        self.fragment_library = FragmentLibrary.standard_library()
        self.name_library = AtomNameLibrary.standard_library()
        self.device=device
        self.reference_atom_locations = load_pdb_atom_locations(pdb_file).to(device)
        
        # gets indices of atoms that are going constrained to be between bounds
        self.hydrogen_guidance_params = self.get_hydrogen_guidance_params()
        
        # get bounds and or conditions 
        # If a lower bound is equal to '.' set it to 0
        self.nmr_data["lower_bound"] = self.nmr_data["lower_bound"].apply(lambda x: 0 if x == '.' else x)
        self.lower_bound = torch.tensor(self.nmr_data["lower_bound"], dtype=torch.float32, device=device)
        self.upper_bound = torch.tensor(self.nmr_data["upper_bound"], dtype=torch.float32, device=device)
        or_cond = torch.tensor(self.nmr_data["constrain_id"], dtype=torch.float32, device=device)
        self.unique_or, self.inverse_or_indices = torch.unique(or_cond, return_inverse=True)
        
        
        # initialize order parameter related things
        self.methyl_relax_scale = methyl_relax_scale
        self.methyl_rdc_scale = methyl_rdc_scale
        self.amide_rdc_scale = amide_rdc_scale
        self.amide_relax_scale = amide_relax_scale
        self.methyl_relax_loss = S2LossFunction(atom_array, methyl_relax_file, device, type="methyl_relax") if methyl_relax_file else None
        self.methyl_rdc_loss = S2LossFunction(atom_array, methyl_rdc_file, device, type="methyl_rdc") if methyl_rdc_file else None
        self.amide_rdc_loss = S2LossFunction(atom_array, amide_rdc_file, device, type="amide_relax") if amide_rdc_file else None
        self.amide_relax_loss = S2LossFunction(atom_array, amide_relax_file, device, type="amide_rdc") if amide_relax_file else None


    def _initialize_log_values(self):
        self.last_loss = None
        self.methyl_rdc_loss_val = None
        self.methyl_relax_loss_val = None
        self.amide_rdc_loss_val = None
        self.amide_relax_loss_val = None
        self.rmsd_loss = None
        self.lb_loss_val = None
        self.ub_loss_val = None
        self.constraints_satisfied_ub = None
        self.constraints_satisfied_lb = None
        self.num_violations = None
        self.within_chain_clash = None
        self.bootstrapped_methyl_rdc_loss_val = None
        self.bootstrapped_methyl_relax_loss_val = None
        self.bootstrapped_amide_rdc_loss_val = None
        self.bootstrapped_amide_relax_loss_val = None
        
    def get_hydrogen_guidance_params(self):
        guidance_params = []
        for _, row in tqdm(self.nmr_data.iterrows(), total=len(self.nmr_data), desc="generating comparison indices"):
            guidance_params.append((row["residue1_num"], row["residue1_id"], 
                                    row[f"atom1"], 
                                    row["residue2_num"], 
                                    row["residue2_id"], 
                                    row[f"atom2"]))
        return guidance_params
    
    
    def wandb_log(self, x_0_hat):
        _, aligned_structure, _, _ = self_aligned_rmsd(x_0_hat, self.reference_atom_locations, torch.ones_like(x_0_hat, dtype=torch.bool)[...,0])
        rmsd_loss = (aligned_structure - self.reference_atom_locations[None]).norm(dim=-1).mean()
        
        return ({"loss": self.last_loss,
                 "methyl_rdc_loss": self.methyl_rdc_loss_val,
                 "methyl_relax_loss": self.methyl_relax_loss_val,
                "amide_rdc_loss": self.amide_rdc_loss_val,
                "amide_relax_loss": self.amide_relax_loss_val,
                 "rmsd_loss": rmsd_loss,
                 "lb_loss": self.lb_loss_val,
                "ub_loss": self.ub_loss_val,
                "constraints_satisfied_ub": self.constraints_satisfied_ub,
                "constraints_satisfied_lb": self.constraints_satisfied_lb,
                "num_violations": self.num_violations,
                "within_chain_clash": self.within_chain_clash,
                "bootstrapped_methyl_rdc_loss": self.bootstrapped_methyl_rdc_loss_val,
                "bootstrapped_methyl_relax_loss": self.bootstrapped_methyl_relax_loss_val,
                "bootstrapped_amide_rdc_loss": self.bootstrapped_amide_rdc_loss_val,
                "bootstrapped_amide_relax_loss": self.bootstrapped_amide_relax_loss_val,
                 })
        
    def compute_group_coord(self, hydrogen_names_to_index, hydrogen_atoms_batch, cond, group_type):
        if group_type=="M":
            names = methyl_group_names(cond[-1], cond[1])
        if group_type=="Q":
            names = q_group_names(cond[-1], cond[1])
        indices = []
        for name in names:
            indices += [hydrogen_names_to_index[(cond[0], cond[1], name)]]
        coord = hydrogen_atoms_batch[:,indices]
        coord = coord.mean(dim=1)
        return coord
        
    def _get_atom_coord(self, cond, hydrogen_names_to_index, hydrogen_atoms_batch, x_0_hat):
        """Attempts to retrieve atom coordinates either from hydrogens or atom_array."""
        try:
            if "M" in cond[-1]:
                return self.compute_group_coord(hydrogen_names_to_index, hydrogen_atoms_batch, cond, "M")
            if "Q" in cond[-1]:
                return self.compute_group_coord(hydrogen_names_to_index, hydrogen_atoms_batch, cond, "Q")
            return hydrogen_atoms_batch[:, hydrogen_names_to_index[cond]]
        except KeyError:
            return self._resolve_atom_coord(cond, x_0_hat)
    
    def _resolve_atom_coord(self, cond, x_0_hat):
        """Fallback for non-hydrogen atoms based on residue and atom name."""
        atom_array_names = np.array([atom.atom_name for atom in self.atom_array])
        atom_array_res_id = np.array([atom.res_id for atom in self.atom_array])
        try:
            idx = np.where((atom_array_names == cond[2]) & (atom_array_res_id == cond[0]))[0][0]
            return x_0_hat[:, idx]
        except IndexError:
            print(f"hydrogen was not added {cond}")
            return None
        
    def get_hydrogen_coord(self, hydrogen_atoms_batch, hydrogen_names, x_0_hat, mask):
        atoms_to_compare_1, atoms_to_compare_2  = [], []
        hydrogen_names_to_index = {tuple(identifier): idx for idx, identifier in enumerate(hydrogen_names)}
        
        for i,param in enumerate(self.hydrogen_guidance_params):
            coords = []
            conds = [(param[0], param[1], param[2]), (param[3], param[4], param[5])]
            
            for cond in conds:
                coord = self._get_atom_coord(cond, hydrogen_names_to_index, hydrogen_atoms_batch, x_0_hat)
                if coord is None:
                    mask[i] = False
                    break
                coords.append(coord)
            if len(coords)==2:
                atoms_to_compare_1.append(coords[0])
                atoms_to_compare_2.append(coords[1])

            
        atoms_to_compare_1 = torch.stack(atoms_to_compare_1, dim=0).permute(1,0,2)
        atoms_to_compare_2 = torch.stack(atoms_to_compare_2, dim=0).permute(1,0,2)
        return atoms_to_compare_1, atoms_to_compare_2
    
    def integrate_or_conditions(self, curr_loss, mask):
        min_values = torch.zeros((curr_loss.shape[0], len(self.unique_or),),dtype=curr_loss.dtype, device=curr_loss.device)
        inverse_or_indices = self.inverse_or_indices[None].repeat(curr_loss.shape[0],1)
        dim=1
        min_values = torch.scatter_reduce(min_values, dim, inverse_or_indices[...,mask], curr_loss, reduce="amin", include_self=False)
        
        return min_values

    def _compute_loss_bounds(self, dist, mask):
        loss_ub = torch.relu(dist - self.upper_bound[None][:,mask])
        loss_lb = torch.relu(self.lower_bound[None][:, mask] - dist)
        # take or conditions into account
        loss_ub_or = self.integrate_or_conditions(loss_ub, mask)
        loss_lb_or = self.integrate_or_conditions(loss_lb, mask)
        
        loss_ub = loss_ub_or.mean()
        loss_lb = loss_lb_or.mean()
        
        return loss_ub, loss_lb, loss_ub_or, loss_lb_or
    
    def _calculate_bootstrapped_loss(self, s2_func, x_0_hat,time, hydrogen_atoms_batch, hydrogen_names):
        losses = []
        order_params = []
        N = x_0_hat.shape[0]
        for _ in range(self.op_n_bootstrap):
            idx = torch.randint(0, N, (N//2,))
            x_0_hat_sample = x_0_hat[idx]
            hydrogen_atoms_batch_sample = hydrogen_atoms_batch[idx]
            s2_val, pred_order_param = s2_func(x_0_hat_sample, time, hydrogen_atoms_batch_sample, hydrogen_names)
            order_params.append(pred_order_param)
            losses += [s2_val]
        losses = torch.stack(losses)
        return losses, order_params

    def _intensity_normalized_distance(self, dist):
        dist = distance_to_intensity(dist)
        dist = dist.mean(dim=0)
        dist = intensity_to_distance(dist)
        return dist

    def __call__(self, x_0_hat, time):
        x_0_hat = x_0_hat.to(self.device)
        
        # sometime we have constraints for hydrogens that were not added
        mask = torch.ones_like(self.upper_bound, dtype=torch.bool)
        
        # add hydrogens
        hydrogen_atoms_batch, naming_sample = self.fragment_library.calculate_hydrogen_coord_batch(x_0_hat, self.atom_array.bonds, 
                                                                                                   self.atom_array.atom_name, self.atom_array.element, 
                                                                                                   self.atom_array.res_name, self.device)
        hydrogen_names = get_hydrogen_names(self.atom_array, naming_sample, self.name_library)
        atoms_to_compare_1, atoms_to_compare_2 = self.get_hydrogen_coord(hydrogen_atoms_batch, hydrogen_names, x_0_hat, mask)
        
        model_distances = (atoms_to_compare_1 - atoms_to_compare_2).norm(dim=-1)
        if self.average_intensity:
            model_distances = self._intensity_normalized_distance(model_distances)
        else:
            model_distances = model_distances if self.iid_loss else model_distances.mean(dim=0)[None]
        
        self.ub_loss_val, self.lb_loss_val, loss_ub_or, loss_lb_or = self._compute_loss_bounds(model_distances, mask)

        loss = self.ub_loss_val
        loss = self.noe_scale * loss
        
        # guide with order parameters
        for s2_type in ["methyl_relax", "methyl_rdc", "amide_relax", "amide_rdc"]:
            s2_func = getattr(self, f"{s2_type}_loss")
            if not s2_func:
                continue
            s_2_loss_val, pred_order_params = s2_func(x_0_hat, time, hydrogen_atoms_batch, hydrogen_names)
            setattr(self, f"{s2_type}_loss_val", s_2_loss_val.item())
            
            s_2_scale = getattr(self, f"{s2_type}_scale", 0)
            loss = loss + s_2_loss_val * s_2_scale
                
            fig = s2_func.plot_s2_values(pred_order_params, s_2_loss_val)
            # Log the image to wandb
            wandb.log({f"{s2_type}": wandb.Image(fig)})
            plt.close(fig)
            
            # bootstrap s2
            with torch.no_grad():
                if self.op_n_bootstrap > 0:
                    bootstrapped_s_2_loss, boot_order_params = self._calculate_bootstrapped_loss(s2_func, x_0_hat,time, hydrogen_atoms_batch, hydrogen_names)
                    setattr(self, f"bootstrapped_{s2_type}_loss_val", torch.mean(bootstrapped_s_2_loss).item())
                    fig = s2_func.plot_s2_values(boot_order_params[0], bootstrapped_s_2_loss[0])
                    wandb.log({f"{s2_type}_boot": wandb.Image(fig)})
                    plt.close(fig)
        
        # logs
        self.constraints_satisfied_ub = ((loss_ub_or==0).sum().item() / len(self.unique_or)) 
        self.constraints_satisfied_lb = ((loss_lb_or==0).sum().item() / len(self.unique_or)) 
        within_chain_clash, num_violations = calculate_within_chain_clash(x_0_hat, 1.2)
        self.num_violations = num_violations
        self.within_chain_clash = within_chain_clash.item()
        self.last_loss = loss
        
        return loss, None
    
    
    
    
    
# for testing only !!        
            # for sample in x_0_hat:
            #     hydrogen_atoms_lst += [self.generate_hydrogen_atoms(sample, self.fragment_library.calculate_hydrogen_coord)]
                # self.atom_array.coord = x_0_hat[0].detach().cpu().numpy()
                # hydride_fragment_library = hydride.FragmentLibrary.standard_library()
                # test_atom_array = hydride_fragment_library.calculate_hydrogen_coord(self.atom_array)
            
            # self.atom_array.coord = x_0_hat[0].detach().cpu().numpy()
            # test_atom_array = hydride.add_hydrogen(self.atom_array)
        
        # update atom array with hydrogens
        # updated_atom_array, mask_non_hydrogen_atoms = update_hydrogens_in_atom_array(self.atom_array, naming_sample, hydrogen_names)
        # # atom_stack = batch_hydrogen_to_atom_stack_array(updated_atom_array, hydrogen_atoms_batch, mask_non_hydrogen_atoms)
        # save_hydrogen_array(updated_atom_array, path="with_hyd")
        
        # save_hydrogen_array(self.atom_array, path="no_hyd")
        
        # hydride_fragment_library = hydride.FragmentLibrary.standard_library()
        # test_atom_array = [torch.tensor(a) for a in hydride_fragment_library.calculate_hydrogen_coord(self.atom_array)]
        # hydride_updated_atom_array, mask_non_hydrogen_atoms = update_hydrogens_in_atom_array(self.atom_array, test_atom_array, hydrogen_names)
        # save_hydrogen_array(hydride_updated_atom_array, path="hydride_hydrogens")
