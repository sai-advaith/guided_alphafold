from src.metrics.density_metrics_processor import DensityMetricsProcessor
import torch
import os
from src.utils.non_diffusion_model_manager import ProteinxModelManager
import argparse
from src.utils.io import load_config, namespace_to_dict, query_msa_server
import wandb
from src.utils.io import load_pdb_atom_locations, get_sampler_pdb_inputs, delete_hydrogens, write_back_pdb_coordinates
from src.losses import * 
from tqdm import tqdm
import json
import numpy as np

import gemmi
from src.protenix.metrics.rmsd import self_aligned_rmsd
from src.utils.pdb_parsing import find_bonded_pairs
from src.utils.relaxation import relax_pdb

parser = argparse.ArgumentParser()

parser.add_argument('--configuration_file', type=str, required=False, help="the path to the configuration file", default="generated_configurations/2izrA_208_212_end_guided.yaml")
parser.add_argument('--device', type=str, required=False, default="cuda:0")

class ExperimentManager:
    def __init__(self, config, device):
        # general
        self.device = device
        self.config = config
        self.name = f"{self.config.general.name}_{self.config.protein.pdb_id}"
        self.query_msa_server()
        self.model_manager = self._get_model_manager()
        self._setup_wandb()
        self.loss_function = self._get_loss_function()

        self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.general.name)
        os.makedirs(self.experiment_save_dir, exist_ok=True)

        torch.manual_seed(self.config.general.seed)

    def _setup_wandb(self):
        if self.config.wandb.login_key is not None:
            wandb.login(key=self.config.wandb.login_key)
            wandb.init(project=self.config.wandb.project ,name=self.name, mode=self.config.wandb.mode, config=namespace_to_dict(self.config))

    def _get_model_manager(self):
        return ProteinxModelManager(
            self.config.protein.sequence,
            self.config.protein.pdb_id,
            self.config.protein.reference_pdb,
            self.config.model_manager.N_cycle,
            self.config.model_manager.chunk_size,
            self.config.model_manager.diffusion_N,
            self.config.model_manager.gamma0,
            self.config.model_manager.gamma_min,
            self.config.model_manager.noise_scale_lambda,
            self.config.model_manager.step_scale_eta,
            "fp32",
            False,
            self.config.model_manager.msa_save_dir,
            self.config.model_manager.msa_embedding_cache_dir,
            self.config.model_manager.model_checkpoint_path,
            self.config.model_manager.dump_dir,
            self.config.model_manager.use_msa,
            self.config.general.batch_size,
            device=self.device
            )
    
    def _get_loss_function(self):
        if self.config.loss_function.loss_function_type == "rmsd":
            rmsd_config = self.config.loss_function.rmsd_loss_function
            loss_function = MultiRMSDLossFunction(rmsd_config.reference_files, rmsd_config.top_k, rmsd_config.mean_loss_weight, rmsd_config.distance_loss_weight, device=self.device)
        elif self.config.loss_function.loss_function_type == "density":
            density_config = self.config.loss_function.density_loss_function
            loss_function = DensityGuidanceLossFunction(density_config.reference_pdbs, self.config.protein.reference_raw_pdb, self.config.protein.reference_raw_pdb_chain, density_config.density_file, self.config.protein.residue_range, batch_size=self.config.general.batch_size, device=self.device, rmax=self.config.loss_function.density_loss_function.rmax)
        elif self.config.loss_function.loss_function_type == "pairwise":
            pairwise_config = self.config.loss_function.pairwise_loss_function
            distance_matrix = get_distance_matrix_mask(pairwise_config.reference_pdb, pairwise_config.atom_type).to(self.device)
            loss_function = PairwiseDistancesLossFunction(distance_matrix, pairwise_config.rmax)
        elif self.config.loss_function.loss_function_type == "hydrogen_noe":
                noe_config = self.config.loss_function.hydrogen_noe_loss_function
                loss_function = NOEHydrogenLossFunction(restraint_file=noe_config.reference_nmr,
                                                     pdb_file=noe_config.pdb_file,
                                                          atom_array=self.model_manager.atom_array,
                                                          device=self.device, 
                                                          methyl_relax_file=noe_config.methyl_relax_file,
                                                          methyl_relax_scale=noe_config.methyl_relax_scale,
                                                          amide_rdc_file=noe_config.amide_rdc_file,
                                                          amide_rdc_scale=noe_config.amide_rdc_scale,
                                                          amide_relax_file=noe_config.amide_relax_file,
                                                          amide_relax_scale=noe_config.amide_relax_scale,
                                                          methyl_rdc_scale=noe_config.methyl_rdc_scale,
                                                          methyl_rdc_file=noe_config.methyl_rdc_file,
                                                          noe_scale=noe_config.noe_scale,
                                                          op_n_bootstrap=noe_config.op_n_bootstrap,
                                                          iid_loss=noe_config.iid_loss)
        elif self.config.loss_function.loss_function_type == "heavy_noe":
            noe_config = self.config.loss_function.heavy_noe_loss_function
            loss_function = NOEHeavyAtomLossFunction(restraint_file=noe_config.reference_nmr,
                                                    pdb_file=noe_config.pdb_file,
                                                        atom_array=self.model_manager.atom_array,
                                                        device=self.device, 
                                                        iid_loss=noe_config.iid_loss)
        elif self.config.loss_function.loss_function_type == "epr":
            epr_config = self.config.loss_function.epr_loss_function
            loss_function = EPRLossFunction(constraints_file=epr_config.constraints_file,
                                            atom_array=self.model_manager.atom_array,
                                            rotamers_statistics_file=epr_config.rotamers_statistics_file,
                                            rotamers_structures_folder=epr_config.rotamers_structures_folder,
                                            device=self.device, 
                                            batch_size=self.config.general.batch_size,
                                            sample_points=epr_config.sample_points,
                                                        )
        elif self.config.loss_function.loss_function_type == "relax_times":
            relax_times_config = self.config.loss_function.relax_times_loss_function
            loss_function = RelaxTimesLossFunction(atom_array = self.model_manager.atom_array,
                                                    coefficient_files=relax_times_config.coefficient_files,
                                                   data_file=relax_times_config.data_file,
                                                   device=self.device,
                                                   num_discretize_points=relax_times_config.num_discretize_points,
                                                   batch_size = self.config.general.batch_size,
                                                        )
        else:
            raise ValueError(f"the loss function {self.config.loss_function.loss_function_type} is not a valid option")
        
        loss_functions = [loss_function]
        weights = [1]
        if self.config.loss_function.violation_loss_weight > 0:
            loss_functions.append(ViolationLossFunction(self.model_manager.atom_array))
            weights.append(self.config.loss_function.violation_loss_weight)
        if self.config.loss_function.bond_length_loss_weight > 0:
            loss_functions.append(BondLengthLossFunction(self.model_manager.atom_array, self.device))
            weights.append(self.config.loss_function.bond_length_loss_weight)
        return MultiLossFunction(loss_functions, weights)

    def query_msa_server(self):
        query_msa_server(self.config.model_manager.msa_save_dir, self.config.protein.pdb_id, self.config.protein.sequence)

    def save_state(self, structures, name, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        structures = self.model_manager.align_models_to_reference(structures.detach())
        for i in range(structures.shape[0]):
            self.model_manager.save_structure_pdb(structures[i].cpu(), self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb")
        if self.loss_function is not None:
            self.loss_function.save_state(structures, folder_path)

    def get_residue_mask(self):
        residue_range = self.config.protein.residue_range
        residue_mask = [residue_range[0] <= i <= residue_range[1] for i in range(len(self.config.protein.sequence))]
        residue_mask = torch.tensor(residue_mask)
        return residue_mask

    def get_initial_latents(self):
        return self.model_manager.get_x_noisy(self.model_manager.get_x_start(self.config.general.batch_size))

    def run_full_diffusion_process(self, latents):
        structures = latents.clone()
        steps_generator = tqdm(range(self.config.model_manager.diffusion_N), "running diffusion process")

        structures_gradient_norm = self.config.diffusion_process.guidance.guidance_direction_scale_factor
        normalize_gradients = self.config.diffusion_process.guidance.normalize_gradients
        guidance_direction = None
        wandb_log = None
        start_guidance_from = 0
        for i in steps_generator:
            if self.config.general.apply_diffusion_guidance and i > start_guidance_from:      
                structures.requires_grad = True

            x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(structures, start_index=i, inplace_safe=False)
            guidance_direction = None

            if self.config.general.apply_diffusion_guidance and i > start_guidance_from:
                wandb_log = {}
                if i != (self.config.model_manager.diffusion_N - 1):
                    loss_value, new_x_0_hat = self.loss_function(x_0_hat, i / (self.config.model_manager.diffusion_N - 1))
                    wandb_log = self.loss_function.wandb_log(x_0_hat)
                    if new_x_0_hat is not None:
                        wandb_log = self.loss_function.wandb_log(new_x_0_hat)
                        x_0_hat = new_x_0_hat
                    loss_value.backward()
                    steps_generator.set_description(f"running diffusion process, loss: {loss_value.item():.5f}")                
                
                    # Get guidance and add to structures (noisy variable)
                    with torch.no_grad():
                        guidance_direction = structures.grad
                        if normalize_gradients and not (guidance_direction.abs() < 1e-6).all():
                            normalization_norm = guidance_direction.flatten(1,-1).norm(dim=-1)
                            normalization_norm[normalization_norm < 1e-4] = 1
                            guidance_direction = guidance_direction / normalization_norm[:,None, None]
                        guidance_direction = guidance_direction * structures_gradient_norm
                        structures.grad = None

            structures = self.model_manager.get_x_t_from_x_0_hat(structures, x_0_hat, i, i+1, guidance_direction=guidance_direction, step_size=self.config.diffusion_process.guidance.step_size, normalize_gradients=normalize_gradients)
            if self.loss_function is not None:
                self.loss_function.post_optimization_step()
            if i < self.config.model_manager.diffusion_N - 1:
                structures = self.model_manager.get_x_noisy(structures, i + 1)
            structures = structures.detach().clone()
            if wandb_log is not None:
                wandb.log(wandb_log)
        return structures

    def run(self):
        latents = self.get_initial_latents()

        structures = self.run_full_diffusion_process(latents)
        sub_folder_name = "diffusion_process"
        self.save_state(structures, self.config.protein.pdb_id, os.path.join(self.experiment_save_dir, sub_folder_name))

        metadata = {
            "pdb_id": self.config.protein.pdb_id[:4],
            "residue_range": self.config.protein.residue_range,
            "sequence": self.config.protein.sequence,
            "chain": self.config.protein.reference_raw_pdb_chain,
            "pdb_residue_range": self.config.protein.pdb_residue_range
        }
        with open(os.path.join(self.experiment_save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def get_un_broken_samples(self, samples_directory, residue_range, bond_max_threshold):
        """
            this function will go over the files in self.samples_directory and return the files of the proteins which are not borken
        """
        pdb_files = [os.path.join(samples_directory, file) for file in os.listdir(samples_directory) if file.endswith(".pdb")]
        un_broken_files = []
        for pdb_file in pdb_files:
            structure = gemmi.read_pdb(pdb_file)
            model = structure[0]
            chain = model[0]
            bonded_atom_pairs = find_bonded_pairs(chain, residue_range)
            bond_distances = np.array([np.linalg.norm(np.array(list(pair[0][0].pos)) - np.array(list(pair[1][0].pos))) for pair in bonded_atom_pairs])
            if bond_distances.max() < bond_max_threshold:
                un_broken_files.append(pdb_file)
        return un_broken_files

    def relax_files(self, files):
        relaxed_file_names = []
        for file in files:
            folder_name = os.path.dirname(file)
            basename = os.path.basename(file)
            relaxed_file_name = os.path.join(folder_name, "relaxed", basename)
            os.makedirs(os.path.dirname(relaxed_file_name), exist_ok=True)
            relax_pdb(file, relaxed_file_name, use_gpu=False)
            delete_hydrogens(relaxed_file_name)
            relaxed_file_names.append(relaxed_file_name)
        return relaxed_file_names

    def align_relaxed_samples(self, files, reference_pdb_path, residue_range, device):
        aligned_files = []
        with torch.no_grad():
            structures = torch.cat([load_pdb_atom_locations(file, device) for file in files])
            coordinates_pdb, _, _, _, _ = get_sampler_pdb_inputs(reference_pdb_path, residue_range, device)

            ref_pose = coordinates_pdb.repeat(structures.shape[0], 1, 1)
            mask = torch.ones_like(ref_pose[0, :, 0])
            _, structures, _, _ = self_aligned_rmsd(structures, ref_pose, mask)

            for i in range(len(files)):
                structure_i, file_i = structures[i], files[i]
                aligned_file_i = write_back_pdb_coordinates(file_i, file_i, structure_i[None])
                aligned_files.append(aligned_file_i)

        return aligned_files

    def relax_structures(self, bond_max_threshold):
        un_broken_files = self.get_un_broken_samples(os.path.join(self.experiment_save_dir, "diffusion_process"), self.config.protein.residue_range, bond_max_threshold)
        relaxed_files = self.relax_files(un_broken_files)
        aligned_files = self.align_relaxed_samples(relaxed_files, self.config.protein.reference_pdb, self.config.protein.residue_range, self.device)
        return aligned_files

def main():
    args = parser.parse_args()
    file_path = args.configuration_file
    device = args.device
    config = load_config(file_path)
    pipeline = ExperimentManager(config, device)
    # pipeline.run()
    # relaxed_files = pipeline.relax_structures(config.loss_function.density_loss_function.bond_max_threshold)
    relaxed_dir = os.path.join(config.general.output_folder, config.general.name, "diffusion_process", "relaxed")


    # Selection
    if config.loss_function.loss_function_type == "density":
        # Phenix and CCP4 setups
        phenix_setup_sh = config.loss_function.density_loss_function.phenix_env_path
        ccp4_setup_sh = config.loss_function.density_loss_function.ccp4_env_path

        # Start density metrics
        processor = DensityMetricsProcessor(
            config=config,
            relaxed_dir=relaxed_dir,
            device=device,
            ccp4_setup_sh=ccp4_setup_sh,
            phenix_setup_sh=phenix_setup_sh
        )
        processor.process_all_metrics()

if __name__ == "__main__":
    main()
