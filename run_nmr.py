from experiment_manager import ExperimentManager
from src.utils.io import load_config
import argparse
from src.utils.process_pipeline_inputs.preprocess_nmr_inputs import main as preprocess_nmr_inputs
from src.metrics.nmr_metrics import run_nmr_metrics
import os

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('pdb_id', type=str, help="PDB ID for NMR fitting")
    parser.add_argument('--input_directory', type=str, required=False, default="nmr_pipeline_inputs")
    parser.add_argument('--output_directory', type=str, required=False, default="nmr_pipeline_outputs")
    parser.add_argument('--wandb_key', type=str, required=False, default=None)
    parser.add_argument('--wandb_project', type=str, required=False, default=None)
    parser.add_argument('--methyl_rdc_file', type=str, required=False, default=None)
    parser.add_argument('--amide_rdc_file', type=str, required=False, default=None)
    parser.add_argument('--amide_relax_file', type=str, required=False, default=None)
    parser.add_argument('--methyl_relax_file', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default="cuda:0")
    args = parser.parse_args()

    # Prepare config file
    config_file_path = preprocess_nmr_inputs(args.pdb_id, args.input_directory, args.output_directory, args.wandb_key, args.wandb_project, args.methyl_rdc_file, args.amide_rdc_file, args.amide_relax_file, args.methyl_relax_file)

    # Run guidance
    config = load_config(config_file_path)
    pipeline = ExperimentManager(config, args.device)
    pipeline.run()

    # Metrics!
    output_directory = os.path.join(config.general.output_folder, config.general.name)
    metrics_results_path = os.path.join(output_directory, "diffusion_process", f"{config.protein.pdb_id}_metrics.csv")
    run_nmr_metrics(pdb_output_folder=output_directory, md_file=config.protein.reference_pdb, restraint_file=config.loss_function.nmr_loss_function.reference_nmr, add_hydrogen=True, relax_colabfold=True, results_path=metrics_results_path, additional_protein_files=None, order_params_files=None, noe=True, order_params=False)
if __name__ == "__main__":
    main()

# Metrics order parameter file format
#         "amide_relax": "pipeline_inputs/nmr_s_2/ubi_solution_S2.csv",
#         "amide_rdc": "pipeline_inputs/nmr_s_2/ubi_nh_rdc.csv",
#         "methyl_relax": "pipeline_inputs/nmr_s_2/ubi_methyl_relaxation.csv",
#         "methyl_rdc": "pipeline_inputs/nmr_s_2/ubi_methyl_rdc.csv"}