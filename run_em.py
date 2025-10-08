import argparse
from experiment_manager import ExperimentManager
from src.metrics.density_metrics_processor import DensityMetricsProcessor
from src.utils.io import load_config
import os

from src.metrics.em_metrics import run_em_metrics
from src.utils.process_pipeline_inputs.preprocess_em_inputs import main as preprocess_em_inputs

def main():
    """
    Run structure fitting using EM data
    """
    parser = argparse.ArgumentParser(description='EM structure fitting pipeline')
    parser.add_argument('pdb_id', type=str, help='PDB ID for the protein structure')
    parser.add_argument('emdb_id', type=int, help='EMDB ID for the EM data') # For the esp map
    parser.add_argument('renumbered_file_path', type=str, help='Renumbered and reordered PDB file path')
    parser.add_argument('assembly_identifier', type=str, help='Assembly identifier')
    parser.add_argument('--phenix_setup_sh', type=str, default=None, help='Phenix setup shell script')
    parser.add_argument('--dihedrals_file', type=str, default=None, help='Dihedrals file path')
    parser.add_argument('--noe_restraints_file', type=str, default=None, help='NOE restraints file path')
    parser.add_argument('--noe_pdb_file', type=str, default=None, help='NOE PDB file path')
    parser.add_argument('--sequences', nargs='+', help='Sequences for each chain')
    parser.add_argument('--counts', nargs='+', type=int, help='Counts for each sequence')
    parser.add_argument('--input_directory', type=str, default='pipeline_inputs', help='Directory where inputs will be loaded (default: pipeline_inputs)')
    parser.add_argument('--output_directory', type=str, default='pipeline_outputs', help='Directory where outputs will be saved (default: pipeline_outputs)')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb key')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    args = parser.parse_args()
    
    print(f"Running structure fitting for PDB: {args.pdb_id}, EMDB ID: {args.emdb_id}")

    assert args.phenix_setup_sh is not None, "Phenix setup shell script must be provided"
    assert len(args.sequences) == len(args.counts), "Sequences and counts must have the same length"

    # Get the config file
    em_config_file_path = preprocess_em_inputs(args.pdb_id, args.sequences, args.counts, args.emdb_id, args.renumbered_file_path, args.assembly_identifier, args.input_directory, args.output_directory, args.wandb_key, args.wandb_project, args.dihedrals_file, args.noe_restraints_file, args.noe_pdb_file)

    # Run the experiment
    config = load_config(em_config_file_path)
    pipeline = ExperimentManager(config, args.device)
    pipeline.run()

    # Metrics (only one sample)
    guided_model_path = os.path.join(config.general.output_folder, config.protein.pdb_id, config.protein.assembly_identifier, "diffusion_process", f"{config.protein.pdb_id}_0.pdb")
    run_em_metrics(em_config_file_path, guided_model_path, args.phenix_setup_sh)

if __name__ == '__main__':
    main()