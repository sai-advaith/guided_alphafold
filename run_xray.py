import argparse
from experiment_manager import ExperimentManager
from src.metrics.density_metrics_processor import DensityMetricsProcessor
from src.utils.io import load_config
import os

from src.utils.process_pipeline_inputs.preprocess_xray_inputs import main as preprocess_xray_inputs

def main():
    """
    Run ensemble fitting using xray crystallographic data
    """
    parser = argparse.ArgumentParser(description='X-ray ensemble fitting pipeline')
    parser.add_argument('pdb_id', type=str, help='PDB ID for the protein structure')
    parser.add_argument('chain_id', type=str, help='Chain ID within the PDB structure')
    parser.add_argument('sub_seq', type=str, help='Subsequence of the amino acid sequence')
    parser.add_argument('--input_directory', type=str, default='pipeline_inputs', help='Directory where inputs will be loaded (default: pipeline_inputs)')
    parser.add_argument('--output_directory', type=str, default='pipeline_outputs', help='Directory where outputs will be saved (default: pipeline_outputs)')
    parser.add_argument('--ccp4_setup_sh', type=str, default=None, help='CCP4 setup shell script')
    parser.add_argument('--phenix_setup_sh', type=str, default=None, help='Phenix setup shell script')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb key')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')
    parser.add_argument('--map_type', type=str, default='end', help='map type')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    args = parser.parse_args()
    
    print(f"Running preprocessing for PDB: {args.pdb_id}, Chain: {args.chain_id}, Subsequence: {args.sub_seq}")
    guided_config_file_path = preprocess_xray_inputs(args.pdb_id.lower(), args.chain_id, args.sub_seq, args.input_directory, args.output_directory,  args.ccp4_setup_sh, args.phenix_setup_sh, args.wandb_key, args.wandb_project, args.map_type)

    if guided_config_file_path is not None:
        config = load_config(guided_config_file_path)
        pipeline = ExperimentManager(config, args.device)

        print("Running guidance!")
        pipeline.run()

        print("Relaxing structures!")
        pipeline.relax_structures(config.loss_function.density_loss_function.bond_max_threshold)
        relaxed_dir = os.path.join(config.general.output_folder, config.general.name, "diffusion_process", "relaxed")

        print("Running metrics!")
        processor = DensityMetricsProcessor(
            config=config,
            relaxed_dir=relaxed_dir,
            device=args.device,
            ccp4_setup_sh=args.ccp4_setup_sh,
            phenix_setup_sh=args.phenix_setup_sh
        )
        processor.process_all_metrics()
   
    else:
        raise ValueError("Error: No config file was generated from preprocessing")

if __name__ == '__main__':
    main()
