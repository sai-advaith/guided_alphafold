import subprocess
import argparse

from src.utils.process_pipeline_inputs.preprocess_inputs import main as preprocess_inputs

def main():
    parser = argparse.ArgumentParser(description='Run X-ray preprocessing with specified parameters')
    parser.add_argument('pdb_id', type=str, help='PDB ID for the protein structure')
    parser.add_argument('chain_id', type=str, help='Chain ID within the PDB structure')
    parser.add_argument('sub_seq', type=str, help='Subsequence of the amino acid sequence')
    parser.add_argument('--directory', type=str, default='pipeline_inputs', 
                       help='Directory where inputs will be loaded (default: pipeline_inputs)')
    parser.add_argument('--ccp4_setup_sh', type=str, default='/nfs/scistore20/bronsgrp/amaddipa/ccp4-8.0/bin/ccp4.setup-sh', help='CCP4 setup shell script')
    parser.add_argument('--phenix_setup_sh', type=str, default='/nfs/scistore20/bronsgrp/amaddipa/phenix-1.21.2-5419/phenix_env.sh', help='Phenix setup shell script')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb key')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')
    parser.add_argument('--map_type', type=str, default='end', help='map type')


    args = parser.parse_args()
    
    print(f"Running preprocessing for PDB: {args.pdb_id}, Chain: {args.chain_id}, Subsequence: {args.sub_seq}")
    guided_config_file_path = preprocess_inputs(args.pdb_id.lower(), args.chain_id, args.sub_seq, args.directory, args.ccp4_setup_sh, args.phenix_setup_sh, args.wandb_key, args.wandb_project, args.map_type)
    
    if guided_config_file_path:
        print(f"Preprocessing completed. Config file generated: {guided_config_file_path}")
        print("Running experiment_manager.py...")
        
        # Run experiment_manager.py with the generated config file
        try:
            result = subprocess.run([
                'python', 'experiment_manager.py', 
                '--configuration_file', guided_config_file_path,
            ], check=True, capture_output=True, text=True)
            
            print("Experiment manager completed successfully!")
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"Error running experiment_manager.py: {e}")
            print(f"Return code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return 1
   
    else:
        print("Error: No config file was generated from preprocessing")
        return 1
    return 0

if __name__ == '__main__':
    main()
