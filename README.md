Setup conda environment using `environment.yaml` file.

Absolute scale electron denstiy maps were rendered as END maps. Download end rapid script from https://bl831.als.lbl.gov/END/RAPID/end.rapid/Distributions/end.rapid.tar.gz
 
Installation manual for the script can be found in https://bl831.als.lbl.gov/END/RAPID/end.rapid/Documentation/end.rapid.Manual.htm#InstallationInstructions

The code was extensively tested using Phenix (1.21.2) from http://www.phenix-online.org/ and CCP4 (8.0) from http://www.ccp4.ac.uk/

For AMBER99 relaxation, it is recommended to download AlphaFold2 package (https://github.com/google-deepmind/alphafold).

To run the script follow:
`python3  run_xray.py <pdb_id> <chain_id> <region_sub_sequence> --input_directory <input_directory_path> --output_directory <output_directory_path> --ccp4_setup_sh <ccp4 installation path> --phenix_setup_sh <phenix_setup_path> --wandb_key <wandb api key> --wandb_project <wandb_project_name> --map_type <2fofc or end> --device cuda:0`
