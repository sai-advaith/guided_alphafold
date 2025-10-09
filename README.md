# Guided Protein Structure Prediction

## Overview

This codebase implements a guided protein structure prediction pipeline that incorporates experimental data from three different structural biology modalities to improve AlphaFold3's prediction accuracy. The system uses a diffusion-based approach guided by experimental log-likelihoods to generate protein structures that are consistent with:

- **Cryo-EM**: Electrostatic potential maps from electron microscopy
- **X-ray Crystallography**: Real-space electron density maps (2mFo-DFc or END maps) from crystallographic data  
- **NMR Spectroscopy**: Distance, order parameters, and dihedrals restraints (NOE, dihedral angles, RDC, order parameters)

The pipeline processes experimental data, runs experiment-guided structure prediction, performs structural relaxation using AMBER99 force field, and evaluates results using modality-specific metrics.

## Installation

### Environment Setup

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate guided_af3
   ```

2. **Download Protenix model weights and data:**
   
   This pipeline is built on top of [Protenix](https://github.com/bytedance/Protenix), a PyTorch reproduction of DeepMind's AlphaFold3. Download the required pre-trained model weights and chemical component data files:
   
   ```bash
   # Download model weights (v0.2.0)
   wget -P src/af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
   
   # Download chemical component dictionary files
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl
   ```
   
   For more information, visit the Protenix repository: https://github.com/bytedance/Protenix

### External Dependencies

3. **END RAPID (for X-ray absolute scale maps):**
   
   Download and install the END RAPID script for rendering absolute scale electron density maps:
   ```bash
   wget https://bl831.als.lbl.gov/END/RAPID/end.rapid/Distributions/end.rapid.tar.gz
   tar -xzf end.rapid.tar.gz
   ```
   
   Installation manual: https://bl831.als.lbl.gov/END/RAPID/end.rapid/Documentation/end.rapid.Manual.htm#InstallationInstructions

4. **Phenix 1.21.2 (for X-ray and Cryo-EM):**
   
   Required for structure refinement and validation metrics.
   
   Download from: http://www.phenix-online.org/

5. **CCP4 8.0 (for X-ray):**
   
   Required for crystallographic computations and map processing.
   
   Download from: http://www.ccp4.ac.uk/

6. **AMBER99 relaxation using AlphaFold2 (X-ray and NMR):**
   
   Recommended for final structure relaxation.
   
   Download from: https://github.com/google-deepmind/alphafold

## Usage

### 1. Cryo-EM Guided Structure Prediction

Fits protein structures to electrostatic potential maps using cryo-EM data from the EMDB.

**Command:**
```bash
python3 run_em.py <pdb_id> <emdb_id> <renumbered_file_path> <assembly_identifier> \
    --phenix_setup_sh <phenix_setup_path> \
    --sequences <seq1> <seq2> ... \
    --counts <count1> <count2> ... \
    [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the protein structure
- `emdb_id`: EMDB identifier for the EM density map
- `renumbered_file_path`: Path to renumbered and reordered PDB file
- `assembly_identifier`: Identifier for the assembly (e.g., biological assembly name)
- `--phenix_setup_sh`: Path to Phenix setup shell script (e.g., `/path/to/phenix-1.21.2/phenix_env.sh`)
- `--sequences`: Space-separated sequences for each chain in the assembly
- `--counts`: Space-separated integer counts corresponding to each sequence (must match length of sequences)

**Optional Parameters:**
- `--dihedrals_file`: Path to dihedral restraints file
- `--noe_restraints_file`: Path to NOE restraints file  
- `--noe_pdb_file`: Path to NOE reference PDB file
- `--input_directory`: Directory for input files (default: `pipeline_inputs`)
- `--output_directory`: Directory for output files (default: `pipeline_outputs`)
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
python3 run_em.py 7dac 30622 pdb7dac_seqaligned_short.pdb amyloid_7dac_short_mmseq2 \
    --phenix_setup_sh  /opt/ccp4-8.0/bin/ccp4.setup-sh \
    --sequences PLVNIYNCSGVQVGDNNYLTMQQT \
    --counts 3 \
    --device cuda:0
```
The renumbered file is the path to the PDB file containing atomic coordinates where the residues were renumbered to match the absolute 1-index of the residues of the sequence. An example `pdb7dac_seqaligned_short.pdb` is included in the repository.

### 2. X-ray Crystallography Guided Structure Prediction

Generates ensemble structures fitted to X-ray crystallographic electron density maps.

**Command:**
```bash
python3 run_xray.py <pdb_id> <chain_id> <region_sub_sequence> \
    --ccp4_setup_sh <ccp4_setup_path> \
    --phenix_setup_sh <phenix_setup_path> \
    [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the protein structure
- `chain_id`: Chain identifier within the PDB structure (e.g., `A`, `B`)
- `region_sub_sequence`: Subsequence of amino acids defining the region of interest
- `--ccp4_setup_sh`: Path to CCP4 setup shell script (e.g., `/path/to/ccp4-8.0/bin/ccp4.setup-sh`)
- `--phenix_setup_sh`: Path to Phenix setup shell script

**Optional Parameters:**
- `--input_directory`: Directory for input files (default: `pipeline_inputs`)
- `--output_directory`: Directory for output files (default: `pipeline_outputs`)
- `--map_type`: Type of electron density map to use: `2fofc` (standard) or `end` (absolute scale END map) (default: `end`)
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name  
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
python3 run_xray.py 2izr A SLTGT \
    --ccp4_setup_sh /opt/ccp4-8.0/bin/ccp4.setup-sh \
    --phenix_setup_sh /opt/phenix-1.21.2/phenix_env.sh \
    --map_type end \
    --device cuda:0
```

### 3. NMR Guided Structure Prediction

Fits protein structures to NMR experimental restraints including NOE distances, dihedral angles, RDC, and relaxation data.

**Command:**
```bash
python3 run_nmr.py <pdb_id> [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the NMR structure

**Optional Parameters:**
- `--input_directory`: Directory containing NMR input files (default: `nmr_pipeline_inputs`)
  - Should contain subdirectories: `pdbs/`, `restraints/`, `metadata/`
- `--output_directory`: Directory for output files (default: `nmr_pipeline_outputs`)
- `--methyl_rdc_file`: Path to methyl RDC (Residual Dipolar Coupling) file
- `--amide_rdc_file`: Path to amide RDC file
- `--amide_relax_file`: Path to amide relaxation (S²) file
- `--methyl_relax_file`: Path to methyl relaxation file
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
python3 run_nmr.py 1u0p \
    --input_directory nmr_pipeline_inputs \
    --output_directory nmr_pipeline_outputs \
    --device cuda:0
```

## Experiment Tracking

The pipeline supports experiment tracking via Weights & Biases (wandb). To enable tracking:

1. Create a wandb account at https://wandb.ai
2. Obtain your API key from https://wandb.ai/authorize
3. Pass the API key and project name to any run script:
   ```bash
   --wandb_key <your_api_key> --wandb_project <project_name>
   ```

## Citation

Soon.

## License

Soon.

## Contact

Correspondence Email: `Alexander.Bronstein@ist.ac.at`
