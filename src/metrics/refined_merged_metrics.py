from Bio import PDB
from Bio.PDB import PDBIO
import os
import torch
from Bio.PDB import PDBParser
import numpy as np
from Bio.PDB.PDBIO import Select

from .density_omp import OMPMetric, DensityLogger, PDBManager
from ..utils.io import AMINO_ACID_ATOMS_ORDER, get_sampler_pdb_inputs

def log_fc_maps(sample_pdbs, reference_pdbs, ensemble_occupancies, pdb_occupancies, density_file, pdb_residue_range, samples_directory, selection_method, pdb_id, rmax, chain_id, raw_pdb_file, device=torch.device("cuda:0")):
    # Fc selected, fc gt, and fo
    fc_selected_file_path = os.path.join(samples_directory, selection_method, f"{pdb_id}_fc_sel_{selection_method[:3]}.ccp4")
    fc_pdb_file_path = os.path.join(samples_directory, selection_method, f"{pdb_id}_fc_pdb_{selection_method[:3]}.ccp4")

    # PDB density
    fc_pdb_density = 0
    ref_pdb_path = reference_pdbs[0]
    for (i, reference_pdb_file_path) in enumerate(reference_pdbs):
        density_calc_obj = DensityLogger(density_file, [reference_pdb_file_path], pdb_residue_range, rmax, device, raw_pdb_file_path=raw_pdb_file, chain_id=chain_id, ref_pdb_path=ref_pdb_path)
        reference_pdb_obj = PDBManager(reference_pdb_file_path, pdb_residue_range, device)
        fc_obj_i = density_calc_obj.calculate_density([reference_pdb_obj]) * pdb_occupancies[i]
        fc_pdb_density = fc_pdb_density + fc_obj_i
    fc_pdb = fc_pdb_density
    density_calc_obj.save_density(fc_pdb, fc_pdb_file_path)

    # Get selected Fc
    fc_selected_density = 0

    # Split the samples
    for (i, pdb_file_path) in enumerate(sample_pdbs):
        density_calc_obj = DensityLogger(density_file, [pdb_file_path], pdb_residue_range, rmax, device, raw_pdb_file_path=raw_pdb_file, chain_id=chain_id, ref_pdb_path=ref_pdb_path)
        reference_pdb_obj = PDBManager(pdb_file_path, pdb_residue_range, device)
        fc_obj_i = density_calc_obj.calculate_density([reference_pdb_obj]) * ensemble_occupancies[i]
        fc_selected_density = fc_selected_density + fc_obj_i
    fc_selected = fc_selected_density
    density_calc_obj.save_density(fc_selected, fc_selected_file_path)

def get_refined_cosine_similarity(sample_pdbs, reference_pdbs, ensemble_occupancies, pdb_occupancies, selection_method, rmax, relaxed_dir, pdb_residue_range, config, device=torch.device("cuda:0")):
    # Stack structures together
    ensemble_structures = []
    for sample_pdb in sample_pdbs:
        coords, _, _, _, _ = get_sampler_pdb_inputs(sample_pdb, pdb_residue_range, device)
        ensemble_structures.append(coords)
    ensemble_structures_cat = torch.cat(ensemble_structures, dim=0)

    new_metrics_calculator = OMPMetric(
        samples_directory=relaxed_dir,
        rmax=rmax,
        reference_density_file=config.loss_function.density_loss_function.density_file,
        residue_range=pdb_residue_range,
        altloc_a_path=reference_pdbs[0],
        altloc_b_path=reference_pdbs[1] if len(reference_pdbs) > 1 else None,
        bond_max_threshold=2.1,
        device=device,
        raw_pdb_file_path=config.protein.reference_raw_pdb,
        chain_id=config.protein.reference_raw_pdb_chain,
        mtz_file_path=config.loss_function.density_loss_function.mtz_file,
        reference_pdb_file_path=reference_pdbs[0],
        pdb_id=config.protein.pdb_id
    )
    print(ensemble_structures_cat.shape[1] == new_metrics_calculator.altloc_A["structure_coordinates"].shape[1])
    ensemble_cosine_similarity = new_metrics_calculator.get_optimized_cosine_similarity(ensemble_structures_cat, optimize_bfactor=False).item()
    pdb_cosine_similarity = new_metrics_calculator.get_optimized_cosine_similarity(torch.cat([new_metrics_calculator.altloc_A["structure_coordinates"], new_metrics_calculator.altloc_B["structure_coordinates"]]), optimize_bfactor=False).item()

    # Fc
    log_fc_maps(sample_pdbs, reference_pdbs, ensemble_occupancies, pdb_occupancies, config.loss_function.density_loss_function.density_file, pdb_residue_range, relaxed_dir, selection_method, new_metrics_calculator.pdb_id, rmax, config.protein.reference_raw_pdb_chain, config.protein.reference_raw_pdb, device=torch.device("cuda:0"))

    # Fo
    fo_slice_file_path = os.path.join(new_metrics_calculator.samples_directory, selection_method, f"{new_metrics_calculator.pdb_id}_fo_{selection_method[:3]}.ccp4")
    fo_slice = new_metrics_calculator.sliced_density.clone()
    new_metrics_calculator.save_zone_of_interest_density(fo_slice.clone().detach(), fo_slice_file_path)

    return ensemble_cosine_similarity, pdb_cosine_similarity

def get_cryst1_line(pdb_file):
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("CRYST1"):
                return line
    return None

def replace_cryst1_inplace(pdb_file, new_cryst1_line):
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    cryst1_found = False
    new_lines = []

    for line in lines:
        if line.startswith("CRYST1"):
            new_lines.append(new_cryst1_line.rstrip('\n') + '\n')
            cryst1_found = True
        else:
            new_lines.append(line)

    if not cryst1_found:
        # If no CRYST1 line found, insert it at the top
        new_lines.insert(0, new_cryst1_line.rstrip('\n') + '\n')

    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)

def count_altlocs_in_region(pdb_file, chain_id, res_start, res_end):
    """
    Detect altloc identifiers present in a residue region.
    
    Returns ['A'], [1.0] if no altlocs found.
    Returns actual altloc IDs and occupancies if altlocs are present.
    
    Parameters:
    - pdb_file: path to PDB file
    - chain_id: chain identifier  
    - res_start: start residue number
    - res_end: end residue number
    
    Returns:
    - altloc_ids: list of altloc identifiers in the region
    - occupancies: list of occupancy values
    """
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    
    # Dictionary to store unique altloc IDs and their occupancies
    found_altlocs = {}
    
    # Search through the structure
    for model in structure:
        if chain_id not in model:
            continue
            
        chain = model[chain_id]
        
        for residue in chain:
            residue_number = residue.id[1]
            
            # Check if residue is in our target range
            if res_start <= residue_number <= res_end:
                
                # Examine every atom in this residue
                for atom in residue:
                    
                    if atom.is_disordered():
                        # Disordered atom - multiple altlocs
                        for altloc_key in atom.child_dict:
                            alt_atom = atom.child_dict[altloc_key]
                            altloc_id = alt_atom.get_altloc()
                            occupancy = alt_atom.get_occupancy()
                            found_altlocs[altloc_id] = occupancy
                    
                    else:
                        # Regular atom - check if it has altloc label
                        altloc_id = atom.get_altloc()
                        occupancy = atom.get_occupancy()
                        
                        # Only store non-blank altlocs
                        if altloc_id.strip():  # Non-empty altloc
                            found_altlocs[altloc_id] = occupancy
    
    # Return results
    if not found_altlocs:
        # No altlocs found - return default
        return ['A'], [1.0]
    else:
        # Return found altlocs
        altloc_ids = list(found_altlocs.keys())
        occupancies = list(found_altlocs.values())
        return altloc_ids, occupancies

def save_refined_pdb_by_altloc(pdb_path, chain_id, start_resnum, end_resnum, altloc, temp_dir_root, prefix):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)

    class AltLocChainSelectNoAltLoc(Select):
        def __init__(self, chain_id, start_resnum, end_resnum, altloc):
            self.chain_id = chain_id
            self.start_resnum = start_resnum
            self.end_resnum = end_resnum
            self.altloc = altloc

        def accept_chain(self, chain):
            return chain.id == self.chain_id

        def accept_residue(self, residue):
            resname = residue.get_resname()
            if resname not in AMINO_ACID_ATOMS_ORDER.keys():
                return False
            if resname == 'HOH':
                return False
            return True

        def accept_atom(self, atom):
            residue = atom.get_parent()
            resnum = residue.id[1]
            atom_altloc = atom.get_altloc() or ' '
            if self.start_resnum <= resnum <= self.end_resnum:
                if atom_altloc == self.altloc or atom_altloc == " ":
                    # Remove altloc label by setting it to ' '
                    atom.set_altloc(' ')
                    return True
                else:
                    return False
            else:
                if atom_altloc == 'A' or atom_altloc == ' ':
                    atom.set_altloc(' ')
                    return True
                else:
                    return False

    io = PDBIO()
    io.set_structure(structure)
    output_file_path = os.path.join(temp_dir_root, f"{prefix}_altloc_{altloc}.pdb")
    io.save(output_file_path, select=AltLocChainSelectNoAltLoc(chain_id, start_resnum, end_resnum, altloc))
    return output_file_path

def save_raw_pdb_by_altloc(raw_pdb_path, chain, pdb_residue_range, one_indexed_residue_range, altloc):
    # Temp dir for logs
    temp_dir_root = "refined_metrics_temp"
    os.makedirs(temp_dir_root, exist_ok=True)

    pdb_id = os.path.basename(raw_pdb_path).split(".")[0][:4]
    raw_altloc_output_file = save_refined_pdb_by_altloc(raw_pdb_path, chain, pdb_residue_range[0], pdb_residue_range[1], altloc, temp_dir_root, prefix="pdb")
    raw_altloc_output_fixed_file = os.path.join(temp_dir_root, f"pdb_altloc_{altloc}_fixed.pdb")
    
    raw_pdb_file_path = os.path.join(os.path.dirname(raw_pdb_path), f"{pdb_id}_chain_{chain}_altloc_{altloc}_fixed.pdb")
    replace_residue_range(raw_altloc_output_file, raw_pdb_file_path, raw_altloc_output_fixed_file, pdb_residue_range[0], pdb_residue_range[1], one_indexed_residue_range[0], one_indexed_residue_range[1], chain, "A")

    cryst1_line = get_cryst1_line(raw_pdb_path)
    replace_cryst1_inplace(raw_altloc_output_fixed_file, cryst1_line)
    return raw_altloc_output_fixed_file

def replace_residue_range(pdb_target_path, pdb_source_path, output_path, 
                         start_pdb_index, end_pdb_index, 
                         start_one_index, end_one_index,
                         chain_id_target='A', chain_id_source="B"):
    parser = PDB.PDBParser(QUIET=True)
    
    # Load both structures
    structure_a = parser.get_structure('A', pdb_target_path)
    structure_b = parser.get_structure('B', pdb_source_path)
    
    # Get the first model and specified chain from both structures
    model_a = structure_a[0]
    model_b = structure_b[0]
        
    chain_a = model_a[chain_id_target]
    chain_b = model_b[chain_id_source]
    
    # Create a mapping from one-based index to PDB index for replacement
    one_to_pdb_mapping = {}
    pdb_indices = list(range(start_pdb_index, end_pdb_index + 1))
    one_indices = list(range(start_one_index, end_one_index + 1))
    
    if len(pdb_indices) != len(one_indices):
        print(f"Warning: Range sizes don't match. PDB range: {len(pdb_indices)}, One-based range: {len(one_indices)}")
    
    # Create mapping
    for i, (pdb_idx, one_idx) in enumerate(zip(pdb_indices, one_indices)):
        one_to_pdb_mapping[one_idx] = pdb_idx
    
    
    # Remove residues in the specified range from chain A
    residues_to_remove = []
    for residue in chain_a:
        res_id = residue.get_id()
        res_num = res_id[1]  # residue number
        if start_pdb_index <= res_num <= end_pdb_index:
            residues_to_remove.append(res_id)
    
    for res_id in residues_to_remove:
        chain_a.detach_child(res_id)
    
    # Collect all remaining residues from chain A and new residues from chain B
    all_residues = []
    
    # Add remaining residues from chain A
    for residue in chain_a:
        all_residues.append(residue)
    
    # Prepare new residues from chain B with PDB numbering
    new_residues = []
    residues_added = 0
    for residue_b in chain_b:
        res_id_b = residue_b.get_id()
        res_num_b = res_id_b[1]  # residue number in one-based indexing
        
        # Check if this residue is in our replacement range
        if start_one_index <= res_num_b <= end_one_index:
            # Get the corresponding PDB index
            new_pdb_index = one_to_pdb_mapping[res_num_b]
            
            # Create a copy of the residue with new ID
            new_res_id = (res_id_b[0], new_pdb_index, res_id_b[2])  # (hetero_flag, res_num, insertion_code)
            
            # Create new residue with updated ID
            new_residue = PDB.Residue.Residue(new_res_id, residue_b.get_resname(), residue_b.get_segid())
            
            # Copy all atoms from the original residue
            for atom in residue_b:
                new_atom = atom.copy()
                new_residue.add(new_atom)
            
            new_residues.append(new_residue)
            residues_added += 1
    
    # Combine all residues and sort by residue number
    all_residues.extend(new_residues)
    all_residues.sort(key=lambda x: x.get_id()[1])
    
    # Clear chain A completely and rebuild it in correct order
    chain_a_residues = list(chain_a.child_dict.keys())
    for res_id in chain_a_residues:
        chain_a.detach_child(res_id)
    
    # Add all residues back in sorted order
    for residue in all_residues:
        chain_a.add(residue)
    
    # Write the merged structure
    io = PDBIO()
    io.set_structure(structure_a)
    io.save(output_path)
    return output_path

def split_refined_pdb_by_altloc(refined_pdb_path, chain, pdb_residue_range):
    samples = []
    # Temp dir for logs
    temp_dir_root = "refined_metrics_temp"
    os.makedirs(temp_dir_root, exist_ok=True)
    altlocs, occupancies = count_altlocs_in_region(refined_pdb_path, chain, pdb_residue_range[0], pdb_residue_range[1])

    for altloc in altlocs:
        altloc_output_file = save_refined_pdb_by_altloc(refined_pdb_path, chain, pdb_residue_range[0], pdb_residue_range[1], altloc, temp_dir_root, prefix="ensemble")
        samples.append(altloc_output_file)

    # Compute cosine similarity
    return samples, occupancies

def split_pdb_by_altloc(raw_pdb_path, chain, pdb_residue_range, one_indexed_residue_range):
    samples = []
    # TODO: Insert residue range into the raw pdb file
    raw_altlocs, raw_occupancies = count_altlocs_in_region(raw_pdb_path, chain, pdb_residue_range[0], pdb_residue_range[1])
    for raw_altloc in raw_altlocs:
        if raw_altloc == " ":
            raw_altloc = "A"
        raw_altloc_output_file = save_raw_pdb_by_altloc(raw_pdb_path, chain, pdb_residue_range, one_indexed_residue_range, raw_altloc)
        samples.append(raw_altloc_output_file)

    return samples, raw_occupancies

def get_refined_cosine_and_maps(refined_pdb_path, config, selection_method, relaxed_dir):
    pdb_residue_range = config.protein.pdb_residue_range
    one_index_residue_range = config.protein.residue_range
    raw_pdb_path = config.protein.reference_raw_pdb
    chain_id = config.protein.reference_raw_pdb_chain

    # Split by altlocs
    sample_paths, occupancies = split_refined_pdb_by_altloc(refined_pdb_path, chain_id, pdb_residue_range)
    pdb_sample_paths, pdb_occupancies = split_pdb_by_altloc(raw_pdb_path, chain_id, pdb_residue_range, one_index_residue_range)

    # Get cosine similarity
    rmax = 2.5

    # Log the density maps and the cosine similarity
    ensemble_cosine_similarity, pdb_cosine_similarity = get_refined_cosine_similarity(sample_paths, pdb_sample_paths, occupancies, pdb_occupancies, selection_method, rmax, relaxed_dir, pdb_residue_range, config, device=torch.device("cuda:0"))

    return ensemble_cosine_similarity, occupancies, pdb_cosine_similarity, pdb_occupancies
