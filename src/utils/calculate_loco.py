import json
from pathlib import Path
from loco_hd import *
import numpy as np
from Bio.PDB import StructureBuilder
from Bio.PDB.Atom import Atom as BioAtom
from collections import defaultdict


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"

    return PrimitiveAtom(
        prat.primitive_type, 
        source,  # this is the tag field!
        prat.coordinates
    )
    
def atomarray_to_biopython(atom_array):
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure("converted")
    structure_builder.init_model(0)
    
    # Group atoms by (chain_id, res_id)
    atoms_by_residue = defaultdict(list)
    for i in range(len(atom_array)):
        key = (
            atom_array.chain_id[i],
            atom_array.res_id[i],
            atom_array.res_name[i],
        )
        atoms_by_residue[key].append(i)
    
    for (chain_id, res_id, res_name), indices in atoms_by_residue.items():
        structure_builder.init_chain(chain_id)
        structure_builder.init_seg("    ")
        structure_builder.init_residue(res_name, " ", int(res_id), " ")
        
        for i in indices:
            coord = atom_array.coord[i]
            element = atom_array.element[i]
            name = atom_array.atom_name[i] if atom_array.atom_name[i] else element
            bio_atom = BioAtom(
                name=name,
                coord=coord,
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=" " + name.ljust(3),
                serial_number=i,
                element=element
            )
            structure_builder.structure[0][chain_id][(" ", int(res_id), " ")].add(bio_atom)

    return structure_builder.get_structure()

    
def biopython_struct(atom_arrays):
    structures = []
    for i in range(len(atom_arrays)):
        structures.append(atomarray_to_biopython(atom_arrays[i]))
    return structures

class CalculateLoco:
    # This is a score between 0 and 1, with larger values meaning greater dissimilarity.
    def __init__(self, primitive_path = "src/metrics/loco_primitive/all_atom_with_centroid.config.json"):
        self.primitive_path = primitive_path
        self.primitive_assigner = PrimitiveAssigner(Path(primitive_path))
        w_func = WeightFunction("uniform", [3., 10.])
        tag_pairing_rule = TagPairingRule({"accept_same": False})
        self.lchd = LoCoHD(
            self.primitive_assigner.all_primitive_types, 
            w_func,
            tag_pairing_rule,
            n_of_threads=4
        )
        
    def _make_anchor_pairs(self, templates):
        if "centroid" in self.primitive_path:
            return [
                (idx, idx)
                for idx, prat in enumerate(templates)
                if prat.primitive_type == "Cent"
            ]
        else:
            return [(idx, idx) for idx in range(len(templates))]
        
    def _compute_lchd_scores(self, primitives1, primitives2, anchor_pairs):
        scores = []
        for i, pra1 in enumerate(primitives1):
            for j, pra2 in enumerate(primitives2):
                if pra1 is pra2:
                    continue
                lchd = self.lchd.from_primitives(pra1, pra2, anchor_pairs, 10.0)
                scores.append(lchd)
        return np.array(scores)
    
        
    def run(self, atom_arrays, gt_atom_array):
        structures = biopython_struct(atom_arrays)
        gt_structures = biopython_struct(gt_atom_array)
        pra_templates = [self.primitive_assigner.assign_primitive_structure(s) for s in structures]
        gt_pra_templates = [self.primitive_assigner.assign_primitive_structure(s) for s in gt_structures]
        
        
        anchor_pairs = self._make_anchor_pairs(pra_templates[0])
        
        pra_primitives = [list(map(prat_to_pra, template)) for template in pra_templates]
        gt_primitives = [list(map(prat_to_pra, template)) for template in gt_pra_templates]
            
        self_scores = self._compute_lchd_scores(pra_primitives, pra_primitives, anchor_pairs)
        gt_scores = self._compute_lchd_scores(pra_primitives, gt_primitives, anchor_pairs)

        return {
            "self_loco_score_mean": json.dumps(self_scores.mean(axis=0).tolist()),
            "self_loco_score_std": json.dumps(self_scores.std(axis=0).tolist()),
            
            "gt_loco_score_mean": json.dumps(gt_scores.mean(axis=0).tolist()),
            "gt_loco_score_std": json.dumps(gt_scores.std(axis=0).tolist()),
        }
        
        