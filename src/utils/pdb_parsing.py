# Dictionary of bonds for all standard amino acids
AMINO_ACID_BONDS = {
    "ALA": {  # Alanine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": []
    },
    "ARG": {  # Arginine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")]
    },
    "ASN": {  # Asparagine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")]
    },
    "ASP": {  # Aspartic acid
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")]
    },
    "CYS": {  # Cysteine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "SG")]
    },
    "GLN": {  # Glutamine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")]
    },
    "GLU": {  # Glutamic acid
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")]
    },
    "GLY": {  # Glycine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [],
        "side_chain": []
    },
    "HIS": {  # Histidine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2")]
    },
    "ILE": {  # Isoleucine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")]
    },
    "LEU": {  # Leucine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")]
    },
    "LYS": {  # Lysine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")]
    },
    "MET": {  # Methionine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "SD"), ("SD", "CE")]
    },
    "PHE": {  # Phenylalanine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")]
    },
    "PRO": {  # Proline
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB"), ("N", "CD")],
        "side_chain": [("CB", "CG"), ("CG", "CD")]
    },
    "SER": {  # Serine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "OG")]
    },
    "THR": {  # Threonine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "OG1"), ("CB", "CG2")]
    },
    "TRP": {  # Tryptophan
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"), ("CD2", "CE3"),
                       ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2")]
    },
    "TYR": {  # Tyrosine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"),
                       ("CE2", "CZ"), ("CZ", "OH")]
    },
    "VAL": {  # Valine
        "backbone": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "backbone_to_side_chain": [("CA", "CB")],
        "side_chain": [("CB", "CG1"), ("CB", "CG2")]
    }
}

# Define peptide bonds between residues
RESIDUE_BONDS = [("C", "N")]

def find_bonded_pairs(chain, residue_range):
    bonded_pairs = []
    residues = list(chain)
    
    for i, residue in enumerate(residues):
        res_name = residue.name

        # Check for broken bonds if broken bonds are in residue range
        resid = residue.seqid.num
        if resid < residue_range[0] - 2 or resid > residue_range[1] + 2:
            continue

        if res_name not in AMINO_ACID_BONDS:
            continue

        bonds = AMINO_ACID_BONDS[res_name]
        
        # Add backbone, backbone-to-side-chain, and side-chain bonds
        for atom1, atom2 in bonds["backbone"] + bonds["backbone_to_side_chain"] + bonds["side_chain"]:
            if atom1 in residue and atom2 in residue:
                bonded_pairs.append((residue[atom1], residue[atom2]))
        
        # Add peptide bonds between residues
        if i < len(residues) - 1:
            next_residue = residues[i + 1]
            for atom1, atom2 in RESIDUE_BONDS:
                if atom1 in residue and atom2 in next_residue:
                    bonded_pairs.append((residue[atom1], next_residue[atom2]))
    
    return bonded_pairs

# if __name__ == "__main__":
#     import gemmi
#     chain = gemmi.read_pdb("/nfs/scistore20/bronsgrp/nsellam/proteinx_guidance/temp.pdb")[0][0]
#     bonds = find_bonded_pairs(chain)
#     a = 2

