GLUTAMINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon and its substituents (including backbone C′ and side-chain Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # This would be the hydroxyl in COOH
    
    # Side chain: Cβ → Cγ → Cδ → (Oε1, Nε2)
    "CB":   ["CA", "HB2", "HB2", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "CD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "CG":   ["CG", "OE1", "NE2"],
    "OE1":  ["CG"],

    # Amide group on the side chain
    "NE2":  ["CG", "HE21", "HE22"],
    "HE21": ["NE2"],
    "HE22": ["NE2"],
}

ALANINE_GRAPH = {
    "N":    ["H", "HN", "CA"],
    "H":  ["N"],
    "HN":  ["N"],

    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],

    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # This would be the hydroxyl in COOH

    "CB":   ["CA", "MB"],
    "MB": ["CB"]
}

GLYCINE_GRAPH = {
    "N":    ["H", "H1", "CA"],
    "H":  ["N"],
    "H1":  ["N"],

    "CA":   ["N", "HA2", "HA3", "C"],
    "HA2":   ["CA"],
    "HA3":   ["CA"],

    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # This would be the hydroxyl in COOH
}

ISOLEUCINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "HN", "CA"],
    "H":  ["N"],
    "H1": ["N"],
    "HN":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and the side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) with its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # The hydroxyl in COOH

    # Side chain: Cβ → Cγ → Cδ → Cε → Nζ
    # Each methylene (CH2) group has two hydrogens
    "CB":   ["CA", "HB", "CG1", "CG2"],
    "HB":  ["CB"],
    
    "CG1":   ["CD1", "HG11", "HG12"],
    "HG11":  ["CG1"],
    "HG12":  ["CG1"],

    "CD1":   ["CG1", "MD", "HG11", "HG12", "HG13"],
    "MD":  ["CD1"],
    "HG13":  ["CD1"],
    "HG12":  ["CD1"],
    "HG11":  ["CD1"],
    
    "CG2":   ["CB", "MG", "HG21", "HG22", "HG23"],
    "MG":  ["CG2"],
    "HG21":  ["CG2"],
    "HG22":  ["CG2"],
    "HG23":  ["CG2"],
}

PROLINE_GRAPH = {
    # Backbone nitrogen (N) has only one hydrogen in Proline,
    # because the side chain closes the ring with N.
    "N":    ["H", "CA", "CG"],
    "H":  ["N"],

    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],

    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # This would be the hydroxyl in COOH

    # Side chain forming a ring: Cβ → Cγ → Cδ → back to N
    "CB":   ["CA", "CG", "HB2", "HB3"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "CD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "CD":   ["CG", "HD2", "HD3"],
    "HD2":  ["CD"],
    "HD3":  ["CD"],
}

SERINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH
    
    # Side chain: Cβ (with two hydrogens) and a hydroxyl (Oγ)
    "CB":   ["CA", "HB2", "HB3", "OG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "OG":   ["CB", "HG"],
    "HG":   ["OG"],
}

VALINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ (connected to Cγ1 and Cγ2)
    "CB":   ["CA", "HB", "CG1", "CG2"],
    "HB":   ["CB"],

    # Cγ1 (isopropyl group) and its hydrogens
    "CG1":  ["CB", "MG1", "HG11", "HG12", "HG13"],
    "MG1": ["CG1"],
    "HG11": ["CG1"],
    "HG12": ["CG1"],
    "HG13": ["CG1"],

    # Cγ2 (isopropyl group) and its hydrogens
    "CG2":  ["CB", "MG2", "HG21", "HG22", "HG23"],
    "MG2": ["CG2"],
    "HG21": ["CG2"],
    "HG22": ["CG2"],
    "HG23": ["CG2"],
}

THREONINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ (with a hydroxyl group Oγ1 and a Cγ2 group)
    "CB":   ["CA", "HB", "OG", "CG2"],
    "HB":   ["CB"],

    "OG":  ["CB", "HG1"],
    "HG1":  ["OG"],

    # Cγ2 group (methyl) with its three hydrogens
    "CG2":  ["CB", "MG", "HG21", "HG22", "HG23"],
    "MG":  ["CG2"],
    "HG21":  ["CG2"],
    "HG22":  ["CG2"],
    "HG23":  ["CG2"],
}

TYROSINE_GRAPH = {
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ (connected to Cγ)
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "CD1", "CD2"],
    
    "CD1":  ["CG", "HD1", "CE1"],
    "HD1":  ["CD1"],
    "QD":  ["CD1"],
    "CE1":  ["CD1", "HE1", "CZ"],
    "HE1":  ["CE1"],
    "QE": ["CE1"],

    "CD2":  ["CG", "HD2", "CE2"],
    "HD2":  ["CD2"],
    "CE2":  ["CD2", "HE2", "CZ"],
    "HE2":  ["CE2"],
    
    "CZ":   ["CE1", "CE2", "OH"],
    "OH":   ["CZ"],
    "HH":   ["OH"],
}

CYSTEINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H1":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ connected to Sγ (sulfur group)
    "CB":   ["CA", "HB2", "HB3", "SG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    # Sulfur atom and its bonded hydrogen
    "SG":   ["CB", "HG"],
    "HG":   ["SG"],
}

ARGININE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    # Side chain: Cβ → Cγ → Cδ → Cε → Guanidino group
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "CD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "CD":   ["CG", "HD2", "HD3", "NE"],
    "HD2":  ["CD"],
    "HD3":  ["CD"],
    
    "NE":   ["CD", "HE", "CZ"],
    "HE":  ["NE"],

    "CZ":   ["NE", "NH1", "NH2"],
    "NH1":  ["CZ", "HH11", "HH12"],
    "NH2":  ["CZ", "HH21", "HH22"],
    "HH11": ["NH1"],
    "HH12": ["NH1"],
    "HH21": ["NH2"],
    "HH22": ["NH2"],

}

ASPARAGINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    # Side chain: Cβ connected to Cγ (amide group)
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    # Amide group: Cγ connected to Oδ1 and Nδ2
    "CG":   ["CB", "OD1", "ND2"],
    "OD1":  ["CG"],

    # Amide nitrogen (Nδ2) and its hydrogens
    "ND2":  ["CG", "HD21", "HD22"],
    "HD21": ["ND2"],
    "HD22": ["ND2"],
}

ASPARTIC_ACID_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ connected to Cγ and two hydrogens
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    # Side chain Cγ connected to Oδ1 and Oδ2 (carboxylic acid group)
    "CG":   ["CB", "OD1", "OD2"],
    "OD1":  ["CG"],
    "OD2":  ["CG", "HD2"],  # Oδ2 is bonded to an H in the protonated form
    "HD2":  ["OD2"],
}

PHENYLALANINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ connected to Cγ and two hydrogens
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    # Side chain: Benzyl group
    "CG":   ["CB", "CD1", "CD2"],
    "CD1":  ["CG", "HD1", "CE1"],
    "HD1":  ["CD1"],
    "CE1":  ["CD1", "HE1", "CZ"],
    "HE1":  ["CE1"],
    "QD":   ["CG"],


    "CD2":  ["CG", "HD2", "CE2"],
    "HD2":  ["CD2"],
    "CE2":  ["CD2", "HE2", "CZ"],
    "HE2":  ["CE2"],
    "QE":   ["CE2"],

    "CZ":   ["CE1", "CE2", "OH"],
    "HZ":   ["CZ"],
}

HISTIDINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    # Side chain: Cβ connected to Cγ and two hydrogens
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    # Side chain: Cγ connected to the imidazole ring (Cδ1 and Nδ1)
    "CG":   ["CB", "CD2", "ND1"],
    "ND1":  ["CG", "HD1", "CE1"],
    "HD1":  ["ND1"],
    "CD2":  ["CG", "HD2", "NE2"],
    "HD2":  ["CD2"],
    "NE2":  ["CD2", "HE2", "CE1"],
    "HE2":  ["NE2"],
    "CE1":  ["NE2", "HE1"],
    "HE1":  ["CE1"],

}

LYSINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H1":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    # Side chain: Cβ → Cγ → Cδ → Cε → Nζ
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "CD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "CD":   ["CG", "HD2", "HD3", "NE"],
    "HD2":  ["CD"],
    "HD3":  ["CD"],

    "CE":   ["CD", "HE2", "HE3", "NZ"],
    "HE2":  ["CE"],
    "HE3":  ["CE"],

    "NZ":   ["CE", "HZ"],
    "HZ":  ["NZ"],
    "QZ":  ["HZ"],
}

METHIONINE_GRAPH = {
    # Backbone nitrogen and its hydrogens
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H1":  ["N"],
    "H2N":  ["N"],
    
    # Alpha carbon, its hydrogen, the backbone carboxyl carbon (C′), and side chain (Cβ)
    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    # Backbone carboxyl carbon (C′) and its oxygens
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],  # hydroxyl in COOH

    # Side chain: Cβ → Cγ → Sδ → Cε
    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "SD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "SD":   ["CG", "CE"],

    # Terminal methyl group (Cε) with three hydrogens
    "CE":   ["SD", "ME", "HE"],
    "ME":   ["CE"],
    "HE":   ["CE"],
}

GLUTAMIC_ACID_GRAPH = {
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],

    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],

    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "HG2", "HG3", "CD"],
    "HG2":  ["CG"],
    "HG3":  ["CG"],

    "CD":   ["CG", "OE1", "OE2"],
    "OE1":  ["CD"],
    "OE2":  ["CD", "HE2"],
    "HE2":  ["OE2"],
}

TRYPTOPHAN_GRAPH = {
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],

    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],

    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],

    "CG":   ["CB", "CD1", "CD2"],
    "HD1":  ["CD1"],
    
    "CD1": ["CG", "CE2", "HD1"],
    "NE1": ["CD1", "HE1", "CE2"],
    "HE1": ["NE1"],
    
    "CE2": ["CZ2", "CD2", "NE1"],
    "CZ2": ["CE2", "HZ2", "CH2"],
    "HZ2": ["CZ2"],
    
    "CH2": ["CZ2", "CZ3", "HH2"],
    "HH2": ["CH2"],
    
    "CZ3": ["CE3", "HE3", "CH2"],
    "HE3": ["CE3"],
    
    "CE3": ["CD2", "HE3", "CZ3"],
    "HZ3": ["CZ3"],
    
    "CD2": ["CG", "CE3", "CE2"]
}

LEUCINE_GRAPH = {
    "N":    ["H", "H2N", "CA"],
    "H":  ["N"],
    "H2N":  ["N"],

    "CA":   ["N", "HA", "C", "CB"],
    "HA":   ["CA"],
    
    "C":   ["CA", "O", "OH"],
    "O":    ["C"],
    "OH":   ["C"],

    "CB":   ["CA", "HB2", "HB3", "CG"],
    "HB2":  ["CB"],
    "HB3":  ["CB"],
    
    "CG": ["CD1", "CD2", "CB", "HG"],
    "HG": ["CG"],
    "CD1": ["CG", "MD1", "HG11", "HG12", "HG13"],
    "MD1": ["CD1"],
    "HG11": ["CD1"],
    "HG12": ["CD1"],
    "HG13": ["CD1"],

    "CD2": ["CG", "MD2", "HG21", "HG22", "HG23"],
    "MD2": ["CD2"],
    "HG21": ["CD2"],
    "HG22": ["CD2"],
    "HG23": ["CD2"],
}

THREE_AAS_GRAPHS = {
    "GLU": GLUTAMINE_GRAPH,
    "GLY": GLYCINE_GRAPH,
    "ILE": ISOLEUCINE_GRAPH,
    "PRO": PROLINE_GRAPH,
    "SER": SERINE_GRAPH,
    "VAL": VALINE_GRAPH,
    "THR": THREONINE_GRAPH,
    "TYR": TYROSINE_GRAPH,
    "CYS": CYSTEINE_GRAPH,
    "ARG": ARGININE_GRAPH,
    "ASN": ASPARAGINE_GRAPH,
    "ASP": ASPARTIC_ACID_GRAPH,
    "PHE": PHENYLALANINE_GRAPH,
    "HIS": HISTIDINE_GRAPH,
    "LYS": LYSINE_GRAPH,
    "MET": METHIONINE_GRAPH,
    "TRP": TRYPTOPHAN_GRAPH,
    "LEU": LEUCINE_GRAPH,
    "GLN": GLUTAMINE_GRAPH,
    "ALA": ALANINE_GRAPH,
}


# if __name__ == "__main__":
#     print(len(THREE_AAS_GRAPHS))
