from alphafold.relax import relax
from alphafold.common import protein, residue_constants
import gemmi

MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

def pdb_to_string(pdb_file, chains=None, models=[1]):
  '''read pdb file and return as string'''

  if chains is not None:
    if "," in chains: chains = chains.split(",")
    if not isinstance(chains,list): chains = [chains]
  if models is not None:
    if not isinstance(models,list): models = [models]

  modres = {**MODRES}
  lines = []
  seen = []
  model = 1
  for line in open(pdb_file,"rb"):
    line = line.decode("utf-8","ignore").rstrip()
    if line[:5] == "MODEL":
      model = int(line[5:])
    if models is None or model in models:
      if line[:6] == "MODRES":
        k = line[12:15]
        v = line[24:27]
        if k not in modres and v in residue_constants.restype_3to1:
          modres[k] = v
      if line[:6] == "HETATM":
        k = line[17:20]
        if k in modres:
          line = "ATOM  "+line[6:17]+modres[k]+line[20:]
      if line[:4] == "ATOM":
        chain = line[21:22]
        if chains is None or chain in chains:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          if resn[-1].isalpha(): # alternative atom
            resn = resn[:-1]
            line = line[:26]+" "+line[27:]
          key = f"{model}_{chain}_{resn}_{resi}_{atom}"
          if key not in seen: # skip alternative placements
            lines.append(line)
            seen.append(key)
      if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
        lines.append(line)
  return "\n".join(lines)

def reorder_pdb_atoms(input_pdb, relaxed_pdb):
    original = gemmi.read_structure(input_pdb)
    relaxed = gemmi.read_structure(relaxed_pdb)
    relaxed.remove_hydrogens()

    # Build atom order mapping
    atom_order = {
        res.seqid.num: [atom.name for atom in res]
        for model in original for chain in model for res in chain
    }

    # Reorder atoms in the relaxed structure
    for model in relaxed:
        for chain in model:
            for res in chain:
                key = res.seqid.num
                if key in atom_order:
                    res_atoms = {atom.name: atom.clone() for atom in res}
                    while len(res) > 0:
                       del res[0]
                    for atom_name in atom_order[key]:  # Add back atoms in correct order
                        res.add_atom(res_atoms[atom_name])

    # Save changes to the same file
    relaxed.write_pdb(relaxed_pdb)

def mse_to_met(input_pdb, output_pdb):
  st = gemmi.read_pdb(input_pdb)
  chain = st[0][0]
  for residue in chain:
      if residue.name == 'MSE':
          residue.name = 'MET'
          residue.het_flag = 'A'
          for atom in residue:
              if atom.name == 'SE':
                  atom.name = 'SD'
                  atom.element = gemmi.Element('S')
  st.write_pdb(output_pdb)

def fix_met_to_mse_based_on_reference(reference_pdb, target_pdb):
  reference_chain = gemmi.read_pdb(reference_pdb)[0][0]
  target_structure = gemmi.read_pdb(target_pdb)
  target_chain = target_structure[0][0]
  for source_residue, target_residue in zip(reference_chain, target_chain):
    if source_residue.name == "MSE":
        target_residue.name = "MSE"
        for atom in target_residue:
          target_residue.het_flag = "H"
          if atom.name == 'SD':
                atom.name = 'SE'
                atom.element = gemmi.Element('Se')
  target_structure.write_pdb(target_pdb)

def relax_pdb(pdb_in, pdb_out, max_iterations=2000, tolerance=2.39, stiffness=10.0, use_gpu=False, reorder_atoms=True):
    mse_to_met(pdb_in, pdb_out)
    pdb_str = pdb_to_string(pdb_out)
    protein_obj = protein.from_pdb_string(pdb_str)
    amber_relaxer = relax.AmberRelaxation(
      max_iterations=max_iterations,
      tolerance=tolerance,
      stiffness=stiffness,
      exclude_residues=[],
      max_outer_iterations=3,
      use_gpu=use_gpu
    )
    # try 3 times, sometimes it fails for the first try for some reason
    for _ in range(3):
      try:
        relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=protein_obj)
        break
      except Exception as e:
        print(e)
    with open(pdb_out, 'w') as f:
        f.write(relaxed_pdb_lines)
    fix_met_to_mse_based_on_reference(pdb_in, pdb_out)
    if reorder_atoms:
      reorder_pdb_atoms(pdb_in, pdb_out) # for some reason it changes the atom order in the pdb, and that can cause bugs
