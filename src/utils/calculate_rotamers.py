import json
import numpy as np
import pandas as pd
import biotite.structure as struc


class CalculateRotamers:
    def __init__(self, rotamer_angles_table="src/metrics/rotamer_angles_table.csv", angle_tolerance=30):
        self.rotamer_angles_table = rotamer_angles_table
        self.angle_tolerance = angle_tolerance

    # ==============================
    # Chi1 Calculation
    # ==============================

    def compute_chi1(self, atom_array, structure, residue_index):
        res_name = atom_array.res_name[residue_index]
        res_id = atom_array.res_id[residue_index]

        atom_indices = self._get_chi1_atom_indices(atom_array, res_id, res_name)
        if atom_indices is None:
            return None

        coords = structure[:, atom_indices].squeeze(2).cpu()
        chi1 = struc.dihedral(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
        return np.rad2deg(chi1)

    def _get_chi1_atom_indices(self, atom_array, res_id, res_name):
        n = np.where((atom_array.res_id == res_id) & (atom_array.atom_name == "N"))[0]
        ca = np.where((atom_array.res_id == res_id) & (atom_array.atom_name == "CA"))[0]
        cb = np.where((atom_array.res_id == res_id) & (atom_array.atom_name == "CB"))[0]

        if res_name in ["VAL", "ILE"]:
            cg1 = np.where((atom_array.res_id == res_id) & (atom_array.atom_name == "CG1"))[0]
            if n.size and ca.size and cb.size and cg1.size:
                return [n[0], ca[0], cb[0], cg1[0]]
        elif res_name == "THR":
            og1 = np.where((atom_array.res_id == res_id) & (atom_array.atom_name == "OG1"))[0]
            if n.size and ca.size and cb.size and og1.size:
                return [n[0], ca[0], cb[0], og1[0]]

        return None  # Missing required atoms or unsupported residue

    # ==============================
    # Rotamer Classification
    # ==============================

    def classify_rotamer(self, angle):
        if abs(angle - 180) <= self.angle_tolerance or abs(angle + 180) <= self.angle_tolerance:
            return "180"
        elif abs(angle + 60) <= self.angle_tolerance:
            return "-60"
        elif abs(angle - 60) <= self.angle_tolerance:
            return "+60"
        else:
            return "Other"

    def label_chi1_angles(self, angles, res_name, res_id):
        return [
            {
                "res_name": res_name,
                "res_id": res_id,
                "Chi1_Angle": angle,
                "Rotamer": self.classify_rotamer(angle)
            }
            for angle in angles
        ]

    # ==============================
    # Rotamer Analysis Pipeline
    # ==============================

    def analyze_chi1_rotamers(self, atom_array, structures):
        residue_df = pd.read_csv(self.rotamer_angles_table)
        results = []

        for _, row in residue_df.iterrows():
            res_id, res_name = row["res_id"], row["res_name"]
            residue_index = self._find_residue_index(atom_array, res_id, res_name)
            if residue_index is None:
                print(f"Warning: Residue {res_name} {res_id} not found. Skipping.")
                continue

            chi1 = self.compute_chi1(atom_array, structures, residue_index)
            if chi1 is not None:
                results.extend(self.label_chi1_angles(chi1, res_name, res_id))

        df = pd.DataFrame(results)
        df["Chi1_Angle"] = df["Chi1_Angle"].apply(lambda x: x + 360 if x < 0 else x)
        return self._summarize_rotamers(df)

    def _find_residue_index(self, atom_array, res_id, res_name):
        matches = np.where((atom_array.res_id == res_id) & (atom_array.res_name == res_name))[0]
        return matches[0] if matches.size > 0 else None

    # ==============================
    # Rotamer Summary Statistics
    # ==============================

    def _summarize_rotamers(self, df):
        summary = df.groupby(["res_name", "res_id"]).agg({
            "Rotamer": list,
            "Chi1_Angle": list
        }).reset_index()

        for rotamer in ["180", "-60", "+60", "Other"]:
            summary[f"x_1_pop_{rotamer.replace('+', '')}"] = self._calculate_population(summary, rotamer)
            summary[f"x_1_pop_{rotamer.replace('+', '')}_std"] = self._calculate_population_std(summary, rotamer)
            mean_col, std_col = f"x_1_{rotamer.replace('+', '')}_mean", f"x_1_{rotamer.replace('+', '')}_std"
            summary[mean_col], summary[std_col] = zip(*summary.apply(
                lambda row: self._calculate_mean_std(df, row, rotamer), axis=1))

        return summary.drop(columns=["Rotamer", "Chi1_Angle"])

    def _calculate_population(self, summary, rotamer):
        return summary["Rotamer"].apply(lambda lst: lst.count(rotamer) / len(lst) if lst else 0)

    def _calculate_population_std(self, summary, rotamer):
        return summary.apply(
            lambda row: np.std([
                row["Chi1_Angle"][i]
                for i in range(len(row["Rotamer"]))
                if row["Rotamer"][i] == rotamer
            ]) if rotamer in row["Rotamer"] else 0,
            axis=1
        )

    def _calculate_mean_std(self, df, row, rotamer_type):
        angles = df[
            (df["res_name"] == row["res_name"]) &
            (df["res_id"] == row["res_id"]) &
            (df["Rotamer"] == rotamer_type)
        ]["Chi1_Angle"]
        return (angles.mean(), angles.std()) if not angles.empty else (np.nan, np.nan)

    # ==============================
    # Public Interface
    # ==============================

    def run(self, atom_arrays, structures):        
        summary_df = self.analyze_chi1_rotamers(atom_arrays[0], structures)
        summary_df = summary_df.set_index("res_id").to_dict(orient="index")
        return {"rotamers": json.dumps(summary_df)}
