import os
import json
import torch
import numpy as np
import shutil
from typing import List, Dict, Tuple, Optional

from .density_occ import OCCMetric
from .density_omp import OMPMetric
from .refined_merged_metrics import get_refined_cosine_and_maps

class DensityMetricsProcessor:
    """
    A class to handle density metrics processing using both OMP and Occupancy Optimization methods.
    """
    def __init__(
        self,
        config,
        relaxed_dir: str,
        device: str,
        ccp4_setup_sh: Optional[str] = None,
        phenix_setup_sh: Optional[str] = None,
    ):
        self.config = config
        self.relaxed_dir = relaxed_dir
        self.device = device
        self.ccp4_setup_sh = ccp4_setup_sh
        self.phenix_setup_sh = phenix_setup_sh
        self.rmax = 2.5
        self.pdb_cosine_similarity = None
        self.pdb_occupancy = None
        self.map_type = self.config.loss_function.density_loss_function.map_type

    def _create_metrics_dict(
        self,
        file_names: List[str],
        refined_file: str,
        ensemble_cosine_similarity: float,
        pdb_cosine_similarity: float,
        ensemble_rscc: List[float],
        pdb_rscc: List[float],
        r_work: float,
        r_free: float,
        pdb_r_work: float,
        pdb_r_free: float,
        ensemble_occupancy: List[float],
        pdb_occupancy: List[float]
    ) -> Dict:
        """Create a standardized metrics dictionary."""
        sequence = self.config.protein.sequences[0]["sequence"]
        return {
            "pdb": self.config.protein.pdb_id,
            "residue range start": self.config.protein.residue_range[0][0],
            "residue range end": self.config.protein.residue_range[0][1],
            "pdb residue range start": self.config.protein.pdb_residue_range[0][0],
            "pdb residue range end": self.config.protein.pdb_residue_range[0][1],
            "chain": self.config.protein.reference_raw_pdb_chain,
            "file names": file_names,
            "selected ensemble size": len(file_names),
            "ensemble cosine similarity": ensemble_cosine_similarity,
            "pdb cosine similarity": pdb_cosine_similarity,
            "ensemble R-work": r_work,
            "ensemble R-free": r_free,
            "pdb R-work": pdb_r_work,
            "pdb R-free": pdb_r_free,
            "ensemble rscc": ensemble_rscc,
            "ensemble rscc means": [np.mean(rscc_arr) for rscc_arr in ensemble_rscc],
            "pdb rscc": pdb_rscc,
            "pdb rscc means": [np.mean(rscc_arr) for rscc_arr in pdb_rscc],
            "ensemble occupancy": ensemble_occupancy,
            "pdb occupancy": pdb_occupancy,
            "subseq": sequence[self.config.protein.residue_range[0][0]-1:self.config.protein.residue_range[0][1]]
        }

    def _save_metrics(self, metrics: Dict, method: str):
        """Save metrics to a JSON file."""
        filename = f"{self.config.protein.pdb_id}{self.config.protein.reference_raw_pdb_chain}_{self.config.protein.pdb_residue_range[0][0]}_{self.config.protein.pdb_residue_range[0][1]}_{method}_{self.map_type}_guided_metrics.json"
        output_path = os.path.join(self.relaxed_dir, method, filename)
        with open(output_path, "w") as file:
            json.dump(metrics, file, indent=4)

    def _copy_raw_pdb(self, method: str):
        """Copy PDB file from pdb-redo to the output directory."""
        relaxed_folder_path = os.path.dirname(os.path.join(self.relaxed_dir, method))
        shutil.copy(
            self.config.protein.reference_raw_pdb,
            os.path.join(relaxed_folder_path, method, f"{self.config.protein.pdb_id}_raw.pdb")
        )

    def process_omp_metrics(self) -> Dict:
        """Process metrics using the OMP algorithm."""
        torch.cuda.empty_cache()
        
        omp_calculator = OMPMetric(
            samples_directory=self.relaxed_dir,
            rmax=self.rmax,
            reference_density_file=self.config.loss_function.density_loss_function.density_file,
            residue_range=self.config.protein.residue_range[0],
            altloc_a_path=self.config.loss_function.density_loss_function.reference_pdbs[0],
            altloc_b_path=self.config.loss_function.density_loss_function.reference_pdbs[1] 
                if len(self.config.loss_function.density_loss_function.reference_pdbs) > 1 else None,
            bond_max_threshold=self.config.loss_function.density_loss_function.bond_max_threshold,
            device=self.device,
            raw_pdb_file_path=self.config.protein.reference_raw_pdb,
            chain_id=self.config.protein.reference_raw_pdb_chain,
            mtz_file_path=self.config.loss_function.density_loss_function.mtz_file,
            reference_pdb_file_path=self.config.loss_function.density_loss_function.reference_pdbs[0],
            pdb_id=self.config.protein.pdb_id,
            pdb_residue_range=self.config.protein.pdb_residue_range[0],
            ccp4_setup_sh=self.ccp4_setup_sh,
            phenix_setup_sh=self.phenix_setup_sh,
            map_type=self.map_type
        )
        unguided = False if self.map_type is not None else True
        metrics_dict = omp_calculator.run(unguided)

        ensemble_cosine_similarity, ensemble_occupancies, pdb_cosine_similarity, pdb_occupancies = get_refined_cosine_and_maps(
            metrics_dict["refined file"], 
            self.config, 
            "omp", 
            self.relaxed_dir
        )
        self.pdb_cosine_similarity = pdb_cosine_similarity
        self.pdb_occupancy = pdb_occupancies

        metrics = self._create_metrics_dict(
            metrics_dict["pdb file names"],
            metrics_dict["refined file"],
            ensemble_cosine_similarity,
            pdb_cosine_similarity,
            metrics_dict["ensemble rscc"],
            metrics_dict["pdb rscc"],
            metrics_dict["r_free_r_work refined"][0],
            metrics_dict["r_free_r_work refined"][1],
            metrics_dict["pdb R-work refined"],
            metrics_dict["pdb R-free refined"],
            ensemble_occupancies,
            pdb_occupancies
        )

        self._save_metrics(metrics, "omp")
        self._copy_raw_pdb("omp")
        return metrics

    def process_occupancy_metrics(self) -> Dict:
        """Process metrics using the Occupancy Optimization algorithm."""
        torch.cuda.empty_cache()

        occ_calculator = OCCMetric(
            reference_pdbs=self.config.loss_function.density_loss_function.reference_pdbs,
            raw_pdb_file_path=self.config.protein.reference_raw_pdb,
            chain_id=self.config.protein.reference_raw_pdb_chain,
            pdb_folder=os.path.dirname(os.path.join(self.relaxed_dir, "occupancy_optim")),
            density_file=self.config.loss_function.density_loss_function.density_file,
            residue_range=self.config.protein.residue_range[0],
            pdb_id=self.config.protein.pdb_id,
            device=self.device,
            regularization_weight=self.config.loss_function.density_loss_function.occ_lambda,
            rmax=self.rmax,
            mtz_file=self.config.loss_function.density_loss_function.mtz_file,
            pdb_residue_range=self.config.protein.pdb_residue_range[0],
            phenix_setup_sh=self.phenix_setup_sh,
            ccp4_setup_sh=self.ccp4_setup_sh
        )
        occ_metrics_dict = occ_calculator.run(unguided=False if self.map_type is not None else True, map_type=self.map_type)

        ensemble_cosine_similarity, ensemble_occupancies, _, _ = get_refined_cosine_and_maps(
            occ_metrics_dict["refined file"], 
            self.config, 
            "occupancy_optim", 
            self.relaxed_dir
        )

        metrics_dict = self._create_metrics_dict(
            occ_metrics_dict["pdb file names"],
            occ_metrics_dict["refined file"],
            ensemble_cosine_similarity,
            self.pdb_cosine_similarity,
            occ_metrics_dict["ensemble rscc"],
            occ_metrics_dict["pdb rscc"],
            occ_metrics_dict["r_work_r_free_refined"][0],
            occ_metrics_dict["r_work_r_free_refined"][1],
            occ_metrics_dict["reference pdb r_work_r_free_refined"][0],
            occ_metrics_dict["reference pdb r_work_r_free_refined"][1],
            ensemble_occupancies,
            self.pdb_occupancy
        )

        self._save_metrics(metrics_dict, "occupancy_optim")
        self._copy_raw_pdb("occupancy_optim")
        return metrics_dict

    def process_all_metrics(self) -> Tuple[Dict, Dict]:
        """Process both OMP and Occupancy Optimization metrics."""
        omp_metrics = self.process_omp_metrics()
        occ_metrics = self.process_occupancy_metrics()
        return omp_metrics, occ_metrics
