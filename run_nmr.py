import argparse
import glob
import os

from experiment_manager import ExperimentManager
from src.utils.io import load_config, write_multi_model_pdb
from src.utils.process_pipeline_inputs.preprocess_nmr_inputs import main as preprocess_nmr_inputs
from src.utils.process_pipeline_inputs.preprocess_nmr_inputs import (
    main_from_custom_inputs as preprocess_custom_nmr_inputs,
)
from src.metrics.nmr_metrics import run_nmr_metrics

RELAXED_SUFFIX = "_colab_relaxed.pdb"


def relaxed_files_in_model_order(verbose_directory):
    """
    Relaxed ensemble members, ordered by their diffusion sample index.

    Filenames look like "{identifier}_{i}_hyd_added_colab_relaxed.pdb". The identifier
    may itself contain underscores, so the index is taken as the last underscore-
    separated token before the generated suffixes.
    """
    files = glob.glob(os.path.join(verbose_directory, f"*{RELAXED_SUFFIX}"))

    def model_index(path):
        stem = os.path.basename(path)
        for suffix in ("_hyd_added" + RELAXED_SUFFIX, RELAXED_SUFFIX):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        try:
            return (0, int(stem.rsplit("_", 1)[-1]))
        except (ValueError, IndexError):
            # Unparseable name: keep it, but sort it after the well-formed ones.
            return (1, 0)

    return sorted(files, key=lambda p: (model_index(p), p))


def main():
    parser = argparse.ArgumentParser(
        description="NMR-guided structure ensemble generation.",
        epilog=(
            "Two input modes:\n"
            "  (1) deposited entry:  run_nmr.py 1u0p\n"
            "  (2) custom restraints: run_nmr.py --conformation_id my_conf_A "
            "--sequence GYIP... --restraints my_restraints.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'pdb_id',
        type=str,
        nargs='?',
        default=None,
        help="PDB ID to fetch restraints, coordinates and sequence for. Omit when using --conformation_id.",
    )
    parser.add_argument(
        '--conformation_id',
        type=str,
        default=None,
        help="Identifier for a custom run, used for output naming instead of a PDB ID. Requires --sequence and --restraints.",
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help="One-letter amino acid sequence of the construct (custom mode).",
    )
    parser.add_argument(
        '--restraints',
        type=str,
        default=None,
        help="Path to restraints: a .csv in the documented schema, or a .str/.mr NMR-STAR file to convert (custom mode).",
    )
    parser.add_argument(
        '--reference_pdb',
        type=str,
        default=None,
        help="Optional reference structure (custom mode). When given, metrics include the MD comparison row.",
    )
    parser.add_argument('--input_directory', type=str, required=False, default="nmr_pipeline_inputs")
    parser.add_argument('--output_directory', type=str, required=False, default="nmr_pipeline_outputs")
    parser.add_argument('--wandb_key', type=str, required=False, default=None)
    parser.add_argument('--wandb_project', type=str, required=False, default=None)
    parser.add_argument('--methyl_rdc_file', type=str, required=False, default=None)
    parser.add_argument('--amide_rdc_file', type=str, required=False, default=None)
    parser.add_argument('--amide_relax_file', type=str, required=False, default=None)
    parser.add_argument('--methyl_relax_file', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default="cuda:0")
    parser.add_argument(
        '--run-receipts',
        action='store_true',
        default=False,
        help='Write SHA-256 chained run receipts for post-hoc verification',
    )
    parser.add_argument(
        '--deterministic-mode',
        action='store_true',
        default=False,
        help='Stricter deterministic path + run receipts (seed_experiment still always runs)',
    )
    args = parser.parse_args()

    custom_flags = {
        '--conformation_id': args.conformation_id,
        '--sequence': args.sequence,
        '--restraints': args.restraints,
    }
    supplied = {flag: value for flag, value in custom_flags.items() if value is not None}

    if args.pdb_id is not None and supplied:
        parser.error(
            f"Cannot combine the pdb_id positional argument with {', '.join(sorted(supplied))}. "
            f"Use either a deposited PDB ID or a full custom specification, not both."
        )
    if args.pdb_id is None and not supplied:
        parser.error(
            "Nothing to run. Supply either a pdb_id positional argument, or all of "
            "--conformation_id, --sequence and --restraints."
        )
    if args.pdb_id is None:
        missing = sorted(set(custom_flags) - set(supplied))
        if missing:
            parser.error(
                f"Custom mode needs all of --conformation_id, --sequence and --restraints; "
                f"missing: {', '.join(missing)}."
            )
    if args.pdb_id is not None and args.reference_pdb is not None:
        parser.error(
            "--reference_pdb only applies to custom mode; in PDB ID mode the deposited "
            "structure is fetched automatically."
        )

    # Prepare config file
    if args.pdb_id is not None:
        config_file_path = preprocess_nmr_inputs(
            args.pdb_id, args.input_directory, args.output_directory, args.wandb_key,
            args.wandb_project, args.methyl_rdc_file, args.amide_rdc_file,
            args.amide_relax_file, args.methyl_relax_file,
        )
    else:
        config_file_path = preprocess_custom_nmr_inputs(
            args.conformation_id, args.sequence, args.restraints, args.input_directory,
            args.output_directory, args.wandb_key, args.wandb_project,
            reference_pdb=args.reference_pdb,
            methyl_rdc_file=args.methyl_rdc_file, amide_rdc_file=args.amide_rdc_file,
            amide_relax_file=args.amide_relax_file, methyl_relax_file=args.methyl_relax_file,
        )

    # Loading the config file and merging it with the arguments
    config = load_config(config_file_path)

    # Seeding the experiment (unchanged baseline reproducibility)
    ExperimentManager.seed_experiment(config.general.seed)

    # Running the experiment
    pipeline = ExperimentManager(
        config,
        args.device,
        run_receipts=args.run_receipts or args.deterministic_mode,
        deterministic_mode=args.deterministic_mode,
    )
    pipeline.run()

    # Metrics!
    identifier = config.protein.pdb_id
    output_directory = os.path.join(config.general.output_folder, config.general.name)
    diffusion_directory = os.path.join(output_directory, "diffusion_process")
    metrics_results_path = os.path.join(diffusion_directory, f"{identifier}_metrics.csv")
    run_nmr_metrics(
        pdb_output_folder=output_directory,
        md_file=config.protein.reference_pdb,
        restraint_file=config.loss_function.nmr_loss_function.reference_nmr,
        add_hydrogen=True,
        relax_colabfold=True,
        results_path=metrics_results_path,
        additional_protein_files=None,
        order_params_files=None,
        noe=True,
        order_params=False,
        pdb_id=identifier,
    )

    # Collapse the relaxed ensemble into a single multi-model PDB. Per-structure files
    # and their hydrogenated/relaxed derivatives stay under diffusion_process/verbose/.
    verbose_directory = os.path.join(diffusion_directory, "verbose")
    relaxed_files = relaxed_files_in_model_order(verbose_directory)
    if relaxed_files:
        ensemble_path = write_multi_model_pdb(
            relaxed_files, os.path.join(diffusion_directory, f"{identifier}_ensemble.pdb")
        )
        print(f"Wrote {len(relaxed_files)}-model ensemble -> {ensemble_path}")
    else:
        print(
            f"WARNING: no relaxed structures found in {verbose_directory}; "
            f"skipped writing the merged ensemble."
        )
    print(f"Metrics -> {metrics_results_path}")


if __name__ == "__main__":
    main()

# Metrics order parameter file format
#         "amide_relax": "pipeline_inputs/nmr_s_2/ubi_solution_S2.csv",
#         "amide_rdc": "pipeline_inputs/nmr_s_2/ubi_nh_rdc.csv",
#         "methyl_relax": "pipeline_inputs/nmr_s_2/ubi_methyl_relaxation.csv",
#         "methyl_rdc": "pipeline_inputs/nmr_s_2/ubi_methyl_rdc.csv"}
