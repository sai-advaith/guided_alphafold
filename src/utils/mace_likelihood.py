import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from typing import Tuple, Optional


def sample_reference_embeddings(
    embeddings: torch.Tensor,
    n_samples: int = 1000,
    random_seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """Randomly sample reference embeddings from a dataset."""
    torch.manual_seed(random_seed)
    embeddings = embeddings.to(device)
    n_total = embeddings.shape[0]
    if n_samples >= n_total:
        return embeddings
    
    indices = torch.randperm(n_total, device=device)[:n_samples]
    return embeddings[indices]


def compute_likelihood(
    current_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    node_attrs: Optional[torch.Tensor] = None,
    element_kernel_sigmas: Optional[torch.Tensor] = None,
    typical_length_scale: float = 1.0,
    time: Optional[float] = None,
    kernel_width: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """Compute log likelihood of embeddings under the reference distribution."""
    # Move tensors to device
    current_embeddings = current_embeddings.to(device)
    reference_embeddings = reference_embeddings.to(device)
    
    # Create default attributes if not provided
    if node_attrs is None:
        node_attrs = torch.ones(current_embeddings.shape[0], 1, dtype=torch.float32, device=device)
    else:
        node_attrs = node_attrs.to(device)
    
    if element_kernel_sigmas is None:
        n_elements = node_attrs.shape[1] if len(node_attrs.shape) > 1 else 1
        element_kernel_sigmas = torch.ones(n_elements, dtype=torch.float32, device=device)
    else:
        element_kernel_sigmas = element_kernel_sigmas.to(device)
    
    # Calculate squared distances between current and reference embeddings
    # print(current_embeddings.shape, reference_embeddings.shape)
    # import ipdb; ipdb.set_trace()
    squared_distance_matrix = torch.cdist(current_embeddings, reference_embeddings, p=2)**2
    # embedding_deltas = current_embeddings[:, None, :] - reference_embeddings[None, :, :]
    # squared_distance_matrix = torch.sum(embedding_deltas**2, dim=-1)
    
    # Normalize by typical length scale
    squared_distance_matrix = squared_distance_matrix / typical_length_scale
    
    # Apply element-specific scaling
    elements = node_attrs.detach().cpu().numpy()
    element_index = np.argmax(elements, axis=1)
    element_specific_sigmas = element_kernel_sigmas[element_index, None]
    squared_distance_matrix = squared_distance_matrix / element_specific_sigmas
    
    # Apply time-dependent scaling (if time is provided)
    if time is not None:
        additional_multiplier = 119 * (1 - (time / 10) ** 0.25) + 1 if time <= 10 else 1
        squared_distance_matrix = squared_distance_matrix * additional_multiplier
    
    # Compute log likelihood using logsumexp (kernel density estimation)
    log_likelihood = torch.logsumexp(-squared_distance_matrix / kernel_width, dim=1)
    
    return log_likelihood


class MACEbasedLikelihood:
    """
    A class to compute embedding likelihoods based on a reference set of embeddings
    using kernel density estimation.
    """
    
    def __init__(
        self,
        reference_embeddings: torch.Tensor,
        node_attrs: Optional[torch.Tensor] = None,
        element_kernel_sigmas: Optional[torch.Tensor] = None,
        typical_length_scale: float = 1.0,
        kernel_width: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the MACEbasedLikelihood model."""
        self.device = device
        
        # Move reference embeddings to device
        self.reference_embeddings = reference_embeddings.to(device)
        
        # Set default node_attrs if not provided
        if node_attrs is None:
            self.node_attrs = torch.ones(reference_embeddings.shape[0], 1, dtype=torch.float32, device=device)
        else:
            self.node_attrs = node_attrs.to(device)
        
        # Set default element_kernel_sigmas if not provided
        if element_kernel_sigmas is None:
            n_elements = self.node_attrs.shape[1] if len(self.node_attrs.shape) > 1 else 1
            self.element_kernel_sigmas = torch.ones(n_elements, dtype=torch.float32, device=device)
        else:
            self.element_kernel_sigmas = element_kernel_sigmas.to(device)
        
        self.typical_length_scale = typical_length_scale
        self.kernel_width = kernel_width
    
    def compute_likelihood(
        self,
        current_embeddings: torch.Tensor,
        node_attrs: Optional[torch.Tensor] = None,
        time: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute log likelihood of embeddings under the reference distribution."""
        return compute_likelihood(
            current_embeddings,
            self.reference_embeddings,
            node_attrs,
            self.element_kernel_sigmas,
            self.typical_length_scale,
            time,
            self.kernel_width,
            self.device
        )
    
    @classmethod
    def from_structure_type(
        cls,
        descriptors: dict,
        structure_type: str = 'H',
        max_reference_size: int = 10000,
        typical_length_scale: float = 1.0,
        kernel_width: float = 0.1,
        random_seed: int = 42,
        device = "cpu"
    ):
        """Create a model from a specific secondary structure type."""
        # Extract data from descriptors
        print("Creating likelihood model from structure type: ", structure_type)
        print(descriptors['descriptors'].shape)
        embeddings = torch.tensor(descriptors['descriptors'], dtype=torch.float32)
        metadata_df = descriptors['metadata']
        
        metadata_df = metadata_df.reset_index(drop=True)
        
        # Create node attributes (simplified - using ones for now)
        node_attrs = torch.ones(embeddings.shape[0], 1, dtype=torch.float32)
        
        # Filter indices by secondary structure
        torch.manual_seed(random_seed)
        struct_indices = torch.tensor(
            metadata_df[metadata_df['secondary_structure'] == structure_type].index.tolist(),
            dtype=torch.long,
        )
        
        # Limit reference size
        if len(struct_indices) > max_reference_size:
            perm = torch.randperm(len(struct_indices))
            reference_indices = struct_indices[perm[:max_reference_size]]
        else:
            reference_indices = struct_indices
        
        reference_embeddings = embeddings[reference_indices].to(device)
        reference_node_attrs = node_attrs[reference_indices].to(device)
        
        print(f"Created likelihood model with {len(reference_indices)} '{structure_type}' embeddings as reference")
        
        return cls(
            reference_embeddings,
            reference_node_attrs,
            None,  # Default element_kernel_sigmas
            typical_length_scale,
            kernel_width,
            device
        )
        
    @classmethod
    def from_amino_acid_type(
        cls,
        descriptors: dict,
        amino_acid_type: str = 'ALA',
        max_reference_size: int = 10000,
        typical_length_scale: float = 1.0,
        kernel_width: float = 0.1,
        random_seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Create a model from a specific amino acid type."""
        # Extract data from descriptors
        print("Creating likelihood model from amino acid type: ", amino_acid_type)
        print(descriptors['descriptors'].shape)
        embeddings = torch.tensor(descriptors['descriptors'], dtype=torch.float32)
        metadata_df = descriptors['metadata']
        
        metadata_df = metadata_df.reset_index(drop=True)
        
        # Create node attributes (simplified - using ones for now)
        node_attrs = torch.ones(embeddings.shape[0], 1, dtype=torch.float32)
        
        # Filter indices by amino acid type
        torch.manual_seed(random_seed)
        aa_indices = torch.tensor(
            metadata_df[metadata_df['name'] == amino_acid_type].index.tolist(),
            dtype=torch.long,
        )
        
        # Limit reference size
        if len(aa_indices) > max_reference_size:
            perm = torch.randperm(len(aa_indices))
            reference_indices = aa_indices[perm[:max_reference_size]]
        else:
            reference_indices = aa_indices
        
        reference_embeddings = embeddings[reference_indices].to(device)
        reference_node_attrs = node_attrs[reference_indices].to(device)
        
        print(f"Created likelihood model with {len(reference_indices)} '{amino_acid_type}' embeddings as reference")
        
        return cls(
            reference_embeddings,
            reference_node_attrs,
            None,  # Default element_kernel_sigmas
            typical_length_scale,
            kernel_width,
            device
        )


def plot_log_likelihoods(
    log_likelihoods: dict, 
    reference_structure: str = "H", 
    save_path: str = "loglikelihood_plot.png",
    remove_outliers: bool = True,
    outlier_threshold: float = 1.5
):
    """
    Plot log-likelihoods from multiple models as histograms.
    
    Parameters
    ----------
    log_likelihoods : dict
        Dictionary mapping structure type to list of arrays of log likelihoods
        [helix_model_likelihoods, sheet_model_likelihoods, turn_model_likelihoods]
    reference_structure : str
        The reference structure used for likelihood computation
    save_path : str
        Path to save the plot
    remove_outliers : bool
        Whether to remove outliers before plotting
    outlier_threshold : float
        Threshold for outlier removal (multiplier for IQR)
    """
    model_names = ['H', 'E', 'T']
    model_colors = {
        'H': 'lightcoral',
        'E': 'lightblue',
        'T': 'lightgreen'
    }
    
    # Create a subplot for each structure type
    num_structures = len(log_likelihoods)
    fig, axes = plt.subplots(1, num_structures, figsize=(5*num_structures, 6), sharey=True)
    
    # Handle case with only one structure
    if num_structures == 1:
        axes = [axes]
    
    for i, (structure, likelihoods_list) in enumerate(log_likelihoods.items()):
        ax = axes[i]
        
        for j, (likelihoods, model_name) in enumerate(zip(likelihoods_list, model_names)):
            # Remove outliers if requested
            if remove_outliers:
                q1 = np.percentile(likelihoods, 25)
                q3 = np.percentile(likelihoods, 75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                
                mask = (likelihoods >= lower_bound) & (likelihoods <= upper_bound)
                filtered_values = likelihoods[mask]
                
                print(f"Structure {structure}, Model {model_name}: "
                      f"Removed {len(likelihoods) - len(filtered_values)} outliers out of {len(likelihoods)} values")
            else:
                filtered_values = likelihoods
            
            # Plot histogram
            ax.hist(filtered_values, bins=30, alpha=0.7, 
                   color=model_colors.get(model_name, 'lightgray'),
                   label=f"$p_{{{model_name}}}$ (μ={np.mean(filtered_values):.2f})")
        
        ax.set_xlabel('Log Likelihood')
        if i == 0:
            ax.set_ylabel('Frequency')
        ax.set_title(f'Structure: {structure}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    title_suffix = " (outliers removed)" if remove_outliers else ""
    fig.suptitle(f'Log Likelihoods of Different Structural Elements Under Various Models{title_suffix}', 
                 fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_aa_log_likelihoods(
    log_likelihoods: dict, 
    amino_acid_models: list,
    save_path: str = "aa_loglikelihood_plot.png",
    remove_outliers: bool = True,
    outlier_threshold: float = 1.5
):
    """
    Plot log-likelihoods from multiple amino acid models as histograms.
    
    Parameters
    ----------
    log_likelihoods : dict
        Dictionary mapping amino acid type to lists of arrays of log likelihoods
        (one list per model)
    amino_acid_models : list
        List of amino acid models used (e.g., ['A', 'G', 'L', 'V', 'P', 'W', 'Y'])
    save_path : str
        Path to save the plot
    remove_outliers : bool
        Whether to remove outliers before plotting
    outlier_threshold : float
        Threshold for outlier removal (multiplier for IQR)
    """
    # Color palette for different amino acid models
    aa_colors = {
        'A': 'lightcoral',    # Alanine
        'G': 'lightblue',     # Glycine
        'L': 'lightgreen',    # Leucine
        'V': 'lightsalmon',   # Valine
        'P': 'plum',          # Proline
        'W': 'lightseagreen', # Tryptophan
        'Y': 'goldenrod',     # Tyrosine
        # Add more colors as needed
    }
    
    # Create a subplot for each amino acid type
    num_aas = len(log_likelihoods)
    fig, axes = plt.subplots(1, num_aas, figsize=(5*num_aas, 6), sharey=True)
    
    # Handle case with only one amino acid
    if num_aas == 1:
        axes = [axes]
    
    for i, (aa_type, likelihoods_list) in enumerate(log_likelihoods.items()):
        ax = axes[i]
        
        for j, (likelihoods, model_name) in enumerate(zip(likelihoods_list, amino_acid_models)):
            # Remove outliers if requested
            if remove_outliers:
                q1 = np.percentile(likelihoods, 25)
                q3 = np.percentile(likelihoods, 75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_threshold * iqr
                upper_bound = q3 + outlier_threshold * iqr
                
                mask = (likelihoods >= lower_bound) & (likelihoods <= upper_bound)
                filtered_values = likelihoods[mask]
                
                print(f"Amino acid {aa_type}, Model {model_name}: "
                      f"Removed {len(likelihoods) - len(filtered_values)} outliers out of {len(likelihoods)} values")
            else:
                filtered_values = likelihoods
            
            # Plot histogram
            ax.hist(filtered_values, bins=30, alpha=0.7, 
                   color=aa_colors.get(model_name, 'lightgray'),
                   label=f"$p_{{{model_name}}}$ (μ={np.mean(filtered_values):.2f})")
        
        ax.set_xlabel('Log Likelihood')
        if i == 0:
            ax.set_ylabel('Frequency')
        ax.set_title(f'Amino Acid: {aa_type}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    title_suffix = " (outliers removed)" if remove_outliers else ""
    fig.suptitle(f'Log Likelihoods of Different Amino Acids Under Various Models{title_suffix}', 
                 fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Define atoms and kernel sizes to loop over
    atoms = ['CA', 'N', 'H', 'HA', 'C', 'O']
    kernel_sizes = [0.05, 0.1, 0.2, 0.5]
    device = "cpu"
    
    # Create output directory if it doesn't exist
    os.makedirs("likelihood_results_mace", exist_ok=True)
    
    # Loop over all atom types and kernel sizes
    for atom in atoms:
        print(f"\n=== Processing atom type: {atom} ===")
        
        # Load descriptors for this atom type
        descriptor_path = f"./descriptors/mace_invariant/{atom}/{atom}_descriptors.pkl"
        try:
            with open(descriptor_path, "rb") as f:
                descriptors = pickle.load(f)
        except FileNotFoundError:
            print(f"Descriptor file not found for {atom}. Skipping.")
            continue
            
        # Split descriptors by pdb_id
        pdb_ids = descriptors['metadata']['pdb_id'].unique().tolist()
        
        # Train PDB IDs
        train_pdb_ids = pdb_ids[:int(0.8*len(pdb_ids))]
        
        # Test PDB IDs
        test_pdb_ids = pdb_ids[-int(0.2*len(pdb_ids)):]
        
        # Filter metadata by PDB IDs
        train_mask = descriptors['metadata']['pdb_id'].isin(train_pdb_ids)
        test_mask = descriptors['metadata']['pdb_id'].isin(test_pdb_ids)
        
        # Create filtered descriptor dictionaries
        train_descriptors = {
            'metadata': descriptors['metadata'][train_mask].reset_index(drop=True),
            'descriptors': np.array(descriptors['descriptors'])[train_mask.values]
        }
        
        test_descriptors = {
            'metadata': descriptors['metadata'][test_mask].reset_index(drop=True),
            'descriptors': np.array(descriptors['descriptors'])[test_mask.values]
        }
        
        # Process each kernel size
        for kernel_width in kernel_sizes:
            print(f"\n  Processing kernel width: {kernel_width}")
            
            # Create likelihood models for different structures
            helix_model = MACEbasedLikelihood.from_structure_type(
                train_descriptors, 
                structure_type='H',
                max_reference_size=5000, 
                kernel_width=kernel_width,
                device=device
            )
            sheet_model = MACEbasedLikelihood.from_structure_type(
                train_descriptors, 
                structure_type='E',
                max_reference_size=5000, 
                kernel_width=kernel_width,
                device=device
            )
            turn_model = MACEbasedLikelihood.from_structure_type(
                train_descriptors, 
                structure_type='T',
                max_reference_size=5000, 
                kernel_width=kernel_width,
                device=device
            )
            
            # Get test embeddings
            metadata_df = test_descriptors['metadata']
            metadata_df = metadata_df.reset_index(drop=True)
            embeddings = torch.tensor(test_descriptors['descriptors'], dtype=torch.float32, device=device)
            
            # Compute log likelihoods for all test data
            helix_log_liks = helix_model.compute_likelihood(embeddings)
            sheet_log_liks = sheet_model.compute_likelihood(embeddings)
            turn_log_liks = turn_model.compute_likelihood(embeddings)
            
            # Add predicted likelihoods to metadata
            metadata_df['helix_log_lik'] = helix_log_liks.detach().cpu().numpy()
            metadata_df['sheet_log_lik'] = sheet_log_liks.detach().cpu().numpy()
            metadata_df['turn_log_lik'] = turn_log_liks.detach().cpu().numpy()
            
            # Group by secondary structure
            grouped_by_structure = metadata_df.groupby('secondary_structure')
            
            # Prepare data for plotting
            log_likelihoods = {}
            for struct_type, group in grouped_by_structure:
                if struct_type not in ['H', 'E', 'T']:
                    continue
                    
                log_likelihoods[struct_type] = [
                    group['helix_log_lik'].values, 
                    group['sheet_log_lik'].values, 
                    group['turn_log_lik'].values
                ]
                
                print(f"    Structure '{struct_type}' statistics:")
                print(f"      Mean log likelihood under helix model: {group['helix_log_lik'].mean():.4f}")
                print(f"      Mean log likelihood under sheet model: {group['sheet_log_lik'].mean():.4f}")
                print(f"      Mean log likelihood under turn model: {group['turn_log_lik'].mean():.4f}")
            
            # Create filenames with atom type and kernel size
            csv_filename = f"likelihood_results_mace/metadata_with_likelihoods_{atom}_kernel_{kernel_width}.csv"
            plot_filename = f"likelihood_results_mace/likelihoods_by_structure_{atom}_kernel_{kernel_width}.png"
            
            # Plot likelihoods grouped by secondary structure
            plot_log_likelihoods(
                log_likelihoods,
                save_path=plot_filename,
                remove_outliers=True
            )
            
            # Save results to CSV
            metadata_df.to_csv(csv_filename, index=False)
            print(f"  Saved results to {csv_filename} and {plot_filename}")
            
            # === AMINO ACID ANALYSIS ===
            print(f"\n  Performing amino acid analysis with kernel width: {kernel_width}")
            
            # Select representative amino acids to model
            amino_acids = ['A', 'G', 'L', 'V', 'P', "W", "Y"]
            
            # Create likelihood models for different amino acids
            amino_acid_models = {}
            for aa in amino_acids:
                try:
                    aa_model = MACEbasedLikelihood.from_amino_acid_type(
                        train_descriptors,
                        amino_acid_type=aa,
                        max_reference_size=5000,
                        kernel_width=kernel_width,
                        device=device
                    )
                    amino_acid_models[aa] = aa_model
                except Exception as e:
                    print(f"    Error creating model for {aa}: {e}")
            
            # Get test embeddings (already done above)
            
            # Compute log likelihoods for all test data under each amino acid model
            for aa, model in amino_acid_models.items():
                aa_log_liks = model.compute_likelihood(embeddings)
                metadata_df[f'{aa}_log_lik'] = aa_log_liks.detach().cpu().numpy()
            
            # Group by amino acid type
            grouped_by_aa = metadata_df.groupby('name')
            
            # Prepare data for plotting
            aa_log_likelihoods = {}
            for aa_type, group in grouped_by_aa:
                # Only include the amino acids we're interested in
                if aa_type not in amino_acids:
                    continue
                
                # Collect log likelihoods under each amino acid model
                aa_log_likelihoods[aa_type] = [
                    group[f'{aa}_log_lik'].values for aa in amino_acids
                ]
                
                # Print statistics
                print(f"    Amino acid '{aa_type}' statistics:")
                for aa in amino_acids:
                    print(f"      Mean log likelihood under {aa} model: {group[f'{aa}_log_lik'].mean():.4f}")
            
            # Create filenames for amino acid analysis
            aa_csv_filename = f"likelihood_results_mace/aa_metadata_with_likelihoods_{atom}_kernel_{kernel_width}.csv"
            aa_plot_filename = f"likelihood_results_mace/aa_likelihoods_by_type_{atom}_kernel_{kernel_width}.png"
            
            # Plot likelihoods grouped by amino acid type
            plot_aa_log_likelihoods(
                aa_log_likelihoods,
                amino_acids,  # Pass the list of amino acid models
                save_path=aa_plot_filename,
                remove_outliers=True
            )
            
            # Save results to CSV
            metadata_df.to_csv(aa_csv_filename, index=False)
            print(f"  Saved amino acid results to {aa_csv_filename} and {aa_plot_filename}")
