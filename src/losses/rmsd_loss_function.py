from .abstract_loss_funciton import AbstractLossFunction
from ..utils.io import load_pdb_atom_locations_full, alignment_mask_by_chain
from ..protenix.metrics.rmsd import self_aligned_rmsd
import torch
import gemmi


class RMSDLossFunction(AbstractLossFunction):
    def __init__(
        self,
        reference_pdb,
        mask,
        sequences_dictionary,
        chains_to_read,
        rmsd_loss_sequence_indices,  # List of sequence indices (0-indexed) that get RMSD loss
        device="cpu",
        should_align_to_chains=None,  # Which chains to align to (for alignment mask)
        frozen_atoms_dict=None,  # For handling frozen atoms (if needed)
    ):
        """
        RMSD loss function for specific sequences/chains.
        
        Args:
            reference_pdb: Path to reference PDB file
            mask: Mask indicating which atoms are resolved in PDB (AF3_to_pdb_mask)
            sequences_dictionary: Dictionary of sequences (same format as ESP loss)
            chains_to_read: Which chains to read from PDB
            rmsd_loss_sequence_indices: List of sequence indices (0-indexed) that get RMSD loss
            device: Device to use
            should_align_to_chains: Which chains to align to (default: all sequences)
            frozen_atoms_dict: Dictionary with frozen atoms info (if should_concatenate_frozen_atoms is True)
        """
        self.device = device
        self.reference_pdb_path = reference_pdb
        self.rmsd_loss_sequence_indices = rmsd_loss_sequence_indices if rmsd_loss_sequence_indices is not None else []
        
        # Build full_sequences list (same as ESP loss)
        full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in sequences_dictionary]
        full_sequences = [item for sublist in full_sequences for item in sublist]
        self.full_sequences = full_sequences
        self.sequences_dictionary = sequences_dictionary
        
        self.sequence_types = [
            sequence_type
            for dictionary in sequences_dictionary
            for sequence_type in [dictionary.get("sequence_type", "proteinChain")] * dictionary["count"]
        ]
        
        # Create masks per sequence
        self.masks_per_sequence = [
            alignment_mask_by_chain(full_sequences, [sequence_id], self.sequence_types).to(device) 
            for sequence_id, sequence in enumerate(full_sequences)
        ]
        
        # Read PDB coordinates (same as ESP loss)
        self.coordinates_gt, _, _, self.element_gt = \
            load_pdb_atom_locations_full(
                pdb_file=reference_pdb, 
                full_sequences_dict=sequences_dictionary,
                chains_to_read=chains_to_read,
                return_elements=True,
                return_bfacs=True,
                return_mask=True,
            )
        
        self.coordinates_gt = self.coordinates_gt.to(device)
        self.element_gt = self.element_gt.to(device)
        self.AF3_to_pdb_mask = mask.to(device)
        
        # Determine which chains to align to
        if should_align_to_chains is None:
            should_align_to_chains = list(range(len(full_sequences)))
        self.should_align_to_chains = should_align_to_chains
        self.align_to_chain_mask = alignment_mask_by_chain(
            full_sequences, 
            chains_to_align=should_align_to_chains, 
            sequence_types=self.sequence_types
        ).to(device)
        
        # Create RMSD loss mask for specified sequences
        self.rmsd_loss_mask = torch.zeros((mask.shape[0]), dtype=torch.bool, device=device)
        for seq_idx in self.rmsd_loss_sequence_indices:
            if seq_idx < len(self.masks_per_sequence):
                seq_mask = self.masks_per_sequence[seq_idx]
                if seq_mask.shape[0] == mask.shape[0]:
                    self.rmsd_loss_mask = self.rmsd_loss_mask | seq_mask
        
        # Create alignment mask specifically for RMSD loss computation
        # This ensures alignment is done only on the chains we care about (rmsd_loss_sequence_indices)
        # rather than all chains in should_align_to_chains, making alignment more stable
        self.rmsd_alignment_mask = alignment_mask_by_chain(
            full_sequences,
            chains_to_align=self.rmsd_loss_sequence_indices,
            sequence_types=self.sequence_types
        ).to(device)
        
        # Handle frozen atoms if provided
        self.frozen_atoms_dict = frozen_atoms_dict
        self.should_concatenate_frozen_atoms = True if self.frozen_atoms_dict is not None else False
        if self.should_concatenate_frozen_atoms:
            self.close_to_relevant_chains_positions = self.frozen_atoms_dict["other_atoms_from_pdb_positions"].clone().detach()
            self.concatenation_of_close_to_relevant_chains_mask = torch.cat([
                torch.ones((self.coordinates_gt.shape[0]), dtype=torch.bool, device=self.device),
                torch.zeros((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
            ]).to(torch.bool)
        
        # Values for logging
        self.last_rmsd_loss_value = None
    
    def pre_optimization_step(self, x_0_hat, i=None, step=None):
        """
        Handle frozen atoms concatenation if needed.
        Note: If frozen atoms are already concatenated by the main loss (ESP), we detect that
        and extend our masks accordingly. Otherwise, we pass through.
        RMSD loss should only be computed on resolved atoms, not frozen ones.
        """
        # Check if frozen atoms are already concatenated (x_0_hat size matches extended size)
        if self.should_concatenate_frozen_atoms:
            expected_extended_size = self.concatenation_of_close_to_relevant_chains_mask.shape[0]
            if x_0_hat.shape[1] == expected_extended_size:
                # Frozen atoms are already concatenated - extend our masks
                self.rmsd_loss_mask_extended = torch.cat([
                    self.rmsd_loss_mask,
                    torch.zeros((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
                ]).to(torch.bool)
                self.coordinates_gt_extended = torch.cat([
                    self.coordinates_gt,
                    self.close_to_relevant_chains_positions,
                ])
                self.AF3_to_pdb_mask_extended = torch.cat([
                    self.AF3_to_pdb_mask,
                    torch.ones((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
                ]).to(torch.bool)
            else:
                # Frozen atoms not yet concatenated - we need to concatenate them
                # Align before concatenating (use RMSD alignment mask focused on rmsd_loss_sequence_indices)
                alignment_mask_for_rmsd = self.AF3_to_pdb_mask & self.rmsd_alignment_mask
                
                _, x_0_hat_aligned, _, _ = self_aligned_rmsd(
                    x_0_hat,
                    self.coordinates_gt.unsqueeze(0),
                    alignment_mask_for_rmsd
                )
                
                # Concatenate frozen atoms
                batch = x_0_hat_aligned.shape[0]
                total_atoms = self.concatenation_of_close_to_relevant_chains_mask.shape[0]
                
                concatenated_x_0_hat = torch.zeros(
                    (batch, total_atoms, 3),
                    device=x_0_hat_aligned.device,
                )
                
                # Place diffusing atoms
                concatenated_x_0_hat[:, self.concatenation_of_close_to_relevant_chains_mask, :] = x_0_hat_aligned
                
                # Place frozen atoms (detached, no gradients)
                frozen = self.close_to_relevant_chains_positions.clone().detach()
                if frozen.ndim == 2:
                    frozen = frozen.unsqueeze(0).expand(batch, -1, -1)
                concatenated_x_0_hat[:, ~self.concatenation_of_close_to_relevant_chains_mask, :] = frozen
                
                # Extend RMSD loss mask (add False for frozen atoms - they don't get RMSD loss)
                self.rmsd_loss_mask_extended = torch.cat([
                    self.rmsd_loss_mask,
                    torch.zeros((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
                ]).to(torch.bool)
                
                # Extend coordinates_gt and AF3_to_pdb_mask for loss computation
                self.coordinates_gt_extended = torch.cat([
                    self.coordinates_gt,
                    self.close_to_relevant_chains_positions,
                ])
                self.AF3_to_pdb_mask_extended = torch.cat([
                    self.AF3_to_pdb_mask,
                    torch.ones((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
                ]).to(torch.bool)
                
                return concatenated_x_0_hat
        
        return x_0_hat
    
    def post_optimization_step(self, x_0_hat):
        """
        Remove frozen atoms if we concatenated them (not if main loss did it).
        Clean up extended masks.
        """
        if self.should_concatenate_frozen_atoms:
            # Only remove if we concatenated (check if extended masks exist and x_0_hat matches extended size)
            if hasattr(self, 'rmsd_loss_mask_extended') and x_0_hat.shape[1] == self.rmsd_loss_mask_extended.shape[0]:
                # We concatenated - remove them
                x_0_hat = x_0_hat[:, self.concatenation_of_close_to_relevant_chains_mask, :]
            
            # Clean up extended masks (they'll be recreated in next pre_optimization_step if needed)
            if hasattr(self, 'rmsd_loss_mask_extended'):
                delattr(self, 'rmsd_loss_mask_extended')
            if hasattr(self, 'coordinates_gt_extended'):
                delattr(self, 'coordinates_gt_extended')
            if hasattr(self, 'AF3_to_pdb_mask_extended'):
                delattr(self, 'AF3_to_pdb_mask_extended')
        return x_0_hat
    
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        """
        Compute RMSD loss for specified sequences.
        
        Args:
            x_0_hat: Predicted coordinates [B, N, 3] (may include frozen atoms if concatenated)
            time: Time step (unused)
            structures: Structures (unused)
            i: Iteration index (unused)
            step: Step index (unused)
        
        Returns:
            loss: RMSD loss value
            None: No modified x_0_hat
            None: No additional return value
        """
        # Use extended masks if frozen atoms were concatenated
        if self.should_concatenate_frozen_atoms and hasattr(self, 'rmsd_loss_mask_extended'):
            rmsd_loss_mask = self.rmsd_loss_mask_extended
            coordinates_gt = self.coordinates_gt_extended
            AF3_to_pdb_mask = self.AF3_to_pdb_mask_extended
            # Extend RMSD alignment mask for frozen atoms case
            rmsd_alignment_mask = torch.cat([
                self.rmsd_alignment_mask,
                torch.zeros((self.close_to_relevant_chains_positions.shape[0]), dtype=torch.bool, device=self.device),
            ]).to(torch.bool)
        else:
            rmsd_loss_mask = self.rmsd_loss_mask
            coordinates_gt = self.coordinates_gt
            AF3_to_pdb_mask = self.AF3_to_pdb_mask
            rmsd_alignment_mask = self.rmsd_alignment_mask
        
        # Use RMSD-specific alignment mask (only chains in rmsd_loss_sequence_indices)
        # This ensures stable alignment focused on the chains we care about
        alignment_mask_for_rmsd = AF3_to_pdb_mask & rmsd_alignment_mask
        
        # Compute RMSD loss only on atoms specified by rmsd_loss_mask
        rmsd_mask_for_loss = alignment_mask_for_rmsd & rmsd_loss_mask
        if rmsd_mask_for_loss.sum() > 0:
            # CRITICAL: Detach non-masked atoms BEFORE alignment to prevent gradients
            # from flowing to non-masked atoms through the alignment transformation.
            # This ensures RMSD loss gradients ONLY flow to masked atoms.
            # Other losses are unaffected because they use the original x_0_hat directly.
            x_0_hat_masked = x_0_hat.clone()
            x_0_hat_masked[:, ~rmsd_mask_for_loss, :] = x_0_hat[:, ~rmsd_mask_for_loss, :].detach()
            
            # Align structures using RMSD alignment (only on chains specified in rmsd_loss_sequence_indices)
            # Alignment is computed from masked atoms (which have gradients) but applied to
            # a tensor where non-masked atoms are already detached, so they won't receive gradients.
            _, aligned_x_0_hat, _, _ = self_aligned_rmsd(
                x_0_hat_masked,
                coordinates_gt.unsqueeze(0),
                alignment_mask_for_rmsd
            )
            
            rmsd_loss = (
                (
                    aligned_x_0_hat[:, rmsd_mask_for_loss, :] - 
                    coordinates_gt[rmsd_mask_for_loss, :].unsqueeze(0)
                ).square().sum(dim=-1)
            ).mean().sqrt()
        else:
            rmsd_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        self.last_rmsd_loss_value = rmsd_loss.item()
        
        # Return loss only - does not modify x_0_hat (alignment is only for loss computation)
        return rmsd_loss, None, None
    
    def wandb_log(self, x_0_hat):
        """Log RMSD loss value."""
        return {"rmsd_loss": self.last_rmsd_loss_value}

