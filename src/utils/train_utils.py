# src/pairwise_ranking_loss.py
import torch
import torch.nn as nn

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initialize the Pairwise Ranking Loss class.

        Args:
            margin (float): Margin for the ranking loss. Default is 1.0.
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, representations, labels):
        """
        Compute the pairwise ranking loss for a batch of representations and labels.

        Args:
            representations (torch.Tensor): The representations of shape (batch_size, embedding_dim).
            labels (torch.Tensor): The class labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed pairwise ranking loss.
        """
        # Ensure the batch is sorted by labels
        sorted_indices = torch.argsort(labels)
        representations = representations[sorted_indices]
        labels = labels[sorted_indices]

        # Calculate the number of images per class
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        num_images_per_class = len(labels) // num_classes

        # Initialize loss
        pairwise_loss = 0.0
        num_pairs = 0

        # Compute pairwise ranking loss between successive classes
        for i in range(num_classes - 1):
            start_idx_class_i = i * num_images_per_class
            end_idx_class_i = (i + 1) * num_images_per_class
            start_idx_class_j = (i + 1) * num_images_per_class
            end_idx_class_j = (i + 2) * num_images_per_class

            # Representations of the current class
            class_i_repr = representations[start_idx_class_i:end_idx_class_i]
            # Representations of the next class
            class_j_repr = representations[start_idx_class_j:end_idx_class_j]

            # Compute pairwise ranking loss for all pairs between class_i and class_j
            for repr_i in class_i_repr:
                for repr_j in class_j_repr:
                    # Calculate the margin-based pairwise ranking loss
                    loss = torch.relu(self.margin - (repr_i - repr_j).norm(p=2))
                    pairwise_loss += loss
                    num_pairs += 1

        # Average the pairwise loss
        pairwise_loss = pairwise_loss / num_pairs if num_pairs > 0 else pairwise_loss

        return pairwise_loss
