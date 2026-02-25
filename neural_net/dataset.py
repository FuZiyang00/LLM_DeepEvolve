from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Dict
import pandas as pd
import torch

class ContrastivePolicyDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_column: str = 'target',
                 n_negative: int = 4,
                 dummy_cols: List[str] = None, 
                 verbose: bool = False):
        
        """
        Custom Dataset for contrastive learning
        Creates training samples with:
        - anchor: query policy
        - candidates: list of n_negative opposite class policies + 1 same class policy
        - label: index of the same class policy in candidates list
        """

        self.sectors = df['Sector'].values
        dummies = pd.get_dummies(df[dummy_cols], drop_first=False)
        df = pd.concat([df,dummies], axis=1)

        self.df = df.copy().reset_index(drop=True)
        self.target_column = target_column
        self.targets = self.df[self.target_column].values.astype(np.int64)
        
        # create indices 
        self._create_class_sector_indices()

        drop_cols = dummy_cols + [self.target_column]
        self.df = self.df.drop(columns=drop_cols).reset_index(drop=True)
        if any(col in self.df.columns for col in dummy_cols):
            raise ValueError("Dummy columns should not be in the dataframe after processing.")
        
        features_columns = self.df.columns.tolist()
        self.features = self.df[features_columns].values.astype(np.float32)
        self.input_dim = self.features.shape[1]
        self.n_negative = n_negative
        self.n_candidates = n_negative + 1  # +1 for the positive sample 

        if verbose:
            print(f"Dataset initialized:")
            print(f"- Features: {len(features_columns)}")
            print(f"- Negative samples per query: {self.n_negative}")
            print("Indices initialized for efficient sampling.")


    def _create_class_sector_indices(self):
        """Create mapping of indices by class and sector for efficient sampling."""

        unique_sectors = self.df['Sector'].unique()

        # Create nested dictionary: {class: {sector: [indices]}}
        self.class_sector_indices = {}

        for cls in [0, 1]:
            self.class_sector_indices[cls] = {}
            class_df = self.df[self.df[self.target_column] == cls]
            for sector in unique_sectors:
                sector_indices = class_df[class_df["Sector"] == sector].index.tolist()
                if sector_indices:  # Only add if sector has samples for this class
                    self.class_sector_indices[cls][sector] = sector_indices
        
        # Also maintain simple class indices for fallback
        self.class_indices = {
            0: self.df[self.df[self.target_column] == 0].index.tolist(),
            1: self.df[self.df[self.target_column] == 1].index.tolist()
        }

    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        """
        Generate a contrastive learning sample.
        
        Args:
            idx: Index of the anchor sample
            
        Returns:
            Dictionary containing:
            - anchor: numpy array of anchor policy features
            - candidates: list of numpy arrays (n_negative opposite + 1 same class)
            - label: index of the same class policy in candidates list
        """

        if idx >= len(self):
            raise IndexError("Index out of bounds for dataset length.")
        
        # get anchor sample
        anchor_features = self.features[idx].ravel()
        anchor_target = self.targets[idx]
        anchor_sector = self.sectors[idx]

        similar_idx, negative_indices = self._sample_sector_aware_indices(anchor_target,
                                                                          anchor_sector, 
                                                                          idx,
                                                                          self.n_candidates)
                                                                          
        # create candidates list
        candidates = []
        for neg_idx in negative_indices:
            candidates.append(self.features[neg_idx].ravel())
        
        sim_pol = self.features[similar_idx].ravel()

        # shuffle the candidates
        shuffled_indices = list(range(len(candidates)))
        random.shuffle(shuffled_indices)

        # Reorder candidates according to shuffled indices
        shuffled_candidates = [candidates[i] for i in shuffled_indices]

        # Find the new position of the positive sample
        original_positive_idx = len(candidates) - 1  # Was last
        label = shuffled_indices.index(original_positive_idx)

        candidates = shuffled_candidates

        return {
            'anchors': anchor_features,
            'sim_pol': sim_pol,
            'candidates': candidates,
            'labels': label
        }
    
    def _sample_sector_aware_indices(self,
                                     target_class: int,
                                     anchor_sector: str,
                                     anchor_idx: int,
                                     n_samples: int):
        
        """
        Sample indices from the specified class, prioritizing the same sector.
        """
        # Try to sample from same sector first
        sector_indices = self.class_sector_indices[target_class][anchor_sector]
        available_sector_indices = [idx for idx in sector_indices if idx != anchor_idx]
        
        similar_index = None
        if len(available_sector_indices) >= 1:
            similar_index = random.sample(available_sector_indices, 1)
        
        else:
            # Fallback to any index from the class
            fallback_indices = [idx for idx in self.class_indices[target_class] if idx != anchor_idx]
            similar_index = random.sample(fallback_indices, 1)

        opposite_class = 1 - target_class
        opposite_indexes_list = []
        opp_sector_indices = self.class_sector_indices[opposite_class][anchor_sector]
        available_opp_sector_indices = [idx for idx in opp_sector_indices]

        if len(available_opp_sector_indices) >= n_samples - 1:
            opposite_indexes = random.sample(available_opp_sector_indices, n_samples - 1)
        else:
            # Fallback to any indices from the opposite class
            fallback_opp_indices = [idx for idx in self.class_indices[opposite_class]]
            opposite_indexes = random.sample(fallback_opp_indices, n_samples - 1)
        
        opposite_indexes_list.extend(opposite_indexes)

        return similar_index, opposite_indexes_list

    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle the contrastive learning samples.
        """
        anchors = torch.stack([torch.from_numpy(item['anchors']) for item in batch])
        sim_pol = torch.stack([torch.from_numpy(item['sim_pol']) for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])

        # Keep candidates as nested list structure for flexible processing
        candidates_batch = []
        for item in batch:
            item_candidates = [torch.from_numpy(c) for c in item['candidates']]
            item_candidates_stacked = torch.stack(item_candidates)
            # Shape: (n_candidates, input_dim)
            candidates_batch.append(item_candidates_stacked)

        candidates = torch.stack(candidates_batch)

        return {'anchors': anchors,
                'sim_pol': sim_pol,
                'candidates': candidates,  # List of lists of tensors
                'labels': labels}


class FixedSamplesDataset(Dataset):
    """Dataset that returns precomputed contrastive samples. No randomness."""

    def __init__(self, samples: List[Dict], input_dim: int):
        self.samples = samples
        self.input_dim = input_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, any]:
        return self.samples[idx]


def precompute_contrastive_samples(dataset: ContrastivePolicyDataset, seed: int) -> List[Dict]:
    """Precompute one fixed set of contrastive samples using the given seed."""
    random.seed(seed)
    np.random.seed(seed)
    samples = []
    for idx in range(len(dataset)):
        samples.append(dataset[idx])
    return samples