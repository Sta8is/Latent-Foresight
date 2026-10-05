import os.path as osp
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from PIL import Image
from src.dataloaders.data import CityScapesRGBDataset, CS_VideoData
from src.dataloaders.data_nu import NuScenesRGBDataset, NS_VideoData
from src.dataloaders.data_covla import CovlaRGBDataset, COVLA_VideoData

class MultiSourceDataset(data.Dataset):
    """
    A multi-source dataset that combines CityScapes and NuScenes datasets with random sampling.
    
    Args:
        cityscapes_dataset: CityScapes dataset instance
        nuscenes_dataset: NuScenes dataset instance
        random_sampling (bool): Whether to use random sampling instead of sequential
        
    Attributes:
        cityscapes_dataset: The CityScapes dataset
        nuscenes_dataset: The NuScenes dataset
        cs_length: Number of samples in CityScapes dataset
        ns_length: Number of samples in NuScenes dataset
        total_length: Total number of samples from both datasets
        sampling_mode: Sampling mode for multisource dataset
        sampling_indices: Pre-generated random indices for consistent epoch sampling
    """
    
    def __init__(self, cityscapes_dataset, nuscenes_dataset, covla_dataset, sampling_mode='equal_random'):
        super().__init__()
        self.cityscapes_dataset = cityscapes_dataset
        self.nuscenes_dataset = nuscenes_dataset
        self.covla_dataset = covla_dataset
        self.cs_length = len(cityscapes_dataset)
        self.ns_length = len(nuscenes_dataset)
        self.covla_length = len(covla_dataset)
        self.sampling_mode = sampling_mode
        if sampling_mode == 'equal_random':
            self.total_length = max(self.cs_length, self.ns_length, self.covla_length)
        elif sampling_mode == 'sequential':
            self.total_length = self.cs_length + self.ns_length + self.covla_length
        else:
            raise ValueError(f"Invalid sampling_mode: {sampling_mode}. Choose 'equal_random' or 'sequential'.")
        # Generate random sampling indices if random sampling is enabled
        # if self.sampling_mode == 'equal_random':
        #     self._generate_equal_random_indices()
        


    # def _generate_equal_random_indices(self):
    #     """Generate random indices for sampling from both datasets."""
    #     # Create indices for each dataset
    #     ns_indices = [(i % self.ns_length, 'nuscenes') for i in range(self.total_length // 2)]
    #     cs_indices = [(i % self.cs_length, 'cityscapes') for i in range(self.total_length // 2)]
        
    #     # Combine and shuffle
    #     self.sampling_indices = ns_indices + cs_indices
    #     # np.random.shuffle(self.sampling_indices)
    


    def __len__(self):
        """Returns the total number of samples from both datasets."""
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from either CityScapes or NuScenes dataset.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            tuple: Sample data with dataset source identifier.
        """
        if self.sampling_mode == 'equal_random':
            # select with equal probability from 3 datasets
            rand_val = np.random.rand()
            if rand_val < 1/3:
                sample = self.cityscapes_dataset[np.random.randint(0, self.cs_length)]
            elif rand_val < 2/3:
                sample = self.nuscenes_dataset[np.random.randint(0, self.ns_length)]
            else:
                sample = self.covla_dataset[np.random.randint(0, self.covla_length)]
            # Equal probability random sampling from both datasets
            # if np.random.rand() < 0.5:
            #     sample = self.cityscapes_dataset[np.random.randint(0, self.cs_length)]
            # else:
            #     sample = self.nuscenes_dataset[np.random.randint(0, self.ns_length)]
            # First half: CityScapes, Second half: NuScenes
            # max_length = self.total_length // 2
            
            # if idx < max_length:
            #     # CityScapes sample (with wraparound for smaller dataset)
            #     dataset_idx = idx % self.cs_length
            #     sample = self.cityscapes_dataset[dataset_idx]
            # else:
            #     # NuScenes sample (with wraparound for smaller dataset)
            #     dataset_idx = (idx - max_length) % self.ns_length
            #     sample = self.nuscenes_dataset[dataset_idx]


            # # Use pre-generated random indices
            # dataset_idx, dataset_source = self.sampling_indices[idx]
            
            # if dataset_source == 'cityscapes':
            #     sample = self.cityscapes_dataset[dataset_idx]
            # else:  # nuscenes
            #     sample = self.nuscenes_dataset[dataset_idx]
            return sample
        elif self.sampling_mode == 'sequential':
            # Sequential sampling (original behavior)
            if idx < self.cs_length:
                # Sample from CityScapes dataset
                sample = self.cityscapes_dataset[idx]
                return sample
            # else:
                # # Sample from NuScenes dataset
                # ns_idx = idx - self.cs_length
                # sample = self.nuscenes_dataset[ns_idx]
                # return sample
            elif idx < self.cs_length + self.ns_length:
                    # Sample from NuScenes dataset
                    ns_idx = idx - self.cs_length
                    sample = self.nuscenes_dataset[ns_idx]
                    return sample
            else:
                # Sample from CoVLA dataset
                covla_idx = idx - self.cs_length - self.ns_length
                sample = self.covla_dataset[covla_idx]
                return sample
        else:
            raise ValueError(f"Invalid sampling_mode: {self.sampling_mode}. Choose 'sequential', or 'equal_random'.")

class MultiSourceVideoData(pl.LightningDataModule):
    """
    LightningDataModule for loading both CityScapes and NuScenes video data.
    
    Args:
        arguments: An object containing the required arguments for data loading.
        cs_data_path (str): Path to CityScapes data.
        ns_data_path (str): Path to NuScenes data.
        covla_data_path (str): Path to CoVLA data.
        subset (str): The subset of the data to load. Default is "train".
        batch_size (int): Batch size for data loading.
        random_sampling (bool): Whether to use random sampling from both datasets.
        
    Attributes:
        cs_data_module: CityScapes data module
        ns_data_module: NuScenes data module
        covla_data_module: CoVLA data module
        arguments: Configuration arguments
        batch_size: Batch size for loading
        eval_mode: Whether in evaluation mode
        random_sampling: Whether to use random sampling
    """

    def __init__(self, arguments, cs_data_path=None, ns_data_path=None, covla_data_path=None, subset="train", batch_size=8, sampling_mode="equal_random"):
        super().__init__()
        self.arguments = arguments
        self.batch_size = batch_size
        self.eval_mode = arguments.eval_mode
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.sampling_mode = sampling_mode
        
        # Create individual data modules
        cs_args = self._create_cs_args(arguments, cs_data_path)
        ns_args = self._create_ns_args(arguments, ns_data_path)
        covla_args = self._create_covla_args(arguments, covla_data_path)
        
        self.cs_data_module = CS_VideoData(cs_args, subset, batch_size)
        self.ns_data_module = NS_VideoData(ns_args, subset, batch_size)
        self.covla_data_module = COVLA_VideoData(covla_args, subset, batch_size)
        
    def _create_cs_args(self, arguments, cs_data_path):
        """Create arguments object for CityScapes dataset."""
        cs_args = type('Args', (), {})()
        for attr in dir(arguments):
            if not attr.startswith('_'):
                setattr(cs_args, attr, getattr(arguments, attr))
        if cs_data_path:
            cs_args.data_path = cs_data_path
        return cs_args
    
    def _create_ns_args(self, arguments, ns_data_path):
        """Create arguments object for NuScenes dataset."""
        ns_args = type('Args', (), {})()
        for attr in dir(arguments):
            if not attr.startswith('_'):
                setattr(ns_args, attr, getattr(arguments, attr))
        if ns_data_path:
            ns_args.data_path = ns_data_path
        return ns_args

    def _create_covla_args(self, arguments, covla_data_path):
        """Create arguments object for CoVLA dataset."""
        covla_args = type('Args', (), {})()
        for attr in dir(arguments):
            if not attr.startswith('_'):
                setattr(covla_args, attr, getattr(arguments, attr))
        if covla_data_path:
            covla_args.data_path = covla_data_path
        return covla_args
    
    def _create_multisource_dataset(self, subset, eval_mode, sampling_mode='equal_random'):
        """Create the multisource dataset combining both datasets."""
        cs_dataset = self.cs_data_module._dataset(subset, eval_mode)
        ns_dataset = self.ns_data_module._dataset(subset, eval_mode)
        covla_dataset = self.covla_data_module._dataset(subset, eval_mode)
        return MultiSourceDataset(cs_dataset, ns_dataset, covla_dataset, sampling_mode=sampling_mode)

    def _create_multisource_dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False, sampling_mode='equal_random'):
        """Create dataloader for the multisource dataset."""
        dataset = self._create_multisource_dataset(subset, eval_mode, sampling_mode=sampling_mode)
        num_workers = self.num_workers_val if subset == 'val' and eval_mode else self.num_workers
        dataloader = data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=shuffle, 
            drop_last=drop_last
        )
        return dataloader
    
    def train_dataloader(self):
        """Return the dataloader for the training subset."""
        train_subset = "val" if getattr(self.arguments, 'use_val_to_train', False) else "train"
        return self._create_multisource_dataloader(subset=train_subset, drop_last=True, eval_mode=False)
    
    def val_dataloader(self):
        """Return the dataloader for the validation subset."""
        val_subset = "train" if getattr(self.arguments, 'use_train_to_val', False) else "val"
        return self._create_multisource_dataloader(subset=val_subset, shuffle=False, eval_mode=self.eval_mode, sampling_mode='sequential')
    
    def test_dataloader(self):
        """Return the dataloader for the test subset."""
        return self._create_multisource_dataloader(subset="test", shuffle=False, eval_mode=True)


# Example usage function
def create_multisource_datamodule(arguments, cs_data_path, ns_data_path, covla_data_path=None, use_custom_collate=False, sampling_mode="equal_random"):
    """
    Convenience function to create a multisource data module.
    
    Args:
        arguments: Configuration arguments
        cs_data_path: Path to CityScapes data
        ns_data_path: Path to NuScenes data
        use_custom_collate: Whether to use custom collate function
        sampling_mode: Sampling mode for multisource dataset
        
    Returns:
        MultiSourceVideoData: The configured data module
    """
    if use_custom_collate:
        return MultiSourceVideoDataWithCustomCollate(
            arguments, 
            cs_data_path=cs_data_path, 
            ns_data_path=ns_data_path, 
            batch_size=arguments.batch_size if hasattr(arguments, 'batch_size') else 8,
            sampling_mode=sampling_mode
        )
    else:
        return MultiSourceVideoData(
            arguments, 
            cs_data_path=cs_data_path, 
            ns_data_path=ns_data_path, 
            covla_data_path=covla_data_path,
            batch_size=arguments.batch_size if hasattr(arguments, 'batch_size') else 8,
            sampling_mode=sampling_mode
        )