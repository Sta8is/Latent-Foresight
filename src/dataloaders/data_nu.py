import os.path as osp
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import os

class NuScenesRGBDataset(data.Dataset):
    def __init__(self, data_path, args, sequence_length, img_size, subset="train", eval_mode=False, eval_midterm=False, 
                       eval_longterm_ns=False, eval_modality=None):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.subset = subset 
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        self.eval_longterm_ns = eval_longterm_ns
        self.eval_modality = args.eval_modality
        self.camera_name = 'CAM_FRONT'
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
        self.scene_names = create_splits_scenes()[subset]
        self.scene_tokens = [s['token'] for s in self.nusc.scene if s['name'] in self.scene_names]

        self.sequences = self._build_full_frame_sequences()
        self.augmentations = {
                "random_crop" : args.random_crop,
                "random_horizontal_flip" : args.random_horizontal_flip,
                "random_time_flip" : args.random_time_flip,
                "timestep_augm" : args.timestep_augm,
                "no_timestep_augm" : args.no_timestep_augm}


    def _build_full_frame_sequences(self):
        """
        Build a list of valid sequences using ALL frames (samples and sweeps)
        """
        sequences = []
        
        for scene_token in self.scene_tokens:
            scene = self.nusc.get('scene', scene_token)
            
            # First, get all camera sample_data for this scene
            # Starting from the first sample's camera
            sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', sample_token)
            cam_token = sample['data'][self.camera_name]
            
            # Get all camera frames in temporal order
            cam_frames = []
            current_cam_token = cam_token
            
            while current_cam_token:
                cam_data = self.nusc.get('sample_data', current_cam_token)
                cam_frames.append(current_cam_token)
                current_cam_token = cam_data['next']
            
            sequences.append(cam_frames)
            # Create sequences with temporal subsampling
            # for i in range(0, len(cam_frames) - (self.sequence_length - 1) * self.temporal_step, 1):
            #     sequence = [cam_frames[i + j * self.temporal_step] for j in range(self.sequence_length)]
            #     sequences.append(sequence)
            
            # Print stats for this scene
            # print(f"Scene {scene['name']}: {len(cam_frames)} total camera frames, {len(cam_frames) - (self.sequence_length - 1) * self.temporal_step} sequences")
        return sequences

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: The number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the frames and their corresponding file paths for a given index.

        Args:
            idx (int): The index of the sequence.

        Returns:
            tuple: A tuple containing the frames and their corresponding file paths.
        """
        sequence = self.sequences[idx]
        frames_filepaths = []
        metadata = []
        for cam_token in sequence:
            # Get camera data directly (this includes both keyframes and intermediate frames)
            cam_data = self.nusc.get('sample_data', cam_token)
            img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
            frames_filepaths.append(img_path)
        if self.eval_mode:
            # "rgb" needs no extra ground truth: the target for an RGB head is the
            # frame itself, which is what gt already is.
            if self.eval_modality is None or self.eval_modality == "rgb":
                frames, gt = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm,  self.eval_longterm_ns,
                                               eval_modality=None)
                return frames, gt
                # frames, gt, gt_path = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, self.eval_modality)
                # return frames, gt, gt_path
            elif self.eval_modality=="depth":
                frames, gt, gt_depth = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm,
                                                        self.eval_longterm_ns, self.eval_modality)
                return frames, gt, gt_depth
                # frames, gt, gt_depth, gt_path = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, self.eval_modality)
                # return frames, gt, gt_depth, gt_path
                # frames, gt, gt_depth, gt_future, gt_path = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, self.eval_modality)
                # return frames, gt, gt_depth, gt_future, gt_path
            elif self.eval_modality=="surface_normals":
                frames, gt, gt_normals = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, 
                                                          self.eval_longterm_nsself.eval_modality)
                return frames, gt, gt_normals
                # frames, gt, gt_normals, gt_path = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, self.eval_modality)
                # return frames, gt, gt_normals, gt_path
                # frames, gt, gt_normals, gt_future, gt_path = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length, self.eval_midterm, self.eval_modality)
                # return frames, gt, gt_normals, gt_future, gt_path
        else:
            frames = process_trainmode(frames_filepaths,self.img_size, self.subset, self.augmentations, self.sequence_length)
            return frames

       
def process_trainmode(frames_path, img_size, subset, augmentations, sequence_length=5, num_frames_skip=0):
    if subset=="val":
        num_frames_skip = 2 
        step = num_frames_skip + 1  
        start_idx = 20 - step*sequence_length + num_frames_skip
    else:
        if augmentations["no_timestep_augm"] is True:
            num_frames_skip = 2
        elif augmentations["timestep_augm"] is not None:
            # num_frames_skip = np.random.choice(list(range(1,6)),p=[0.1,0.6,0.1,0.1,0.1]) [1,5] 6 is excluded
            num_frames_skip = np.random.choice(list(range(1,len(augmentations["timestep_augm"])+1),p=augmentations["timestep_augm"]))
        else:
            num_frames_skip = np.random.randint(1,4) # [1,3] with equal probabilities 4 is excluded
        step = num_frames_skip + 1 # [2,4] with equal probabilities or [2,N] with non-equal probabilities
        start_idx = np.random.randint(0, len(frames_path) - step*sequence_length + num_frames_skip + 1)
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    # Load frames as tensors and apply transformations]
    if augmentations["random_time_flip"] == True and subset=="train":
        sequence_frames_path = sequence_frames_path[::-1] if np.random.rand()>0.5 else sequence_frames_path
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    W, H = sequence_frames[0].size # PIL IMAGE
    if augmentations["random_crop"] == True and subset=="train":
        s_f = np.random.rand()/2 + 0.5 # [0.5, 1]
        size = (int(H*s_f),int(W*s_f))
        i, j, h, w = T.RandomCrop(size).get_params(sequence_frames[0], output_size=size)
        sequence_frames = [TF.crop(frame,i,j,h,w) for frame in sequence_frames]
    if augmentations["random_horizontal_flip"] == True and subset=="train":
        sequence_frames = [TF.hflip(frame) for frame in sequence_frames] if np.random.rand()>0.5 else sequence_frames
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)
    return sequence_tensor

def process_evalmode(frames_path, img_size, subset, sequence_length=5, eval_midterm=False, eval_longterm_ns=False, eval_modality=None):
    num_frames_skip = 2 
    step = num_frames_skip + 1
    # if eval_midterm and sequence_length<7:
    #     start_idx = 20 - step*sequence_length + num_frames_skip - 6
    # else:
    #     start_idx = 20 - step*sequence_length + num_frames_skip
    print("Len Frames path", len(frames_path))
    if eval_midterm and sequence_length<7:
        start_idx = 29 - step*sequence_length + num_frames_skip - 6 # 6 is 2 rollouts of 3 frames each
    elif eval_longterm_ns and sequence_length<7:
        start_idx = 29 - step*sequence_length + num_frames_skip - 15 # 15 is 5 rollouts of 3 frames each
        # start_idx = 38 - step*sequence_length + num_frames_skip - 24 # For longer term 9 steps
    else:
        start_idx = 29 - step*sequence_length + num_frames_skip
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    # print("idxs", list(range(30))[start_idx : start_idx + step*sequence_length : step])
    # print("28 idx:",list(range(30))[28])
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)
    # gt_path = frames_path[19]
    gt_path = frames_path[28]
    # gt_path = frames_path[37] # For longer term 9 steps
    gt_img = transform(Image.open(gt_path))
    if eval_modality is None:
        return sequence_tensor, gt_img
        # return sequence_tensor, gt_img, gt_path
    elif eval_modality=="segm":
        gt_segm_path = gt_path.replace("leftImg8bit_sequence","gtFine").replace("leftImg8bit","gtFine_labelTrainIds")
        gt_segm_img = Image.open(gt_segm_path)
        transform_segmap = T.PILToTensor()
        gt_segm_img = transform_segmap(gt_segm_img)
        return sequence_tensor, gt_img, gt_segm_img
        # return sequence_tensor, gt_img, gt_segm_img, gt_path
        # gt_future_path = frames_path[25]
        # gt_future_img = transform(Image.open(gt_future_path))
        # return sequence_tensor, gt_img, gt_segm_img, gt_future_img, gt_path
    elif eval_modality=="depth":
        # gt_depth_path = gt_path.replace("leftImg8bit_sequence","leftImg8bit_sequence_depthv2").replace("leftImg8bit.png","leftImg8bit_depth.png")
        gt_depth_path = gt_path.replace("NuScenes", "NuScenes_depthv2")
        gt_depth_img = Image.open(gt_depth_path)
        transform_depthmap = T.PILToTensor()
        gt_depth_img = transform_depthmap(gt_depth_img)
        return sequence_tensor, gt_img, gt_depth_img
        # return sequence_tensor, gt_img, gt_depth_img, gt_path
        # gt_future_path = frames_path[25]
        # gt_future_img = transform(Image.open(gt_future_path))
        # return sequence_tensor, gt_img, gt_depth_img, gt_future_img, gt_path
    elif eval_modality=="surface_normals":
        gt_normals_path = gt_path.replace("NuScenes","NuScenes_normals")
        gt_normals_img = np.load(gt_normals_path.replace("jpg","npy"))
        # gt_normals_img = np.load(gt_normals_path.replace("png","npy"))
        gt_normals_img = torch.from_numpy(gt_normals_img).permute(2, 0, 1)
        return sequence_tensor, gt_img, gt_normals_img
        # return sequence_tensor, gt_img, gt_normals_img, gt_path
        # return sequence_tensor, gt_img, gt_normals_img, gt_path
        # gt_future_path = frames_path[25]
        # gt_future_img = transform(Image.open(gt_future_path))
        # return sequence_tensor, gt_img, gt_normals_img, gt_future_img, gt_path

        

class NS_VideoData(pl.LightningDataModule):
    """
    LightningDataModule for loading CityScapes video data.

    Args:
        arguments: An object containing the required arguments for data loading.
        subset (str): The subset of the data to load. Default is "train".

    Attributes:
        data_path (str): The path to the data folder.
        subset (str): The subset of the data being loaded.
        sequence_length (int): The length of the video sequence.
        batch_size (int): The batch size for data loading.
        shape (tuple): The shape of the video frames.
        downsample_factor (int): The factor by which to downsample the frames.
    """

    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset  # ["train","val","test"]
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.img_size = arguments.img_size
        # assert self.img_size[0]%14==0 and self.img_size[1]%14==0, "Image size should be divisible by 14"
        self.arguments = arguments
        self.eval_midterm = arguments.eval_midterm
        self.eval_longterm_ns = arguments.eval_longterm_ns
        print("Eval midterm:", self.eval_midterm)
        print("Eval longterm:", self.eval_longterm_ns)
        assert not (self.eval_midterm and self.eval_longterm_ns), "Cannot evaluate mid-term and long-term at the same time"
        assert ((not self.eval_midterm) or (not self.eval_longterm_ns)), "Either mid-term or long-term evaluation must be enabled"
        self.eval_modality = arguments.eval_modality
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.eval_mode = arguments.eval_mode
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val

    def _dataset(self, subset, eval_mode):
        """
        Private method to create and return the CityScapesDataset object.

        Args:
            subset (str): The subset of the data to load.

        Returns:
            CityScapesDataset: The dataset object.
        """
        dataset = NuScenesRGBDataset(self.data_path, self.arguments, self.sequence_length, self.img_size, subset, eval_mode, 
                                     self.eval_midterm, self.eval_longterm_ns, self.eval_modality)
        return dataset

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False):
        """
        Private method to create and return the DataLoader object.

        Args:
            subset (str): The subset of the data to load.
            shuffle (bool): Whether to shuffle the data. Default is True.

        Returns:
            DataLoader: The dataloader object.
        """
        dataset = self._dataset(subset, eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        Method to return the dataloader for the training subset.

        Returns:
            DataLoader: The dataloader object for training data.
        """
        train_subset = "val" if self.use_val_to_train else "train"
        return self._dataloader(subset=train_subset, drop_last=True, eval_mode=False)

    def val_dataloader(self):
        """
        Method to return the dataloader for the validation subset.

        Returns:
            DataLoader: The dataloader object for validation data.
        """
        val_subset = "train" if self.use_train_to_val else "val"
        dataset = self._dataset(val_subset, self.eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_val, pin_memory=True, shuffle=False, drop_last=False)
        return dataloader

    def test_dataloader(self):
        """
        Method to return the dataloader for the test subset.

        Returns:
            DataLoader: The dataloader object for test data.
        """
        return self._dataloader(subset="test")


