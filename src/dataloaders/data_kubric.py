import os
import os.path as osp
import glob
import json
import warnings
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from PIL import Image

class KubricRGBDataset(data.Dataset):
    """
    Kubric MOVi dataset, exported to PNGs by export_movi_to_png.py.

    Expects the layout produced by that script (also what VFMF's KubricDataset
    reads), with data_path pointing at the {suite}/{resolution} level:

        {data_path}/{split}/{sequence}/00000_rgb.png ... 00023_rgb.png
                                       depth_000NN.png
                                       segmentation_000NN.png
                                       normal_000NN.png
                                       data_ranges.json

    With sequence_length=5 the eval window is frames [0, 2, 4, 6, 8]: the first
    four are the context and the fifth is the slot the model predicts. This is
    VFMF's "uncertain" Kubric selection exactly.

    Which frame is scored depends on how many autoregressive steps the model
    takes (see _eval_window):

        short term  (1 step,  default)      -> frame 8
        mid term    (8 steps, --eval_midterm) -> frame 22

    matching the two horizons in vfmf/world-model/eval_kubric.py, whose
    TARGET_ROLLOUT "first"/"last" score frames 8 and 22 off the same context.
    """

    def __init__(self, data_path, args, sequence_length, img_size, subset="train", eval_mode=False,
                 eval_midterm=False, eval_modality=None):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.subset = subset
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        # Number of autoregressive steps the model will take at eval, which
        # determines which frame is the ground truth. Derived from args so it
        # cannot disagree with what evaluation_step actually does.
        self.unroll_steps = _eval_unroll_steps(args)
        self.eval_modality = args.eval_modality

        split_dir = self._resolve_split_dir(data_path, subset)
        self.sequences = sorted(
            osp.join(split_dir, seq) for seq in os.listdir(split_dir)
            if osp.isdir(osp.join(split_dir, seq))
        )
        if len(self.sequences) == 0:
            raise RuntimeError(f"No Kubric sequences found under {split_dir}")
        # MOVi clips are a fixed length (24 frames at 12fps), but read it off the
        # data rather than assuming it, so other suites keep working.
        self.frames_per_sequence = len(glob.glob(osp.join(self.sequences[0], "*_rgb.png")))
        self.num_frames = len(self.sequences) * self.frames_per_sequence
        print(f"Found {len(self.sequences)} sequences in Kubric {subset} set ({split_dir})")

        self.augmentations = {
                "random_crop": args.random_crop,
                "random_horizontal_flip": args.random_horizontal_flip,
                "random_time_flip": args.random_time_flip,
                "timestep_augm": args.timestep_augm,
                "no_timestep_augm": args.no_timestep_augm}

    @staticmethod
    def _resolve_split_dir(data_path, subset):
        """Map the repo's train/val naming onto Kubric's train/validation dirs."""
        candidates = ["validation", "val"] if subset in ("val", "validation") else [subset]
        for name in candidates:
            path = osp.join(data_path, name)
            if osp.isdir(path):
                return path
        raise FileNotFoundError(
            f"None of {candidates} found under {data_path}. "
            f"Available: {sorted(os.listdir(data_path)) if osp.isdir(data_path) else 'path missing'}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_dir = self.sequences[idx]
        frames_filepaths = sorted(glob.glob(osp.join(sequence_dir, "*_rgb.png")))
        if self.eval_mode:
            # "rgb" needs no extra ground truth: the target for an RGB head is the
            # frame itself, which is what gt already is.
            if self.eval_modality is None or self.eval_modality == "rgb":
                frames, gt = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length,
                                              self.unroll_steps)
                return frames, gt
            elif self.eval_modality == "segm":
                frames, gt, gt_segm = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length,
                                                       self.unroll_steps, self.eval_modality)
                return frames, gt, gt_segm
            elif self.eval_modality == "depth":
                frames, gt, gt_depth = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length,
                                                        self.unroll_steps, self.eval_modality)
                return frames, gt, gt_depth
            elif self.eval_modality == "surface_normals":
                frames, gt, gt_normals = process_evalmode(frames_filepaths, self.img_size, self.subset, self.sequence_length,
                                                          self.unroll_steps, self.eval_modality)
                return frames, gt, gt_normals
        else:
            frames = process_trainmode(frames_filepaths, self.img_size, self.subset, self.augmentations,
                                       self.sequence_length)
            return frames


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _eval_unroll_steps(args):
    """
    How many autoregressive steps evaluation_step will take.

    Normally resolved in train_mgivt.py so this and the model agree; the
    fallback (8 for Kubric's mid-term horizon, matching VFMF) only applies to
    args predating that argument.
    """
    unroll_steps = getattr(args, 'eval_unroll_steps', None)
    if unroll_steps is None:
        unroll_steps = 8 if getattr(args, 'eval_midterm', False) else 1
    return max(1, unroll_steps)


def _eval_window(frames_path, sequence_length, unroll_steps=1):
    """
    Frame indices for the eval window, plus the index of the scored frame.

    The window is VFMF's Kubric context: start 0, stride 2, i.e. [0,2,4,6] plus
    a trailing slot the model overwrites with noise and predicts.

    sample_unroll predicts that trailing slot, rolls the window forward and
    repeats, so after `unroll_steps` iterations the prediction corresponds to
    window_end + (unroll_steps-1)*stride. The ground-truth frame has to follow
    that, otherwise the model is scored against the wrong frame -- which is how
    Cityscapes' midterm start_idx picks up its -6 offset. With stride 2 on a
    24-frame clip, unroll_steps=8 lands on frame 22, exactly VFMF's gt_idx for
    dino_foresight_eval_selection(horizon="uncertain").
    """
    num_frames_skip = 1
    step = num_frames_skip + 1  # stride 2, as in VFMF's Kubric selection
    indices = list(range(0, step * sequence_length, step))
    gt_idx = indices[-1] + (unroll_steps - 1) * step
    if gt_idx > len(frames_path) - 1:
        raise ValueError(
            f"unroll_steps={unroll_steps} with stride {step} and sequence_length="
            f"{sequence_length} scores frame {gt_idx}, past the end of a "
            f"{len(frames_path)}-frame Kubric clip."
        )
    return indices, gt_idx


def process_trainmode(frames_path, img_size, subset, augmentations, sequence_length=5, num_frames_skip=0):
    if subset in ("val", "validation"):
        # Deterministic window; this path is the plain val loss, never a rollout.
        indices, _ = _eval_window(frames_path, sequence_length, unroll_steps=1)
        sequence_frames_path = [frames_path[i] for i in indices]
    else:
        if augmentations["no_timestep_augm"] is True:
            num_frames_skip = 1
        elif augmentations["timestep_augm"] is not None:
            num_frames_skip = np.random.choice(list(range(1, len(augmentations["timestep_augm"]) + 1)),
                                               p=augmentations["timestep_augm"])
        else:
            # [1,2] -> stride 2 or 3. Kubric clips are only 24 frames long, so a
            # larger stride would not leave room for a full sequence.
            num_frames_skip = np.random.randint(1, 3)
        step = num_frames_skip + 1
        max_start = len(frames_path) - step * sequence_length + num_frames_skip
        if max_start < 0:
            raise ValueError(
                f"Cannot sample {sequence_length} frames with stride {step} from a "
                f"{len(frames_path)}-frame Kubric clip."
            )
        start_idx = np.random.randint(0, max_start + 1)
        sequence_frames_path = frames_path[start_idx: start_idx + step * sequence_length: step]

    if augmentations["random_time_flip"] is True and subset == "train":
        sequence_frames_path = sequence_frames_path[::-1] if np.random.rand() > 0.5 else sequence_frames_path
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    W, H = sequence_frames[0].size  # PIL IMAGE
    if augmentations["random_crop"] is True and subset == "train":
        s_f = np.random.rand() / 2 + 0.5  # [0.5, 1]
        size = (int(H * s_f), int(W * s_f))
        i, j, h, w = T.RandomCrop(size).get_params(sequence_frames[0], output_size=size)
        sequence_frames = [TF.crop(frame, i, j, h, w) for frame in sequence_frames]
    if augmentations["random_horizontal_flip"] is True and subset == "train":
        sequence_frames = [TF.hflip(frame) for frame in sequence_frames] if np.random.rand() > 0.5 else sequence_frames
    mean, std = MEAN, STD
    transform = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)
    return sequence_tensor


def process_evalmode(frames_path, img_size, subset, sequence_length=5, unroll_steps=1, eval_modality=None):
    indices, gt_idx = _eval_window(frames_path, sequence_length, unroll_steps)
    sequence_frames_path = [frames_path[i] for i in indices]
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    mean, std = MEAN, STD
    transform = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    sequence_tensors = [transform(frame) for frame in sequence_frames]
    sequence_tensor = torch.stack(sequence_tensors, dim=0)

    # The scored frame is where the rollout lands, which is only the window's
    # last frame for a single step. With unroll_steps>1 it is further ahead, so
    # index it off gt_idx rather than off the window.
    gt_path = frames_path[gt_idx]
    gt_img = transform(Image.open(gt_path).convert('RGB'))
    if eval_modality is None:
        return sequence_tensor, gt_img

    sequence_dir = osp.dirname(gt_path)
    resize_nearest = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
    resize_bilinear = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)

    if eval_modality == "segm":
        # Instance ids with background 0; movi_a has no semantic classes, so VFMF
        # thresholds this to a binary foreground mask for its benchmark.
        gt_segm_path = osp.join(sequence_dir, f"segmentation_{gt_idx:05d}.png")
        gt_segm_img = resize_nearest(T.PILToTensor()(Image.open(gt_segm_path)))
        return sequence_tensor, gt_img, gt_segm_img
    elif eval_modality == "depth":
        # uint16 PNG quantising the metric range recorded in data_ranges.json.
        gt_depth_path = osp.join(sequence_dir, f"depth_{gt_idx:05d}.png")
        depth = np.array(Image.open(gt_depth_path)).astype(np.float32)
        with open(osp.join(sequence_dir, "data_ranges.json"), "rb") as fs:
            depth_range = json.load(fs)["depth"]
        # depth = depth_range["min"] + depth / 65535.0 * (depth_range["max"] - depth_range["min"])
        depth = depth / 65535.0
        gt_depth_img = resize_bilinear(torch.from_numpy(depth).unsqueeze(0))
        return sequence_tensor, gt_img, gt_depth_img
    elif eval_modality == "surface_normals":
        # 16-bit RGB, returned as unit vectors. The stored values are scaled but
        # deliberately not recentred: only the direction survives normalisation,
        # and a (2x - 1) shift is not a uniform scaling, so it would rotate every
        # vector (~36deg of error when checked against VFMF's normals head).
        # PIL also silently downconverts 16-bit RGB to 8-bit, hence _read_rgb16.
        gt_normals_path = osp.join(sequence_dir, f"normal_{gt_idx:05d}.png")
        normals = _read_rgb16(gt_normals_path).astype(np.float32) / 65535.0
        normals = torch.from_numpy(normals.copy()).permute(2, 0, 1)
        gt_normals_img = torch.nn.functional.normalize(resize_bilinear(normals), dim=0, eps=1e-6)
        return sequence_tensor, gt_img, gt_normals_img


def _read_rgb16(path):
    """
    Read a 16-bit RGB PNG as an (H, W, 3) uint16 array.

    PIL silently downconverts 3-channel 16-bit PNGs to 8-bit, which would throw
    away half the precision of the normals, so go through cv2 (or pypng) first.
    """
    try:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr[..., ::-1][..., :3]  # BGR -> RGB
    except ImportError:
        pass
    # Fallback: PIL gives 8-bit here, so rescale to keep the downstream maths
    # consistent. Precision is lost, hence the warning.
    warnings.warn(f"cv2 unavailable; reading {osp.basename(path)} at 8-bit precision")
    return np.array(Image.open(path).convert("RGB")).astype(np.uint16) * 257


class KUBRIC_VideoData(pl.LightningDataModule):
    """
    LightningDataModule for Kubric MOVi video data.

    Args:
        arguments: An object containing the required arguments for data loading.
        subset (str): The subset of the data to load. Default is "train".
    """

    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset  # ["train","val"]
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.img_size = arguments.img_size
        self.arguments = arguments
        self.eval_midterm = arguments.eval_midterm
        self.eval_modality = arguments.eval_modality
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.eval_mode = arguments.eval_mode
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val

    def _dataset(self, subset, eval_mode):
        dataset = KubricRGBDataset(self.data_path, self.arguments, self.sequence_length, self.img_size, subset,
                                   eval_mode, self.eval_midterm, self.eval_modality)
        return dataset

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False):
        dataset = self._dataset(subset, eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                     pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        train_subset = "val" if self.use_val_to_train else "train"
        return self._dataloader(subset=train_subset, drop_last=True, eval_mode=False)

    def val_dataloader(self):
        val_subset = "train" if self.use_train_to_val else "val"
        dataset = self._dataset(val_subset, self.eval_mode)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_val,
                                     pin_memory=True, shuffle=False, drop_last=False)
        return dataloader

    def test_dataloader(self):
        return self._dataloader(subset="val")
