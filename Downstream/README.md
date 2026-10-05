# Downstream Tasks
The predicted features can be decoded into dense scene-understanding outputs with off-the-shelf [DPT](https://github.com/isl-org/DPT) heads (our implementation is mostly based on the [DPT of DepthAnything](https://github.com/LiheYoung/Depth-Anything/tree/main/depth_anything)). The following tasks are supported:
1. [Semantic Segmentation](#semantic-segmentation)
2. [Depth Estimation](#depth-estimation)
3. [Surface Normal Estimation](#surface-normal-estimation)

All heads are trained on frozen DINOv2 (`vitb14_reg`) features extracted from layers 2, 5, 8 and 11 at 448x896 resolution on Cityscapes, with the same head configuration: `--dpt_out_channels 128,256,512,512 --use_bn --nfeats 256`.

Training scripts and code will be released soon!

## Oracle results
The tables below report the **oracle** performance of each head, i.e. the head applied to the real DINOv2 features of the target frame. This is the upper bound for the same head applied to predicted features.

## Checkpoints
The checkpoints are hosted at [Sta8is/Latent-Foresight_cs](https://huggingface.co/Sta8is/Latent-Foresight_cs). To download all of them via command line:
```bash
hf download Sta8is/Latent-Foresight_cs head_segm.pth head_depth.pth head_normals.pth --local-dir checkpoints
```

## Semantic Segmentation
Head configuration: `--modality segm --num_classes 19`.

| Checkpoint | mIoU | MO_mIoU |
| - | - | - |
| [head_segm.pth](https://huggingface.co/Sta8is/Latent-Foresight_cs/blob/main/head_segm.pth) | 77.1 | 77.2 |

```bash
wget https://huggingface.co/Sta8is/Latent-Foresight_cs/resolve/main/head_segm.pth
```

## Depth Estimation
Head configuration: `--modality depth --num_classes 256`.

| Checkpoint | δ1 ↑ | δ2 ↑ | δ3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE log ↓ | log10 ↓ | SILog ↓ |
| - | - | - | - | - | - | - | - | - | - |
| [head_depth.pth](https://huggingface.co/Sta8is/Latent-Foresight_cs/blob/main/head_depth.pth) | 89.5 | 96.8 | 98.2 | 0.104 | 0.717 | 5.63 | 0.197 | 0.042 | 19.47 |

```bash
wget https://huggingface.co/Sta8is/Latent-Foresight_cs/resolve/main/head_depth.pth
```

## Surface Normal Estimation
Head configuration: `--modality surface_normals --num_classes 3`. The a<sub>1</sub>-a<sub>5</sub> columns report the percentage of pixels with an angular error below 5°, 7.5°, 11.25°, 22.5° and 30° respectively.

| Checkpoint | Mean AE ↓ | Median AE ↓ | RMSE ↓ | a<sub>1</sub> ↑ | a<sub>2</sub> ↑ | a<sub>3</sub> ↑ | a<sub>4</sub> ↑ | a<sub>5</sub> ↑ |
| - | - | - | - | - | - | - | - | - |
| [head_normals.pth](https://huggingface.co/Sta8is/Latent-Foresight_cs/blob/main/head_normals.pth) | 2.86 | 1.46 | 5.13 | 85.5 | 92.3 | 96.4 | 98.9 | 99.3 |

```bash
wget https://huggingface.co/Sta8is/Latent-Foresight_cs/resolve/main/head_normals.pth
```
