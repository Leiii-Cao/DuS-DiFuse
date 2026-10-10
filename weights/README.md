# 📦 Checkpoints

## 🔗 DuS-DiFuse Weights

Download: [Google Drive](https://drive.google.com/drive/folders/1SbWD1RLpYU7dkKKgLZUhu8GUP5j-bBf1)

| Path | Module |
| --- | --- |
| `weights/mda.pth` | Multimodal Degradation-Aware encoder |
| `weights/sde.pt` | dual-stream degradation-decoupling UNet |
| `weights/GFCM.pt` | Groupwise Fusion Control Module |
| `weights/DRFM.pt` | Detail-Restoration Fidelity Module |
| `weights/Generative_Control.pt` | Generative Modulation ControlNet |

## 🧩 Third-Party Models

| Path | Download | Used by |
| --- | --- | --- |
| `weights/v2-1_512-ema-pruned.ckpt` | [Stable Diffusion 2.1](https://huggingface.co/ckpt/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt) | inference, SDE, GFCM, DRFM, Generative Modulation |
| `weights/open_clip_pytorch_model.bin` | [OpenCLIP ViT-H-14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin) | MDA training |
| `weights/groundingdino_swinb_cogcoor.pth` | [GroundingDINO Swin-B](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth) | text localization |
| `weights/sam_vit_h_4b8939.pth` | [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | text localization |
| `weights/bert-base-uncased/` | [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) | text localization |

`weights/GroundingDINO_SwinB_cfg.py` is included in the repository.
