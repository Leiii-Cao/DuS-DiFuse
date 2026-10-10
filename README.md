# [IEEE TPAMI 2026] Towards Heterogeneous-Degradation-Robust Image Fusion with Controllable Generative Modulation

Official PyTorch implementation of **DuS-DiFuse**.  Conference-version code: [Text-DiFuse](https://github.com/Leiii-Cao/Text-DiFuse).

**Lei Cao, Hao Zhang, Peng Zhang, Daiguo Zhou, and Jiayi Ma**

IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026

### [Paper](https://doi.org/10.1109/TPAMI.2026.3724595) | [Code](https://github.com/Leiii-Cao/DuS-DiFuse) | [Data](data/README.md)

## 🛠️ Installation

```bash
git clone https://github.com/Leiii-Cao/DuS-DiFuse.git
cd DuS-DiFuse
conda env create -f environment.yml
conda activate dus-difuse
```

The environment uses Python 3.10 and installs `requirements.txt`.

To compile the optional GroundingDINO CUDA/C++ operator:

```bash
pip install --no-build-isolation -e dus_difuse/target_att/GroundingDINO
```

## 📦 Checkpoints

Download the project checkpoints from [Google Drive](https://drive.google.com/drive/folders/1SbWD1RLpYU7dkKKgLZUhu8GUP5j-bBf1)
and place them in `weights/`:

```text
weights/
├── mda.pth
├── sde.pt
├── GFCM.pt
├── DRFM.pt
└── Generative_Control.pt
```

See [weights/README.md](weights/README.md) for their uses and the third-party
model links.

## 🚀 Inference

Paired examples are provided in `data/example/source_1` and
`data/example/source_2`.

```bash
bash test.sh
```

Set `GENERATIVE_MODULATION` in `test.sh` to enable or disable Generative
Modulation, and use `TEXT_PROMPT` to specify the target content (leave it empty
for global modulation).

## 🗂️ Data

Download the prepared GDFusion data from
[Google Drive](https://drive.google.com/drive/folders/1p96_NS02-k4MfyWxAssHyTPRgawP3Fe3).
See [data/README.md](data/README.md) for examples, source datasets, and the
expected directory layout.

## 🔥 Training

Train the fusion modules in order:

```bash
bash scripts/train_mda.sh
bash scripts/train_degradation_decoupling.sh
bash scripts/train_gfcm.sh
bash scripts/train_drfm.sh
```

Train Generative Modulation separately:

```bash
bash scripts/train_generative_modulation.sh
```

Edit the settings at the top of each shell script. See
[scripts/README.md](scripts/README.md) for stage inputs, batch sizes, and output
paths.

## 🌧️ Degradation Simulation

Edit `scripts/run_simulate_degradation.sh`, then run:

```bash
bash scripts/run_simulate_degradation.sh
```

One degradation name applies that type to the whole folder. Multiple names
randomly select one type for each image.

## 📝 Citation

```bibtex
@article{cao2026towards,
  title={Towards Heterogeneous-Degradation-Robust Image Fusion with Controllable Generative Modulation},
  author={Cao, Lei and Zhang, Hao and Zhang, Peng and Zhou, Daiguo and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```

## 🙏 Acknowledgements

This repository uses code or models from
[DiffBIR](https://github.com/XPixelGroup/DiffBIR),
[Stable Diffusion](https://github.com/Stability-AI/stablediffusion),
[OpenCLIP](https://github.com/mlfoundations/open_clip),
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and
[Segment Anything](https://github.com/facebookresearch/segment-anything).

## 📬 Contact

whu.caolei@whu.edu.cn
