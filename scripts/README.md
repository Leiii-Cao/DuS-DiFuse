# 🔥 Training

Run all commands from the repository root with the `dus-difuse` environment
activated.

## 📌 Training Order

```bash
bash scripts/train_mda.sh
bash scripts/train_degradation_decoupling.sh
bash scripts/train_gfcm.sh
bash scripts/train_drfm.sh
```

Generative Modulation is trained separately:

```bash
bash scripts/train_generative_modulation.sh
```

## ⚙️ Stage Settings

| Stage | Initialization | Trainable module | GPUs | Global batch | Save every | Output |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MDA | OpenCLIP ViT-H-14 | MDA encoder and heads | 1 | 8 | 10 epochs | `experiments/mda/` |
| SDE | SD2.1 + MDA | dual-stream UNet | 4 | 64 | 10,000 steps | `experiments/sde/` |
| GFCM | SD2.1 + MDA + SDE | GFCM | 4 | 16 | 1,000 steps | `experiments/gfcm/` |
| DRFM | SD2.1 + MDA + SDE + GFCM | DRFM | 4 | 4 | 10,000 steps | `experiments/drfm/` |
| Generative Modulation | SD2.1 | ControlNet | 4 | 32 effective | 10,000 steps | `experiments/generation/` |

Generative Modulation uses 2 samples per GPU and 4 gradient-accumulation
steps.

## 📊 Outputs

```text
experiments/mda/
├── events.out.tfevents.*
└── checkpoints/
    ├── best.pth
    ├── latest.pth
    └── mda.pth

experiments/<sde|gfcm|drfm|generation>/
├── events.out.tfevents.*
└── checkpoints/
    ├── <step>.pt
    └── final.pt
```

```bash
tensorboard --logdir experiments --port 6006
```
