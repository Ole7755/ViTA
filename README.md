# ViTA Eye-State Classification

Minimal PyTorch framework for binary eye-state classification: `closed/sleepy=0`, `open/awake=1`.

Local `dataset/` files are kept for CPU smoke tests only. Training and benchmark runs should be executed on the remote Linux machine.

## Structure

```text
configs/              YAML experiment configs
vita/data.py          ImageFolder-style dataset loading
vita/models.py        timm model factory
vita/train.py         training entry point
vita/eval.py          checkpoint evaluation
vita/benchmark.py     CPU/GPU inference FPS benchmark
```

## Remote Setup

```bash
pip install -r requirements.txt
```

## Dataset Roots

`data.datasets[*].root` must point to the directory that directly contains `train/`, `val/`, and `test/`.

| Dataset | Remote root | Local smoke-test root | Classes |
|---|---|---|---|
| MRL Eye Dataset | `/home/featurize/data/data` | `/Users/ole/Documents/ViTA/dataset/mrl` | `sleepy=0`, `awake=1` |
| open-closed-eyes-dataset | `/home/featurize/data` | `/Users/ole/Documents/ViTA/dataset/oceye` | `closed=0`, `open=1` |

The two label vocabularies are treated as equivalent: `sleepy == closed`, `awake == open`.

Expected dataset layout:

```text
root/
  train/<class_name>/*.png
  val/<class_name>/*.png
  test/<class_name>/*.png
```

## Commands

Train:

```bash
python -m vita.train --config configs/mrl_swin_t.yaml
```

Evaluate:

```bash
python -m vita.eval --config configs/mrl_swin_t.yaml --checkpoint outputs/mrl_swin_t_scratch/best.pt --split test
```

Benchmark GPU:

```bash
python -m vita.benchmark --config configs/mrl_swin_t.yaml --checkpoint outputs/mrl_swin_t_scratch/best.pt --device cuda --batch-size 1
```

Benchmark CPU:

```bash
python -m vita.benchmark --config configs/mrl_swin_t.yaml --checkpoint outputs/mrl_swin_t_scratch/best.pt --device cpu --batch-size 1
```

For Swin-S/B or CNN baselines, copy the config and change `model.name`, for example:

```yaml
model:
  name: swin_small_patch4_window7_224
  pretrained: true
```

Common baseline names: `resnet18`, `mobilenetv3_small_100`.

The default MRL Swin-T config trains from scratch (`pretrained: false`). For fine-tuning, set `pretrained: true`, reduce `epochs`, and use a smaller learning rate such as `0.00005`.
