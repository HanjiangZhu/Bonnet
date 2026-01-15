# Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans

Bonnet is an ultra-fast whole-body bone segmentation pipeline for CT scans. It runs in seconds per scan on a single commodity GPU while maintaining reliable segmentation quality across different datasets.

## Train (and evaluate)

1. Set dataset / output paths and other options in:

- `Bonnet/conf/config_eva.yaml`

1. Run:

```
python main.py
```

## Evaluate only (inference)

1. Open:

- `Bonnet/conf/eval/eval_on_test.yaml`

1. Set:

```
eval_only: True
```

1. Run:

```
python main.py
```

