# DEIM Wrapper

A wrapper for the [DEIM](https://github.com/ShihuaHuang95/DEIM) model.

## Installation

Install pixi

```
curl -fsSL https://pixi.sh/install.sh | bash
```

## Clone the repo

Clone with submodules

```
git clone --recurse-submodules https://github.com/dnth/deim-wrapper.git
```

## Edit Config

Edit the `cfg_nano.yml` file to point to the data you want to train on.

## Run Training

Nano model
```
pixi run train_nano
```

Small model
```
pixi run train_small
```

## Export ONNX

Export ONNX with specific model and checkpoint
```
pixi run export_onnx -c cfg_small.yml -r outputs/deim_hgnetv2_s_coco/last.pth
```

## Live Inference

```
pixi run live_inference --onnx outputs/deim_hgnetv2_s_coco/last.onnx --webcam --class-names outputs/deim_hgnetv2_s_coco/classes.txt
```

GPU

```
pixi run -e gpu-env live_inference --onnx outputs/deim_hgnetv2_s_coco/last.onnx --webcam --class-names outputs/deim_hgnetv2_s_coco/classes.txt --gpu 
```

## Gradio Demo

```
pixi run launch_demo
```

## Plot Metrics

```
pixi run plot_metrics --log-file outputs/deim_hgnetv2_n_coco/log.txt
```
