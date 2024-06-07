# AICUP Baseline: BoT-SORT (TEAM_5117)

The code is based on [AICUP Baseline: BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT).

## Installation

The Installation is same as the baseline.

```
conda create -n botsort python=3.7
conda activate botsort

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy
pip install -r requirements.txt
pip install cython; pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
pip install cython_bbox
pip install faiss-gpu

cd BOT_SORT_PATH
```

## Prepare ReID Dataset

```
python fast_reid/datasets/generate_AICUP_patches.py \
--data_path AI_CUP_PATH \
--save_path REID_PATH \
--train_ratio 1
```

## Prepare YOLOv7 Dataset

```
python yolov7/tools/AICUP_to_YOLOv7.py \
--AICUP_dir AI_CUP_PATH \
--YOLOv7_dir YOLO_PATH \
--train_ratio 0.9
```

## Train YOLOv7

```
export OMP_NUM_THREADS=4

python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 yolov7/train_aux.py \
--weights PRETRAINED_WEIGHTS_PATH \
--cfg yolov7/cfg/training/yolov7-e6e_aicup.yaml \
--data yolov7/data/aicup.yaml \
--hyp yolov7/data/hyp_aicup.scratch.p6.yaml \
--epochs 40 \
--batch 24 \
--img 1280 1280 \
--device 0,1 \
--single-cls \
--adam \
--sync-bn \
--workers 2 \
--name yolov7-e6e_aicup \
--freeze 112
```

## Train ReID

```
export FASTREID_DATASETS=REID_PATH

python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/sbs_S101.yml --num-gpus 2
```

## Track

```
bash tools/track_all_timestamps.sh \
--weights runs/train/yolov7-e6e_aicup/weights/best.pt \
--source-dir AI_CUP_PATH/images \
--device 0 \
--fast-reid-config fast_reid/configs/AICUP/sbs_S101.yml \
--fast-reid-weights logs/AICUP/sbs_S101/model_final.pth

python tools/data_sorting.py --output_path runs/track/sbs_S101/train
```

## Postprocessing

```
python tools/postprocessing.py \
--input_path runs/track/sbs_S101/train/txts \
--output_path runs/track/sbs_S101/train/txts_postprocessing \
--percentage 5
```

## Convert Ground Truth

```
python tools/datasets/AICUP_to_MOT15.py \
--AICUP_dir AI_CUP_PATH \
--MOT15_dir MOT15_PATH
```

## Evaluate

```
python tools/evaluate.py \
--gt_dir MOT15_PATH \
--ts_dir runs/track/sbs_S101/train/txts_postprocessing
```
