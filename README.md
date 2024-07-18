# SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild (ECCV 2024)


This is the official repository for the paper "SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild" (ECCV 2024).

## Evaluation Metric - Semantic Tracking-mAP (ST-mAP)

The code of Semantic Tracking-mAP (ST-mAP) is developed based on the MOT tracking toolkit [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## How to use
To evaluate the performance of your tracker, you need to follow these steps:

1. For formatting of the predictions and groundtruth, we follow the data format of TAO dataset. We provide our subset groundtruth and predictions (SemTracker + Meta learning) in folder `results`.
2. To run the evaluation code, you can use the following command (`run_eval.sh`):

```bash
python run_eval.py \
    --GT_FOLDER results/gt \
    --TRACKERS_FOLDER results/predictions \
    --OUTPUT_FOLDER output \
    --TRACKERS_TO_EVAL bytetrack \
    --TRACKER_DISPLAY_NAMES bytetrack
```

where `<GT_FOLDER>` is the path to the groundtruth folder, `<TRACKERS_FOLDER>` is the path to the predictions folder, `<OUTPUT_FOLDER>` is the path to the predictions folder, `<TRACKERS_TO_EVAL>` is the filenames of trackers to eval, and `<TRACKER_DISPLAY_NAMES>` is the name of the track to display.

This should generate a folder named `output/bytetrack`, within it the results of the evaluation.

## Citation

If you use this dataset in your research, please cite the following paper:

```bibtex
@inproceedings{wang2024semtrack,
title={SemTrack: A Large Scale Dataset for Semantic Tracking in the Wild},
author={Wang, Pengfei and Hui, Xiaofei and Wu, Jing and Yang, Zile and Ong, Kian Eng and Zhao, Xinge and Lu, Beijia and Huang, Dezhao and Ling, Evan and Chen, Weiling and Ma, Keng Teck and Hur, Minhoe and Liu, Jun},
booktitle={ECCV},
year={2024}
}
```