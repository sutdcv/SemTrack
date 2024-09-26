# SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild (ECCV 2024)


This is the official repository for the paper "SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild" (ECCV 2024).

## Project Page
[Project Page](https://sutdcv.github.io/SemTrack/)

## Paper
[ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3555_ECCV_2024_paper.php)

## Dataset
Please download our dataset and toolkit [here](https://forms.gle/GsvB5EwZkJdUjA9Q8).

## Evaluation
The evaluation code is available at [Evaluation](https://github.com/sutdcv/SemTrack/tree/eval).

## Abstract
Knowing merely where the target is located is not sufficient for many real-life scenarios. In contrast, capturing rich details about the tracked target via its semantic trajectory, i.e. who/what this target is interacting with and when, where, and how they are interacting over time, is especially crucial and beneficial for various applications (e.g., customer analytics, public safety). We term such tracking as Semantic Tracking and define it as tracking the target based on the user's input and then, most importantly, capturing the semantic trajectory of this target. Acquiring such information can have significant impacts on sales, public safety, etc. However, currently, there is no dataset for such comprehensive tracking of the target. To address this gap, we create SemTrack, a large and comprehensive dataset containing annotations of the target's semantic trajectory. The dataset contains 6.7 million frames from 6961 videos, covering a wide range of 52 different interaction classes with 115 different object classes spanning 10 different supercategories in 12 types of different scenes, including both indoor and outdoor environments. We also propose SemTracker, a simple and effective method, and incorporate a meta-learning approach to better handle the challenges of this task. Our dataset and code can be found at [Project Page](https://sutdcv.github.io/SemTrack/).

![demo](image/Teaser-SemTrack.png "Teaser-SemTrack")

## Citation
If you use this dataset in your research, please cite our paper:
```
@inproceedings{wang2024semtrack,
title={SemTrack: A Large Scale Dataset for Semantic Tracking in the Wild},
author={Wang, Pengfei and Hui, Xiaofei and Wu, Jing and Yang, Zile and Ong, Kian Eng and Zhao, Xinge and Lu, Beijia and Huang, Dezhao and Ling, Evan and Chen, Weiling and Ma, Keng Teck and Hur, Minhoe and Liu, Jun},
booktitle={ECCV},
year={2024}
}
```
