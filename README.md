# SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild (ECCV 2024)


This is the official repository for the paper "SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild" (ECCV 2024).

## Evaluation Metric - Semantic Tracking-mAP (ST-mAP)

The code of Semantic Tracking-mAP (ST-mAP) is developed based on the MOT tracking toolkit [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## How to use
for example, the dataset folder is "data", the prediction of your method is in a folder "trackers", you want to save the evaluation result in the folder "output", your trackier name is "bytetrack", "deepsort" and "sort", the tracker display name is also "bytetrack", "deepsort" and "sort"
run the code as follows:
cd trackeval
python run_stmap.py \
    --GT_FOLDER data \
    --TRACKERS_FOLDER trackers \
    --OUTPUT_FOLDER output \
    --TRACKERS_TO_EVAL bytetrack deepsort sort \
    --TRACKER_DISPLAY_NAMES bytetrack deepsort sort