
""" 
Author: Wang Pengfei

run_stmap.py

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'trackers/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['adult', 'aircraft', 'antelope', 'baby', 'baby_seat', 'baby_walker', 'backpack', 'ball/sports_ball', 'bat', 'bear', 'bench', 'bicycle', 'bird', 'bottle', 'bread', 'bus/truck', 'cake', 'camel', 'camera', 'car', 'cat', 'cattle/cow', 'cellphone', 'chair', 'chicken', 'child', 'crab', 'crocodile', 'cup', 'dish', 'dog', 'duck', 'electric_fan', 'elephant', 'faucet', 'fish', 'fox', 'frisbee', 'fruits', 'giant_panda', 'guitar', 'hamster/rat', 'handbag', 'horse', 'kangaroo', 'laptop', 'leopard', 'lion', 'lizard', 'microwave', 'monkey', 'motorcycle', 'oven', 'penguin', 'piano', 'pig', 'rabbit', 'racket', 'red_panda', 'refrigerator', 'scooter', 'screen/monitor', 'sheep/goat', 'sink', 'skateboard', 'ski', 'snake', 'snowboard', 'sofa', 'squirrel', 'stool', 'stop_sign', 'suitcase', 'surfboard', 'table', 'tiger', 'toilet', 'toy', 'traffic_light', 'train', 'turtle', 'vegetables', 'watercraft', 'whale', 'zebra'],
        'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'val', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'track_results',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEARTR', 'Identity', 'VACE']
"""

import sys
import os
import json
import argparse
from multiprocessing import freeze_support
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

parser = argparse.ArgumentParser("EVENT EVAL")
parser.add_argument("--category_info", type=str, default="data/category_info_new_split", help="path to category info folder")
parser.add_argument("--GT_FOLDER", type=str, default="data", help="path to gt data")
parser.add_argument("--TRACKERS_FOLDER", type=str, default="trackers")
parser.add_argument("--OUTPUT_FOLDER", type=str, default="output", help="path to save eval results")
parser.add_argument("--TRACKERS_TO_EVAL", nargs='+', default=['bytetrack'], help="Filenames of trackers to eval (if None, all in folder) ['bytetrack', 'deepsort', 'sort']")
parser.add_argument("--SPLIT_TO_EVAL", type=str, default="test", choices=['train', 'val', 'test', 'all'], help="Valid: 'train', 'val', 'test', 'all'")
parser.add_argument("--INPUT_AS_ZIP", type=bool, default=False, help="Whether tracker input files are zipped")
parser.add_argument("--PRINT_CONFIG", type=bool, default=True, help="Whether to print current config")
parser.add_argument("--DO_PREPROC", type=bool, default=True, help="Whether to perform preprocessing (never done for MOT15)")
parser.add_argument("--TRACKER_SUB_FOLDER", type=str, default="", help="Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER")
parser.add_argument("--OUTPUT_SUB_FOLDER", type=str, default="", help="")
parser.add_argument("--TRACKER_DISPLAY_NAMES", nargs='+', default=['bytetrack'], help="")
parser.add_argument("--SEQMAP_FOLDER", type=str, default=None, help="")
parser.add_argument("--SEQMAP_FILE", type=str, default=None, help="")
parser.add_argument("--SEQ_INFO", type=str, default=None, help="")
parser.add_argument("--GT_LOC_FORMAT", type=str, default=None, help="")
parser.add_argument("--SKIP_SPLIT_FOL", type=bool, default=False, help="")
parser.add_argument("--USE_PARALLEL", default=None)
parser.add_argument("--NUM_PARALLEL_CORES", default=None)
parser.add_argument("--BREAK_ON_ERROR", default=None)
parser.add_argument("--RETURN_ON_ERROR", default=None)
parser.add_argument("--LOG_ON_ERROR", default=None)
parser.add_argument("--PRINT_RESULTS", default=None)
parser.add_argument("--PRINT_ONLY_COMBINED", default=None)
# parser.add_argument("--PRINT_CONFIG", default=None)
parser.add_argument("--TIME_PROGRESS", default=None)
parser.add_argument("--DISPLAY_LESS_PROGRESS", default=None)
parser.add_argument("--OUTPUT_SUMMARY", default=True)
parser.add_argument("--OUTPUT_EMPTY_CLASSES", default=None)
parser.add_argument("--OUTPUT_DETAILED", default=None)
parser.add_argument("--PLOT_CURVES", default=None)
parser.add_argument("--CLASSES_TO_EVAL", default=['adult', 'baby', 'child', 'toy', 'dog', 'guitar'])
parser.add_argument("--VALID_CLASSES", default=None)



if __name__ == '__main__':
    freeze_support()

    # args = parser.parse_args()
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.VLOGBox.get_default_dataset_config()
    # default_metrics_config = {'METRICS': ['HOTA', 'CLEARTR', 'Identity', 'STMAP', 'TrackMAP']} # , 'THRESHOLD': 0.5
    default_metrics_config = {'METRICS': ['HOTA', 'CLEARTR', 'Identity','STMAP']} # , 'THRESHOLD': 0.5
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    # parser = argparse.ArgumentParser()
    # for setting in config.keys():
    #     if type(config[setting]) == list or type(config[setting]) == type(None):
    #         parser.add_argument("--" + setting, nargs='+')
    #     else:
    #         parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    # nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args['OUTPUT_FOLDER'] = os.path.join(args['OUTPUT_FOLDER'])
    if os.path.exists(args['category_info']):
        with open(os.path.join(args['category_info'], 'category_info.json')) as f:
            category_json = json.load(f)
        f.close()
        object_classes = []
        with open(os.path.join(args['category_info'], 'object_classes.txt')) as f:
            for line in f.readlines():
                object_classes.append(line.strip())
        if args['CLASSES_TO_EVAL'] is None:
            args['CLASSES_TO_EVAL'] = object_classes
        args['VALID_CLASSES'] = object_classes
        args['CLASS_NAME_TO_CLASS_ID'] = category_json['object_category_to_index']
    del args['category_info']
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == True:
                    x = True
                elif args[setting] == False:
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.SemTrack(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.STMAP, trackeval.metrics.CLEARTR, trackeval.metrics.Identity,
                   trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config.copy()))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
