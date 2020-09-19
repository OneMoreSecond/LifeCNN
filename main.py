import os
import json
import argparse

import data
import model

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'test'])
parser.add_argument('model_dir')

args = parser.parse_args()

config_path = os.path.join(args.model_dir, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
