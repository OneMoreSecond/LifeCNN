import os
import json
import argparse

import data
import model

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'test'])
parser.add_argument('model_dir', help='the directory storing models')
parser.add_argument('--model_config', help='if not specified, use model_dir/model_config.json')
parser.add_argument('--train_config', default='config/default/train_config.json')
parser.add_argument('--test_config', default='config/default/test_config.json')

args = parser.parse_args()

model_config_path = os.path.join(args.model_dir, 'model_config.json')
if args.model_config_path is not None:
    model_config_path = args.model_config_path

if args.mode == 'train':
    pass
elif args.mode == 'test':
    pass
else:
    assert False, 'invalid mode'
