import os
import json
import argparse

import data
import model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'test'])
parser.add_argument('model_dir', help='the directory storing models')
parser.add_argument('--model_config', help='if not specified, use model_dir/model_config.json')
parser.add_argument('--train_config', help='if not specified, use model_dir/train_config.json')
parser.add_argument('--eval_config', default='config/default/eval_config.json')

args = parser.parse_args()

model_config_path = os.path.join(args.model_dir, 'model_config.json')
if args.model_config is not None:
    model_config_path = args.model_config
model_config = utils.load_json(model_config_path)
game_step = model_config['game_step']

eval_config = utils.load_json(args.eval_config)

if args.mode == 'train':
    train_config_path = os.path.join(args.model_dir, 'train_config.json')
    if args.train_config is not None:
        train_config_path = args.train_config
    train_config = utils.load_json(train_config_path)

    the_model = model.MinimalModel(**model_config)
    train_data = data.RandomDataset(game_step, **train_config['data'])
    valid_datasets = [ ('random_data', data.RandomDataset(game_step, **eval_config)) ]
    if game_step == 1:
        valid_datasets.append( ('minimal_data', data.MinimalDataset()) )

    utils.fit(the_model, train_data, valid_datasets, **train_config['schedule'])

elif args.mode == 'test':
    pass
else:
    assert False, 'invalid mode'
