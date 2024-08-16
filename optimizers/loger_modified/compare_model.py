
FILE_ID = '100' # job rs top10 test



import os
import torch
import random
from collections.abc import Iterable
from tqdm import tqdm
import pandas as pd
import math
import numpy as np
import pickle
from lib.log import Logger
from lib.timer import timer
from lib.cache import HashCache
from model.dqn import DeepQNet
from model import explorer

from core.oracle import oracle_database
USE_ORACLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_LATENCY = True
SEED = 0

cache_manager = HashCache()
CACHE_FILE = 'latency_cache.pkl'


def validate_top10(beam_width=1, epochs=400):
    use_beam = beam_width >= 1
    start_epoch = 0
    fcx_df = []

    checkpoint_file = 'temps/ckpts/1.{}.checkpoint.pkl'
    dic0 = torch.load(checkpoint_file.format(0), map_location=device)
    dic1 = torch.load(checkpoint_file.format(1), map_location=device)
    dic2 = torch.load(checkpoint_file.format(2), map_location=device)
    print(str(dic0['model']['step1']) == str(dic1['model']['step1']))
    print(str(dic0['model']['step2']) == str(dic1['model']['step2']))
    print(str(dic0['model']['tail']) == str(dic1['model']['tail']))
    print("33")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', nargs=2, type=str, default=['dataset/job_rs_train', 'dataset/job_rs_test'],
                        help='Training and testing dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Total epochs.')
    parser.add_argument('-F', '--id', type=str, default=100,
                        help='File ID.')
    parser.add_argument('-b', '--beam', type=int, default=4,
                        help='Beam width. A beam width less than 1 indicates simple epsilon-greedy policy.')
    parser.add_argument('-s', '--switches', type=int, default=4,
                        help='Branch amount of join methods.')
    parser.add_argument('-l', '--layers', type=int, default=1,
                        help='Number of graph transformer layer in the model.')
    parser.add_argument('-w', '--weight', type=float, default=0.1,
                        help='The weight of reward weighting.')
    parser.add_argument('-N', '--no-restricted-operator', action='store_true',
                        help='Not to use restricted operators.')
    parser.add_argument('--oracle', type=str, default=None, # database/password@localhost:1521
                        help='To use oracle with given connection settings.')
    parser.add_argument('--cache-name', type=str, default=None,
                        help='Cache file name.')
    parser.add_argument('--bushy', action='store_true',
                        help='To use bushy search space.')
    parser.add_argument('--log-cap', type=float, default=1.0,
                        help='Cap of log transformation.')
    parser.add_argument('--warm-up', type=int, default=None,
                        help='To warm up the database with specific iterations.')
    parser.add_argument('--no-exploration', action='store_true',
                        help='To use the original beam search.')
    parser.add_argument('--no-expert-initialization', action='store_true',
                        help='To discard initializing the replay memory with expert knowledge.')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
                        help='Pretrained checkpoint.')
    parser.add_argument('-S', '--seed', type=int, default=3407,
                        help='Random seed.')
    parser.add_argument('-D', '--database', type=str, default='imdb',
                        help='PostgreSQL database.')
    parser.add_argument('-U', '--user', type=str, default='postgres',
                        help='PostgreSQL user.')
    parser.add_argument('-P', '--password', type=str, default=None,
                        help='PostgreSQL user password.')
    parser.add_argument('--port', type=int, default=5432,
                        help='PostgreSQL port.')

    args = parser.parse_args()

    args_dict = vars(args)

    FILE_ID = args.id

    validate_top10(beam_width=args.beam, epochs=args.epochs)
