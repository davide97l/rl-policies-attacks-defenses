from __future__ import print_function, division
import os
import argparse
import torch
from environment import atari_env
from utils import read_config
from model import CnnDQN
from train import train
#from gym.configuration import undo_logger_setup
import time

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.000125,
    metavar='LR',
    help='learning rate (default: 0.000125)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='S',
    help='random seed (default: None)')
parser.add_argument(
    '--total-frames',
    type=int,
    default=6000000,
    metavar='TS',
    help='How many frames to train with')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='PongNoFrameskip-v4',
    metavar='ENV',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='optimizer to use, one of {Adam, RMSprop}')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=False,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--worse-bound',
    default=True,
    help='if this is selected worst case loss uses bound that is further away from mean')

parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--kappa-end',
    type=float,
    default=0.5,
    metavar='SR',
    help='final value of the variable controlling importance of standard loss (default: 0.5)')
parser.add_argument('--robust',
                   dest='robust',
                   action='store_true',
                   help='train the model to be verifiably robust')
parser.add_argument(
    '--load-path',
    type=str,
    default=None,
    help='Path to load a model from. By default starts training a new model')

parser.add_argument(
    '--attack-epsilon-end',
    type=float,
    default=1/255,
    metavar='EPS',
    help='max size of perturbation trained on')
parser.add_argument(
    '--attack-epsilon-schedule',
    type=int,
    default=3000000,
    help='The frame by which to reach final perturbation')
parser.add_argument(
    '--exp-epsilon-end',
    type=float,
    default=0.01,
    help='for epsilon-greedy exploration')
parser.add_argument(
    '--exp-epsilon-decay',
    type=int,
    default=500000,
    help='controls linear decay of exploration epsilon')
parser.add_argument(
    '--replay-initial',
    type=int,
    default=50000,
    help='How many frames of experience to collect before starting to learn')
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    help='Batch size for updating agent')
parser.add_argument(
    '--updates-per-frame',
    type=int,
    default=32,
    help='How many gradient updates per new frame')
parser.add_argument(
    '--buffer-size',
    type=int,
    default=200000,
    help='How frames to store in replay buffer')


parser.set_defaults(robust=False)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
        if args.gpu_id>=0:
            torch.cuda.manual_seed(args.seed)
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    curr_model = CnnDQN(env.observation_space.shape[0], env.action_space)
    
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    
    if args.load_path:
        saved_state = torch.load(
            args.load_path,
            map_location=lambda storage, loc: storage)
        curr_model.load_state_dict(saved_state)
        
    target_model = CnnDQN(env.observation_space.shape[0], env.action_space)
    target_model.load_state_dict(curr_model.state_dict())
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            curr_model.cuda()
            target_model.cuda()
            
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(curr_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(curr_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    
    train(curr_model, target_model, env, optimizer, args)

