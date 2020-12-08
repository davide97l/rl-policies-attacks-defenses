import argparse
import math
import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CnnDQN
from environment import atari_env
from utils import read_config

from torch.autograd import Variable

from adv_attacks.adversary import Adversary
from adv_attacks.gradient_method import FGSM
from adv_attacks.adv_model import PytorchModel
from adv_attacks.PGD import PGDAttack
from ibp import network_bounds

parser = argparse.ArgumentParser(description='A3C')


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
    '--load-path',
    default='trained_models/PongNoFrameskip-v4_robust.pt',
    metavar='LMD',
    help='path to trained model file')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--fgsm-video',
    type=float,
    default=None,
    metavar='FV',
    help='whether to to produce a video of the agent performing under FGSM attack with given epsilon')
parser.add_argument(
    '--pgd-video',
    type=float,
    default=None,
    metavar='PV',
    help='whether to to produce a video of the agent performing under PGD attack with given epsilon')
parser.add_argument('--video',
                    dest='video',
                    action='store_true',
                    help = 'saves a video of standard eval run of model')
parser.add_argument('--fgsm',
                    dest='fgsm',
                    action='store_true',
                    help = 'evaluate against fast gradient sign attack')
parser.add_argument('--pgd',
                   dest='pgd',
                   action='store_true',
                   help='evaluate against projected gradient descent attack')
parser.add_argument('--gwc',
                   dest='gwc',
                   action='store_true',
                   help='whether to evaluate worst possible(greedy) outcome under any epsilon bounded attack')
parser.add_argument('--action-pert',
                   dest='action_pert',
                   action='store_true',
                   help='whether to evaluate performance under action perturbations')
parser.add_argument('--acr',
                   dest='acr',
                   action='store_true',
                   help='whether to evaluate the action certification rate of an agent')
parser.add_argument('--nominal',
                   dest='nominal',
                   action='store_true',
                   help='evaluate the agents nominal performance without any adversaries')

parser.set_defaults(video=False, fgsm=False, pgd=False, gwc=False, action_pert=False, acr=False)


def record_game(curr_model, env, args):
    
    state = env.reset()
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            curr_model = curr_model.cuda()
    
    states = [state*255]
    episode_reward = 0
    
    with torch.no_grad():
        while True:
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            output = curr_model.forward(input_x)
            action = torch.argmax(output, dim=1)
            next_state, reward, done, info = env.step(action[0])
            
            episode_reward += reward
            state = next_state
            states.append(state*255)
            
            if done and not info:
                state = env.reset()
            elif info:
                env.reset()
                states = np.array(states)
                print(states.shape)
                return episode_reward, np.array(states, dtype=np.uint8)

            
def attack_eval(curr_model, env, args, epsilon=1e-4, attack_type='FGSM', record=False):
    assert attack_type in ('FGSM', 'PGD'), 'Invalid attack type'
    loss_func = torch.nn.CrossEntropyLoss()
    m = PytorchModel(curr_model, loss_func,(0, 1), channel_axis=1, nb_classes=env.action_space, device=args.gpu_id)
    
    if attack_type=='FGSM':
        attack = FGSM(m)
        attack_config = {"epsilons": [epsilon], 'steps': 1}
    elif attack_type == 'PGD':
        attack = PGDAttack(m)
        attack_config = {"epsilon": epsilon, "steps": 30, "relative_step_size":0.1}
        
    total_count = 0
    fooling_count = 0
    
    episode_reward = 0
    state = env.reset()
    if record:
        states = []
        
    while True:
        total_count += 1
        input_x = torch.FloatTensor(state).unsqueeze(0)
        if args.gpu_id >= 0:
            with torch.cuda.device(args.gpu_id):
                input_x = input_x.cuda()
        output = curr_model.forward(input_x)
        #print(output)
        action = torch.argmax(output, dim=1)
        inputs, labels= input_x.cpu().numpy(), action.cpu().numpy()
        adversary = Adversary(inputs, labels[0])
        adversary = attack(adversary, **attack_config)
        
        if adversary.is_successful():
            fooling_count += 1
            if record:
                states.append(adversary.adversarial_example[0]*255)
            next_state, reward, done, info = env.step(adversary.adversarial_label)
        else:
            if record:
                states.append(adversary.bad_adversarial_example[0]*255)
            next_state, reward, done, info = env.step(action[0])
        
        episode_reward += reward
        state = next_state
        if done and not info:
            state = env.reset()
        
        elif info:
            state = env.reset()
            print("[TEST_DATASET]: fooling_count={}, total_count={}, fooling_rate={:.3f}".format(
                fooling_count, total_count, float(fooling_count) / total_count))
            print('Reward under {} attack {}'.format(attack_type, episode_reward))
            if record:
                return episode_reward, np.array(states, dtype=np.uint8)
            else:
                return episode_reward


def eval_greedy_wc(curr_model, env, args, epsilon=1e-4):
    episode_reward = 0
    state = env.reset()

    with torch.no_grad():
        while True:
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            output = curr_model.forward(input_x)
            #print(output)

            upper, lower = network_bounds(curr_model.model, input_x, epsilon=epsilon)
            impossible = upper < torch.max(lower, dim=1)[0]
            #add a large number to ignore impossible ones, choose possible action with smallest q-value
            worst_case_action = torch.argmin(output+1e6*impossible, dim=1)
            next_state, reward, done, info = env.step(worst_case_action[0])
            episode_reward += reward
            state = next_state
            if done and not info:
                state = env.reset()
            elif info:
                state = env.reset()
                print('Worst case reward {}'.format(episode_reward))
                return episode_reward

def eval_action_pert(curr_model, env, args, epsilon=0.01):
    episode_reward = 0
    state = env.reset()

    with torch.no_grad():
        while True:
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            output = curr_model.forward(input_x)
            #print(output)
            if random.random() < epsilon:
                action = random.randint(0, output.shape[1]-1)
            else:
                action = torch.argmax(output[0])
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done and not info:
                state = env.reset()
            elif info:
                state = env.reset()
                print('Reward under {} action perturbation:{}'.format(epsilon, episode_reward))
                return episode_reward            

def eval_action_cert_rate(curr_model, env, args, epsilon=1e-4):
    episode_reward = 0
    state = env.reset()
    total = 0
    certified = 0
    with torch.no_grad():
        while True:
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            output = curr_model.forward(input_x)
            action = torch.argmax(output, dim=1)

            upper, lower = network_bounds(curr_model.model, input_x, epsilon=epsilon)
            #remove the action selected from calculations
            upper[:, action] = -1e10
            
            max_other = torch.max(upper, dim=1)[0]
            if lower[:, action] > max_other:
                certified += 1
            total += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done and not info:
                state = env.reset()
            elif info:
                state = env.reset()
                print('Reward:{}, action certification rate {:.4f}'.format(episode_reward, certified/total))
                return certified/total         
            

if __name__ == '__main__':
    args = parser.parse_args()
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
            if i in args.env:
                env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    model = CnnDQN(env.observation_space.shape[0], env.action_space)
    
    if args.gpu_id >= 0:
        weights = torch.load(args.load_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
        model.load_state_dict(weights)
        with torch.cuda.device(args.gpu_id):
            model.cuda()
    else:
        weights = torch.load(args.load_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        
    model.eval()
    
    
    save_name = (args.load_path.split('/')[-1]).split('.')[0]
    if not os.path.exists('videos'):
        os.mkdir('videos')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if not os.path.exists('figures/'+save_name):
        os.mkdir('figures/'+save_name)
            
    if args.video:
        reward, states = record_game(model, env, args)
        print(reward)

        width = env.observation_space.shape[1]
        height = env.observation_space.shape[2]
        FPS = 20
        fourcc = VideoWriter_fourcc(*'MP42')
        video = VideoWriter('videos/{}.avi'.format(save_name), fourcc, float(FPS), (width, height))

        states = np.transpose(np.repeat(states, 3, axis=1), (0, 2, 3, 1))
        for frame in states:
            video.write(frame)
        video.release()
    
    if args.fgsm_video:
        reward, states = attack_eval(model, env, args, args.fgsm_video, 'FGSM', record=True)
        print(reward)

        width = env.observation_space.shape[1]
        height = env.observation_space.shape[2]
        FPS = 20
        fourcc = VideoWriter_fourcc(*'MP42')
        video = VideoWriter('videos/{}_fgsm_{}.avi'.format(save_name, args.fgsm_video), fourcc, float(FPS), (width, height))

        states = np.transpose(np.repeat(states, 3, axis=1), (0, 2, 3, 1))
        for frame in states:
            video.write(frame)
        video.release()
    
    if args.pgd_video:
        reward, states = attack_eval(model, env, args, args.pgd_video, 'PGD',record=True)
        print(reward)

        width = env.observation_space.shape[1]
        height = env.observation_space.shape[2]
        FPS = 20
        fourcc = VideoWriter_fourcc(*'MP42')
        video = VideoWriter('videos/{}_pgd_{}.avi'.format(save_name, args.pgd_video), fourcc, float(FPS), (width, height))

        states = np.transpose(np.repeat(states, 3, axis=1), (0, 2, 3, 1))
        for frame in states:
            video.write(frame)
        video.release()
        
    epsilons = [0.3/255, 1/255, 3/255, 8/255]
    if args.fgsm:
        np.save('figures/{}/fgsm_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(20):
                reward = attack_eval(model, env, args, epsilon, 'FGSM')
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        rewards = np.sort(rewards, axis=1)
        plt.plot(epsilons, np.mean(rewards, axis=1), label='mean')
        plt.fill_between(epsilons, rewards[:, -1], rewards[:, 0], alpha=0.2, label='interval')
        plt.legend()
        plt.xlabel('l-inf perturbation')
        plt.ylabel('reward')
        plt.xscale('log')
        np.save('figures/{}/fgsm.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/fgsm.png'.format(save_name))
        plt.close()
        
    if args.pgd:
        np.save('figures/{}/pgd_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(20):
                reward = attack_eval(model, env, args, epsilon, 'PGD')
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        rewards = np.sort(rewards, axis=1)
        plt.plot(epsilons, np.mean(rewards, axis=1), label='mean')
        plt.fill_between(epsilons, rewards[:, -1], rewards[:, 0], alpha=0.2, label='interval')
        plt.legend()
        plt.xlabel('l-inf perturbation')
        plt.ylabel('reward')
        plt.xscale('log')
        np.save('figures/{}/pgd.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/pgd.png'.format(save_name))
        plt.close()
        
    if args.gwc:
        np.save('figures/{}/greedy_wc_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(20):
                reward = eval_greedy_wc(model, env, args, epsilon)
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        rewards = np.sort(rewards, axis=1)
        plt.plot(epsilons, np.mean(rewards, axis=1), label=' mean')
        plt.fill_between(epsilons, rewards[:, -1], rewards[:, 0], alpha=0.2, label='interval')
        plt.legend()
        plt.xlabel('l-inf perturbation')
        plt.ylabel('reward')
        plt.xscale('log')
        np.save('figures/{}/greedy_wc.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/greedy_wc.png'.format(save_name))
        plt.close()
        
    if args.acr:
        np.save('figures/{}/acr_epsilons.npy'.format(save_name), epsilons)
        rates = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rates = []
            for i in range(20):
                rate = eval_action_cert_rate(model, env, args, epsilon)
                curr_rates.append(rate)
            rates.append(curr_rates)
        
        rates = np.sort(rates, axis=1)
        plt.plot(epsilons, np.mean(rates, axis=1), label='mean')
        plt.fill_between(epsilons, rates[:, -1], rates[:, 0], alpha=0.2, label='interval')
        plt.legend()
        plt.xlabel('l-inf perturbation')
        plt.ylabel('Action certification rate')
        plt.xscale('log')
        np.save('figures/{}/acr.npy'.format(save_name), rates)
        plt.savefig('figures/{}/acr.png'.format(save_name))
        plt.close()
    
    if args.action_pert:
        epsilons = [0.01, 0.02, 0.05, 0.1]
        np.save('figures/{}/action_pert_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(20):
                reward = eval_action_pert(model, env, args, epsilon)
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        rewards = np.sort(rewards, axis=1)
        plt.plot(epsilons, np.mean(rewards, axis=1), label='mean reward')
        plt.fill_between(epsilons, rewards[:, -1], rewards[:, 0], alpha=0.2, label='reward interval')
        plt.legend()
        plt.xlabel('action perturbation')
        plt.ylabel('reward')
        plt.xscale('log')
        np.save('figures/{}/action_pert.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/action_pert.png'.format(save_name))
        plt.close()
        
    if args.nominal:
        curr_rewards = []
        for i in range(20):
            reward = eval_action_pert(model, env, args, epsilon=0)
            curr_rewards.append(reward)
        rewards = np.sort(curr_rewards)
        plt.hist(rewards, bins=10)
        plt.title('Nominal mean reward:{:.1f}'.format(np.mean(rewards)))
        np.save('figures/{}/nominal.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/nominal.png'.format(save_name))
        plt.close()
    