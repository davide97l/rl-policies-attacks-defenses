import os

import numpy as np
import matplotlib.pyplot as plt


def _moving_avg(data, window=1000):
    cum_data = np.cumsum(data)
    return (cum_data[window:] - cum_data[:-window]) / window

def plot(frame_idx, rewards, losses, standard_losses, worst_case_losses, args, start_time):
    f = plt.figure(figsize=(20,5))
    ax = f.add_subplot(131)
    ax.title.set_text('frame {}. last 10 avg reward: {}'.format(frame_idx, np.mean(rewards[-10:])))
    ax.plot(_moving_avg(rewards, window=10), label='training reward')
    ax.legend()
    
    
    ax2 = f.add_subplot(132)
    ax2.title.set_text('Average of loss of last 1000 steps')
    ax2.plot(_moving_avg(losses), label='loss')
    ax2.plot(_moving_avg(standard_losses), label='standard')
    ax2.plot(_moving_avg(worst_case_losses), label='worst_case')
    
    ax2.set_yscale('log')
    ax2.legend()
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig('figures/{}_training_{}.png'.format(args.env, start_time))
    plt.close()