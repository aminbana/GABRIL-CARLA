import os

import random
from tqdm import tqdm
from gaze.gaze_to_mask import GazeToMask
import numpy as np
import torch
import matplotlib.pyplot as plt


Task_to_Route = {
    'Mixed_': {'train':[(r,s) for r in [24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258] for s in range(200, 220)], # route_id, seed
                    'test':[(r, 400) for r in        sorted([24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258])],
                    'test_unseen':[(r, 400) for r in sorted([18305, 1852,  24224, 3099, 3184, 3464, 27529, 26401, 2215,  25951])]                      
                    }, 

    'ParkingCutIn_':      {'train':[(24759, s) for s in range(200, 220)], # route_id, seed
                          'test':[(24759, 400)],
                          'test_unseen':[(18305, 400)]}, # start seed for test

    'AccidentTwoWays_':      {'train':[(25857, s) for s in range(200, 220)], # route_id, seed
                              'test':[(25857, 400)],
                              'test_unseen':[(1852, 400)]}, # start seed for test

    'DynamicObjectCrossing_':      {'train':[(24211, s) for s in range(200, 220)], # route_id, seed
                          'test':[(24211, 400)], # start seed for test
                              'test_unseen':[(24224, 400)]}, # start seed for test                          

    'CrossingBicycleFlow_':      {'train':[(3100, s) for s in range(200, 220)], # route_id, seed
                          'test':[(3100, 400)], # start seed for test
                              'test_unseen':[(3099, 400)]}, # start seed for test                          

    'VanillaNonSignalizedTurnEncounterStopsign_':      {'train':[(2416, s) for s in range(200, 220)], # route_id, seed
                          'test':[(2416, 400)], # start seed for test
                              'test_unseen':[(3184, 400)]}, # start seed for test

    'VehicleOpensDoorTwoWays_':      {'train':[(3472, s) for s in range(200, 220)], # route_id, seed
                            'test':[(3472, 400)], # start seed for test
                              'test_unseen':[(3464, 400)]}, # start seed for test

    'PedestrianCrossing_':      {'train':[(25863, s) for s in range(200, 220)], # route_id, seed
                            'test':[(25863, 400)], # start seed for test
                              'test_unseen':[(27529, 400)]}, # start seed for test
    
    'MergerIntoSlowTrafficV2_':      {'train':[(26408, s) for s in range(200, 220)], # route_id, seed
                            'test':[(26408, 400)], # start seed for test
                              'test_unseen':[(26401, 400)]}, # start seed for test

    'BlockedIntersection_':      {'train':[(27494, s) for s in range(200, 220)], # route_id, seed
                            'test':[(27494, 400)], # start seed for test
                              'test_unseen':[(2215, 400)]}, # start seed for test

    'HazardAtSideLaneTwoWays_':      {'train':[(24258, s) for s in range(200, 220)], # route_id, seed
                            'test':[(24258, 400)], # start seed for test
                              'test_unseen':[(25951, 400)]}, # start seed for test
    
}


MAX_EPISODES = {k: len(v['train']) for k, v in Task_to_Route.items()}

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_episode(datapath, route_id, seed, stack, grayscale, use_gaze, gaze_mask_sigma, gaze_mask_coeff, span = 8):
    # load the data
    path = f'{datapath}/route_{route_id}/seed_{seed}'
    episode_obs = torch.load(f'{path}/observations.pt', weights_only=False)
    episode_actions = torch.load(f'{path}/actions.pt', weights_only=False)['actions']
    
    episode_obs = torch.from_numpy(episode_obs).permute(0, 3, 1, 2)
    episode_actions = torch.from_numpy(np.stack(episode_actions))

    if use_gaze:
        gaze_info = torch.load(f'{path}/gaze.pt', weights_only=False) if use_gaze else None
        gaze_info = torch.tensor(gaze_info)

        g = gaze_info[:, :2]
        g[g < 0] = 0
        g[g > 1] = 1

        gaze_info[:, :2] = g

    
    stride = 2
    short_memory_length = 20 * span // stride
    # Most recent frames have higher index! Also have a vision to the future
    saliency_sigmas = [gaze_mask_sigma/(0.99**(short_memory_length - i)) for i in range(short_memory_length+1)]
    coeficients = [gaze_mask_coeff**(short_memory_length - i) for i in range(short_memory_length+1)]
    # coeficients = [1 for i in range(short_memory_length+1)]
    coeficients += coeficients[::-1][1:]
    saliency_sigmas += saliency_sigmas[::-1][1:]

    MASK = GazeToMask(320, 180, saliency_sigmas, coeficients=coeficients)
    
    if use_gaze:
        episode_saliency_gaze = torch.stack(
                [MASK.find_bunch_of_maps(means=
                    gaze_info[max(0, j - stride * short_memory_length):min(short_memory_length * stride + j + 1, len(gaze_info)):stride],
                        offset_start=max(short_memory_length - j , 0))
                            for j in tqdm(range(len(gaze_info)), total = len(gaze_info))], 0)
        episode_saliency_gaze = episode_saliency_gaze.unsqueeze(1).expand_as(episode_obs) # for RGB channel expansion
        episode_gaze_coordinates = gaze_info[:, :2]
    else:
        episode_saliency_gaze = torch.zeros_like(episode_obs, dtype=torch.float32)
        episode_gaze_coordinates = torch.zeros(len(episode_obs), dtype=torch.float32)
        
    episode_obs = torch.cat([episode_obs[0].unsqueeze(0)] * (stack - 1) + [episode_obs])
    episode_saliency_gaze = torch.cat([episode_saliency_gaze[0].unsqueeze(0)] * (stack - 1) + [episode_saliency_gaze])

    new_episode_obs = []
    new_episode_saliency_gaze = []
    for s in range(stack):
        end = None if s == stack - 1 else s - stack + 1
        new_episode_obs.append(episode_obs[s:end])
        new_episode_saliency_gaze.append(episode_saliency_gaze[s:end])        
    
    episode_obs = torch.stack(new_episode_obs, dim=1)
    episode_saliency_gaze = torch.stack(new_episode_saliency_gaze, dim=1)

    if grayscale:
        episode_obs = episode_obs.float().mean(2, keepdim=True).to(torch.uint8)
        episode_saliency_gaze = episode_saliency_gaze[:, :, 0].unsqueeze(2)
    
    return episode_obs, episode_actions, episode_saliency_gaze, episode_gaze_coordinates


def load_dataset(task, datapath, stack, grayscale, num_episodes, use_gaze, gaze_mask_sigma, gaze_mask_coeff):
    episodes_obs = []
    episodes_actions = []
    episodes_saliency_gaze = []
    episodes_gaze_coordinates = []
    assert num_episodes <= MAX_EPISODES[task], f'Number of episodes is greater than the available episodes for the task {task}'
    for route_id, seed in tqdm(Task_to_Route[task]['train'][:num_episodes]):
        episode_obs, episode_actions, episode_saliency_gaze, episode_gaze_coordinates = load_episode(datapath, route_id, seed, stack, grayscale, use_gaze, gaze_mask_sigma, gaze_mask_coeff)
        episodes_obs.append(episode_obs)
        episodes_actions.append(episode_actions)
        episodes_saliency_gaze.append(episode_saliency_gaze)
        episodes_gaze_coordinates.append(episode_gaze_coordinates)
    
    observations = torch.cat(episodes_obs, 0)
    actions = torch.cat(episodes_actions, 0)
    saliency_gaze = torch.cat(episodes_saliency_gaze, 0)
    gaze_coordinates = torch.cat(episodes_gaze_coordinates, 0)
    
    L, S, C, H, W = observations.shape
    observations = observations.view(L,S*C, H, W)
    saliency_gaze = saliency_gaze.view(L,S*C, H, W)
        
    return observations, actions, saliency_gaze, gaze_coordinates
    

def plot_gaze_and_obs(gaze, obs, save_path=None):
    # Create data for the plots
    y1 = gaze
    
    y2 = obs
    if obs.dtype == torch.uint8:
        y2 = obs.to(torch.float32)/255
    
    y3 = (y1 * y2).to(torch.float32)

    if len(y3.shape) == 3:
        if y3.shape[0] == 3:
            y3 = y3.permute(1, 2, 0)
        elif y3.shape[0] == 1:
            y3 = y3[0]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot 1
    ax1.imshow(y1, cmap='gray', vmax=1.0, vmin=0.0)
    ax1.set_title('pure gaze')

    # Plot 2
    ax2.imshow(y2, cmap='gray', vmax=1.0, vmin=0.0)
    ax2.set_title('pure obs')

    # Plot 3
    ax3.imshow(y3, cmap='gray', vmax=1.0, vmin=0.0)
    ax3.set_title('merged gaze and obs')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    
    # Show the plots
    plt.show()
