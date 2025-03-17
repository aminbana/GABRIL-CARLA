from utils import load_dataset, MAX_EPISODES, set_seed_everywhere, plot_gaze_and_obs
import random
import torch
from models.linear_models import AutoEncoder, Encoder, Decoder
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    default_task = 'Mixed_'

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--task", default=default_task, type=str)
    parser.add_argument("--datapath", default="dataset/bench2drive220/", type=str)
    parser.add_argument("--stack", default=2, type=int)
    parser.add_argument("--grayscale", default=True, type=bool)

    # Save & Evaluation
    parser.add_argument("--num_episodes", default=MAX_EPISODES[default_task], type=int)
    parser.add_argument("--n_epochs", default=500, type=int)

    # VQVAE & Hyperparams
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--step", default=50, type=float)
    parser.add_argument("--val_split_ratio", default=0.99, type=float)
    
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    
    parser.add_argument("--gaze_mask_sigma", type=float, default=30.0, help='Sigma of the Gaussian for the gaze mask')
    parser.add_argument("--gaze_mask_coeff", type=float, default=0.8, help='Base coefficient of the Gaussian for the gaze mask')
    

    args = parser.parse_args()
    return args


def train(args):
    set_seed_everywhere(args.seed)

    save_model_root = f'trained_models/gaze_predictor_models/{args.task}'
    tensorboard_root = f'logs/gaze_predictor/{args.task}'

    save_dir = f'seed_{args.seed}_stack_{args.stack}_ep_{args.num_episodes}_grayscale_{args.grayscale}'

    print('Start loading data ...')
    observations, _, gaze_masks, _ = load_dataset(args.task, args.datapath, args.stack, args.grayscale, args.num_episodes, use_gaze=True, gaze_mask_sigma=args.gaze_mask_sigma, gaze_mask_coeff=args.gaze_mask_coeff)

    observations = torch.as_tensor(observations, dtype=torch.float32)/ 255.0

    #shuffle and validation split
    indices = list(range(len(observations)))
    random.shuffle(indices)
    train_indices = indices[:int(len(observations) * args.val_split_ratio)]
    val_indices = indices[int(len(observations) * args.val_split_ratio):]

    observations, observations_val = observations[train_indices], observations[val_indices]
    gaze_masks, gaze_masks_val = gaze_masks[train_indices], gaze_masks[val_indices]

    train_data_loader = torch.utils.data.DataLoader(
        list(zip(observations, gaze_masks)), batch_size=args.bs, shuffle=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        list(zip(observations_val, gaze_masks_val)), batch_size=args.bs, shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(
        args.stack * (1 if args.grayscale else 3),
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    )
    decoder = Decoder(
        args.stack * (1 if args.grayscale else 3),
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    )

    model = AutoEncoder(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)

    import datetime
    now_time = datetime.datetime.now()
    date_dir = str(now_time.strftime("%Y_%m_%d_%H_%M_%S"))

    os.makedirs(f'{save_model_root}/{save_dir}', exist_ok=True)
    writer = SummaryWriter(f'{tensorboard_root}/{date_dir}_{save_dir}')


    for ep in tqdm(range(args.n_epochs)):
        model.train()
        ep_train_loss, train_size = 0, 0
        ep_test_loss, test_size = 0, 0
        for obs, gz in train_data_loader:
            obs = obs.to(device)
            gz = gz.to(device)

            output = model(obs)

            loss = nn.functional.mse_loss(output, gz)

            optimizer.zero_grad() # Reset gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update model parameters

            ep_train_loss += loss.item() * len(obs)
            train_size += len(obs)

        ep_train_loss /= train_size

        writer.add_scalar('train_loss', ep_train_loss, ep)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)
        scheduler.step()


        model.eval()
        for obs, gz in test_data_loader:
            obs = obs.to(device)
            gz = gz.to(device)

            with torch.no_grad():
                output = model(obs)

            loss = nn.functional.mse_loss(output, gz)

            ep_test_loss += loss.item() * len(obs)
            test_size += len(obs)

        ep_test_loss /= test_size

        writer.add_scalar('test_loss', ep_test_loss, ep)



    # pick a random sample from the test set
    obs, gz = random.choice(list(test_data_loader))
    obs = obs.to(device)
    gz = gz.to(device)

    with torch.no_grad():
        output = model(obs)

        output = output.cpu()[0, -1]
        obs = obs.cpu()[0, -1]
        gz = gz.cpu()[0, -1]

        plot_gaze_and_obs(gz, obs, f'{save_model_root}/{save_dir}/gaze_gt.png')
        plot_gaze_and_obs(output, obs, f'{save_model_root}/{save_dir}/gaze_pred.png')

    # Save the model
    torch.save(model.state_dict(), f'{save_model_root}/{save_dir}/model.torch')
    print('Model saved!')
    writer.close()
    print('Done!')


if __name__ == "__main__":
    args = get_args()
    train(args)