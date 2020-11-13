from pytorch_lightning.loggers import TensorBoardLogger

from touchstone.modules import PPOLightning
import pytorch_lightning as pl
import torch
import numpy as np
import argparse


def main(params) -> None:
    model = PPOLightning(params)

    trainer = pl.Trainer(
        # gpus=1,
        # distributed_backend='dp',
        min_epochs=params.iterations * params.epochs,
        max_epochs=params.iterations * params.epochs,
        val_check_interval=100,
        logger=TensorBoardLogger('tb_logs')
    )

    trainer.fit(model)


torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="betas for ADAM")
parser.add_argument("--env", type=str, default="Pendulum-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--sync_rate", type=int, default=10,
                    help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=1000,
                    help="capacity of the replay buffer")
parser.add_argument("--warm_start_size", type=int, default=1000,
                    help="how many samples do we use to fill our buffer at the start of training")
parser.add_argument("--eps_last_frame", type=int, default=1000,
                    help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
parser.add_argument("--max_episode_reward", type=int, default=200,
                    help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000,
                    help="max episode reward in the environment")
parser.add_argument("--time_steps", type=int, default=128,
                    help="time steps for collecting data")
parser.add_argument("--num_actors", type=int, default=8,
                    help="number of actors to run in parallel")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--iterations", type=int, default=5000)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_param", type=float, default=0.1)
parser.add_argument("--action_std", type=float, default=0.5)

args, _ = parser.parse_known_args()

if __name__ == '__main__':
    main(args)
