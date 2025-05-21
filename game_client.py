import socket
import pickle
import pygame as pg
from main import Game
import socket
import pickle
import numpy as np
import cv2
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from main import Game
import pygame as pg
import wandb
# from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import HumanRendering
import pygame as pg
import settings
from stable_baselines3.common.callbacks import BaseCallback
import torch
import cv2
from stable_baselines3.common.callbacks import CheckpointCallback

class DoomEnv(gym.Env):
    def __init__(self, render_mode="human"):
        # ... existing initialization code ...
        self.observation_space = spaces.Dict(
            {
                "Display": spaces.Box(low=0, high=255, shape=(450, 800, 3), dtype=np.uint8) ,
                "Agent": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)  # Continuous health
            }
        )
        # self.observation_space = spaces.Box(low=0, high=255, shape=(1600, 900, 3), dtype=np.uint8)  # Example: RGB image
        # self.action_space == spaces.Dict(
        #     {
        #         "Mouse": spaces.Box(low=0, high=255, shape=(1600, 900, 3), dtype=np.uint8) ,
        #         "Agent": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)  # Continuous health
        #     }
        # )

        self.action_space = spaces.MultiDiscrete([41 ,2, 2, 2, 2, 2]) # mouse action ,w a s d shoot

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = pg.display.set_mode((1600, 900))
        self.clock = pg.time.Clock()
        self.count_ep = 0
        self.reward_sum = 0

        # Set up the client socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 50007))  # Replace with your server's address and port

    def reset(self,seed=42):

        super().reset(seed=seed)

        if self.count_ep != 0 :
            wandb.log(data={
                "episode" :self.count_ep ,
                "survive time" : self.use_time,
                "reward" : self.reward_sum,
                "death count" : self.death_count
            })
        self.reward_sum = 0
        # self.game.new_game()
        # self.game.check_events()
        # self.game.update()
        # self.game.draw()
        
        observation = self._get_observation()
        
        info = {
            "wow" : 0
        }
        self.count_ep += 1
        print(f"end {self.count_ep}")
        return observation,info
    
    def cal_reward(self,win_check,time,game_over,count_death):
        reward = 0
        terminated = False
        truncated = False

        if win_check == True :
            terminated = True
            reward += 10
            return reward ,terminated,truncated
        
        if game_over == True :
            terminated = True
            reward -= 10
            return reward ,terminated,truncated
        
        if time > 10000 :
            truncated = True
            reward -= 5
            return reward ,terminated,truncated
        
        reward += self.hit_count * 0.5
        reward += count_death

        return reward ,terminated,truncated

    def step(self, action):
        
        # self.game.check_events2(action)
        
        # win_check  = self.game.update2(action) # win or not 
    
        # self.game.draw()
        action_data = pickle.dumps(action)
        self.client_socket.sendall(len(action_data).to_bytes(4, byteorder='big') + action_data)
        
        # self.game.draw()  # draws on self.game.screen
        # self.render()     # blits to window and flips

        observation = self._get_observation()

        reward,terminated,truncated = self.cal_reward(
            self.win_check,
            self.use_time ,
            self.game_over,
            self.death_count
            )
        
        self.reward_sum += reward
        info = {
            "wow" : 0
        }
        # print(self.game.game_over_check)
        return observation, reward, terminated, truncated, info
    
    def _recv_all(self, length):
        """Helper function to receive the exact number of bytes."""
        data = b''
        while len(data) < length:
            more = self.client_socket.recv(length - len(data))
            if not more:
                raise EOFError('Socket connection closed prematurely')
            data += more
        return data

    def _get_observation(self):
        # Receive the length of the incoming data
        length_bytes = self._recv_all(4)
        length = int.from_bytes(length_bytes, byteorder='big')

        # Receive the actual data
        data = self._recv_all(length)

        # Deserialize the data
        state = pickle.loads(data)

        # Extract and process the frame
        frame = state['frame']
        resized_frame = cv2.resize(frame, (800, 450))

        # Extract and process the health
        health = float(state['health'])
        health_array = np.array([health], dtype=np.float32)

        # Store additional metrics for reward calculation
        self.win_check = state['win_check']
        self.game_over = state['game_over']
        self.hit_count = state['hit_count']
        self.death_count = state['death_count']
        self.use_time = state['use_time']

        observation = {
            "Display": resized_frame,
            "Agent": health_array
        }
        return observation
    
gym.register(
    id="DoomEnv",
    entry_point=DoomEnv,
)

name = "r_5_floor_3"
# wandb.init(project="doom_drl_project_2", sync_tensorboard=True,name=name)
env = DoomEnv(render_mode="human")

torch.cuda.empty_cache()
import gc
gc.collect()

model = PPO(
            policy="MultiInputPolicy", 
            env=env, 
            learning_rate=3e-4,
            n_steps = 250,
            batch_size= 250,
            n_epochs = 10,
            gamma = 0.99,
            gae_lambda = 0.95,
            clip_range = 0.2,
            clip_range_vf = None,
            normalize_advantage = True,
            ent_coef = 0.0, # Entropy coefficient for the loss calculation
            vf_coef = 0.5,
            max_grad_norm  = 0.5,
            use_sde = False,
            sde_sample_freq = -1,
            rollout_buffer_class = None,
            rollout_buffer_kwargs = None,
            target_kl = None,
            stats_window_size = 100,
            tensorboard_log ="./tensorboard_logs/",
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            _init_setup_model = True,
            )

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f'./logs/run_{name}',
                                         name_prefix=f'doom_model_{env.count_ep}')

model.learn(
    total_timesteps=100000,
    tb_log_name= name,
    callback=checkpoint_callback
)

model.save(f"Doom_model_{name}")
# wandb.finish()
print("done")