import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from main import Game
import pygame as pg
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import HumanRendering
import pygame as pg
import settings
from stable_baselines3.common.callbacks import BaseCallback

class DoomEnv(gym.Env):

    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,):
 
        
        self.observation_space = spaces.Dict(
            {
                "Display": spaces.Box(low=0, high=255, shape=(900, 1600, 3), dtype=np.uint8) ,
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

        self.action_space = spaces.MultiDiscrete([1600 ,2, 2, 2, 2, 2]) # mouse action ,w a s d shoot

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = pg.display.set_mode((1600, 900))
        self.clock = pg.time.Clock()
        self.count_ep = 0
        self.reward_sum = 0

        self.game = Game()

        # self.game.new_game()

    def reset(self,seed=42):

        super().reset(seed=seed)

        if self.count_ep != 0 :
            wandb.log(data={
                "episode" :self.count_ep ,
                "survive time" : self.game.use_time,
                "reward" : self.reward_sum,
                "death count" : self.game.count_death
            })
        self.reward_sum = 0
        self.game.new_game()
        # self.game.check_events()
        # self.game.update()
        # self.game.draw()
        
        observation = self._get_observation()
        
        info = {
            "wow" : 0
        }
        self.count_ep += 1
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

        reward += count_death

        return reward ,terminated,truncated

    def step(self, action):
        
        self.game.check_events2(action)
        
        win_check  = self.game.update2(action) # win or not 
        
        self.game.draw()

        # self.game.draw()  # draws on self.game.screen
        # self.render()     # blits to window and flips
        observation = self._get_observation()

        reward,terminated,truncated = self.cal_reward(
            win_check,
            self.game.use_time ,
            self.game.game_over_check,
            self.game.count_death
            )
        self.reward_sum += reward
        info = {
            "wow" : 0
        }
        # print(self.game.game_over_check)
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):

        # health = float(self.game.player.health)
        # health_array = np.array([health], dtype=np.float32)
        # observation = {
        #     "Display": pg.surfarray.array3d(self.game.screen),
        #     "Agent": health_array
        # }

        # pg.display.update()
        display_surface = pg.display.get_surface()
        frame = pg.surfarray.array3d(display_surface).swapaxes(0, 1)  # Transpose to (H, W, C)
        
        health = float(self.game.player.health)
        health_array = np.array([health], dtype=np.float32)

        observation = {
            "Display": frame,
            "Agent": health_array
        }
        return observation
    
    # def render(self):
    #     if self.render_mode == "human":
    #         self.window.blit(self.game.screen, (0, 0))
    #         pg.display.flip()
    #         self.clock.tick(settings.FPS)

    def close(self):
        pg.quit()

gym.register(
    id="DoomEnv",
    entry_point=DoomEnv,
)

env = DoomEnv(render_mode="human")
# env = HumanRendering(env)
# print(env.observation_space.sample())
# check_env(env)

# print("ok")
wandb.init(project="doom_drl_project", sync_tensorboard=True,name="test_1")

# ---------------------------------------------------------------

# Choose the RL algorithm: PPO or SAC
model = PPO(
            policy="MultiInputPolicy", 
            env=env, 
            learning_rate=3e-1,
            n_steps = 4096,
            batch_size=  64,
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
            tensorboard_log =f"runs/{wandb.run.id}",
            policy_kwargs = None,
            verbose = 0,
            seed = None,
            device = "auto",
            _init_setup_model = True,
            )

# Train the model
# model.learn(
#     total_timesteps=1000,
#         callback=WandbCallback(
#         gradient_save_freq=100,
#         model_save_path=f"models/{wandb.run.id}",
#         verbose=2,
#     ),) 


model.learn(
    total_timesteps=1000
)

# Save the model after training
model.save("Doom_model")
wandb.finish()

# Optionally: Evaluate the model
obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

# ---------------------------------------------------------------
