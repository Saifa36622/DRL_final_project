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
import socket
import pickle
import pygame as pg
from main import Game

def game_server(host='localhost', port=50007):
    # Initialize the game
    game = Game()
    action = None

    # Set up the server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Game server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # Update game state
                if action is None:
                    game.check_events()
                    game.update()
                    game.draw()
                else:
                    game.check_events2(action)
                    game.update2(action)
                    game.draw()

                # Capture the screen
                frame = pg.surfarray.array3d(pg.display.get_surface())
                frame = frame.swapaxes(0, 1)  # Transpose to (H, W, C)

                # Prepare the state data
                state = {
                    'frame': frame,
                    'health': game.player.health,
                    'win_check': game.win_check,
                    'game_over': game.game_over_check,
                    'hit_count': game.hit_count,
                    'death_count': game.count_death,
                    'use_time' : game.use_time
                }

                # Serialize and send the state data
                data = pickle.dumps(state)
                conn.sendall(len(data).to_bytes(4, byteorder='big') + data)

                # Receive the action command
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, byteorder='big')
                action_data = conn.recv(length)
                action = pickle.loads(action_data)

if __name__ == '__main__':
    game_server()



                
