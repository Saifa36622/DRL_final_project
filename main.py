import pygame as pg
import sys
from settings import *
from map import *
from player import *
from raycasting import *
from object_renderer import *
from sprite_object import *
from object_handler import *
from weapon import *
from sound import *
from pathfinding import *
import gymnasium as gym
from gymnasium.envs.registration import register


class Game:
    def __init__(self):
        pg.init()
        pg.mouse.set_visible(False)
        self.screen = pg.display.set_mode(RES)
        pg.event.set_grab(True)
        self.clock = pg.time.Clock()
        self.delta_time = 1
        self.global_trigger = False
        self.global_event = pg.USEREVENT + 0
        pg.time.set_timer(self.global_event, 40)
        self.count_death = 0
        self.game_over_check = False
        self.use_time = 0
        self.win_check = False
        self.hit_count = 0
        self.new_game()

    def new_game(self):
        self.map = Map(self)
        self.player = Player(self)
        self.object_renderer = ObjectRenderer(self)
        self.raycasting = RayCasting(self)
        self.object_handler = ObjectHandler(self)
        self.weapon = Weapon(self)
        self.sound = Sound(self)
        self.count_death = 0
        self.use_time = 0
        self.hit_count = 0
        self.game_over_check = False
        self.win_check = False
        self.pathfinding = PathFinding(self)
        pg.mixer.music.play(-1)

    def update(self):
        self.player.update()
        self.raycasting.update()
        self.object_handler.update()

        # print(f"Deaths so far: {self.count_death}")
        # print(type(self.player.health))
        self.weapon.update()
        pg.display.flip()
        self.delta_time = self.clock.tick(FPS)

        # print(self.clock.get_time())
        pg.display.set_caption(f'{self.clock.get_fps() :.1f}')
        self.use_time += 1
        print(self.use_time)


    def update2(self,action):

        self.player.update2(action)
        
        self.raycasting.update()

        self.win_check = self.object_handler.update2()

        # print(f"Deaths so far: {self.count_death}")
        # print(type(self.player.health))
        self.weapon.update()
        pg.display.flip()
        self.delta_time = self.clock.tick(FPS)
        pg.display.set_caption(f'{self.clock.get_fps() :.1f}')
        self.use_time += 1
        


    def draw(self):
        # self.screen.fill('black')
        self.object_renderer.draw()
        self.weapon.draw()
        # self.map.draw()
        # self.player.draw()

    def check_events(self):
        self.global_trigger = False
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            elif event.type == self.global_event:
                self.global_trigger = True
            self.player.single_fire_event(event)

    def check_events2(self,action):
        # self.global_trigger = False
        # for event in pg.event.get():
        #     if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
        #         pg.quit()
        #         sys.exit()
        #     elif event.type == self.global_event:
        #         self.global_trigger = True
        self.player.single_fire_event2(action)

    def run(self):
        while True:
            self.check_events()
            self.update()
            self.draw()


if __name__ == '__main__':
    game = Game()
    game.run()
