import pygame
from pygame.locals import *
from routgame.get_topo import Topo
import sys
import math
import numpy as np


WHITE = [255, 255, 255]
BLACK = [0, 0 ,0]
BLUE = [0, 0, 255]
RED = [255, 0, 0]
DOT_SIZE = 30
FPS = 60

pygame.init()
screen = pygame.display.set_mode((DOT_SIZE*4,DOT_SIZE*4), 0, 32)
pygame.display.set_caption('Routing')

tp = Topo(screen)
fpsClock = pygame.time.Clock()
def relu(x):
    if x > 0:
        return x
    else:
        return 0


class Environment:
    
    def __init__(self):
        
        self.fault_node, self.source_target_node, self.n = tp.get_tp()
        self.source_node = self.source_target_node[0]
        self.target_node = self.source_target_node[1]
        self.DP_x = self.source_node[0]*DOT_SIZE
        self.DP_y = self.source_node[1]*DOT_SIZE
        self.dis_st = sum(map(lambda i,j:abs(i-j), self.source_node, self.target_node))
        self.num_step = 0

    def step(self, action):
        
        pygame.event.pump()
        if sum(action) == 1:
            self.num_step += 1
        reward = 0
        terminal = False
        
        if action[0] == 1:
            #rout up
            self.DP_y -= DOT_SIZE
            if self.DP_y <= 0:
                self.DP_y = 0
                
        elif action[1] == 1:
            #rout down
            self.DP_y += DOT_SIZE
            if self.DP_y >= (self.n-1)*DOT_SIZE:
                self.DP_y = (self.n-1)*DOT_SIZE
        
        elif action[2] == 1:
            #rout left
            self.DP_x -= DOT_SIZE
            if self.DP_x <= 0:
                self.DP_x = 0
        
        elif action[3] == 1:
            #rout right
            self.DP_x += DOT_SIZE
            if self.DP_x >= (self.n-1)*DOT_SIZE:
                self.DP_x = (self.n-1)*DOT_SIZE
        
        # if self.num_step >= 1:
        #     curr_dis = abs(self.DP_x//DOT_SIZE-self.target_node[0]) + abs(self.DP_y//DOT_SIZE-self.target_node[1])
        #     reward = relu(self.dis_st - curr_dis)
        #     if curr_dis < self.dis_st:
        #         self.dis_st = curr_dis
            
        if (self.DP_x//DOT_SIZE, self.DP_y//DOT_SIZE) in self.fault_node:
            terminal = True
            reward = -1
            print('you are failed')

        if (self.DP_x//DOT_SIZE, self.DP_y//DOT_SIZE) == self.target_node:
            terminal = True
#             reward = self.dis_st/self.num_step
            reward = 1
            print('you are win')
#             print(reward)
           
        tp.draw_tp(self.DP_x, self.DP_y)
        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        observation = (np.array([[self.DP_x//DOT_SIZE, self.DP_y//DOT_SIZE]]) - np.array([self.target_node]))/4
#         observation = 4*(self.DP_x//DOT_SIZE) + (self.DP_y//DOT_SIZE+1)
        pygame.display.update()
        if terminal:
            self.__init__()
            # self.step([0,0,0,0])
        fpsClock.tick(FPS)
        return observation, reward, terminal

if __name__ == '__main__':    

    env = Environment()
    env.step([0,0,0,0])
    while True:
        action = [0,0,0,0]
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    action = [1,0,0,0]
                elif event.key == K_DOWN:
                    action = [0,1,0,0]
                elif event.key == K_LEFT:
                    action = [0,0,1,0]
                elif event.key == K_RIGHT:
                    action = [0,0,0,1]
    #         elif event.type == KEYUP:
    #             action = [0,0,0,0]
                _, reward, terminal = env.step(action)
                print(reward)
                
            