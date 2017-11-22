from random import randint, sample
import pygame
from pygame.locals import *
import sys


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
BLUE = [0, 0, 255]
RED = [255, 0, 0]
YELLOW = [0,128,0]
DOT_SIZE = 30

class Topo:
    
    def __init__(self, screen):
        
        self.screen = screen
    
    def get_tp(self):
        
#         self.n = randint(3,29)
        self.n = 4
        self.source_target_nodes = ((0,0), (2, 2))
#         sample_num = randint(1,self.n-1)
#         self.fault_nodes = []
#         for i in range(sample_num):
#             node = (randint(0,self.n-1), randint(0,self.n-1))
#             if node not in self.source_target_nodes:
#                 self.fault_nodes.append(node)
#                 i += 1
        self.fault_nodes = [(1,2), (2,1)]
        return self.fault_nodes, self.source_target_nodes, self.n
    
    def draw_tp(self, DP_x, DP_y):
        
        self.screen.fill(WHITE)
        for i in range(0,self.n):
            for j in range(0,self.n):
                if (i,j) in self.fault_nodes:
                    pygame.draw.rect(self.screen, BLACK, (i*DOT_SIZE, j*DOT_SIZE, DOT_SIZE, DOT_SIZE),0)
                else:
                    pygame.draw.rect(self.screen, BLACK, (i*DOT_SIZE, j*DOT_SIZE, DOT_SIZE, DOT_SIZE),1)
        
        target_node_x = (self.source_target_nodes[1][0])*DOT_SIZE
        target_node_y = (self.source_target_nodes[1][1])*DOT_SIZE
        
        pygame.draw.rect(self.screen, BLUE, (target_node_x, target_node_y, DOT_SIZE, DOT_SIZE))
        pygame.draw.rect(self.screen, RED, (DP_x, DP_y, DOT_SIZE, DOT_SIZE))
