import gym

import sys
import random
import matplotlib.pyplot as plt
import pygame
import numpy as np
from pygame import *
import os
from skimage.color import rgb2gray  # Help us to gray our frames
from skimage import transform  # Help us to preprocess the frames
from stable_baselines3.common.env_checker import check_env


class Grass(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(grass_img, (width, height))
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.center = (self.x, self.y)


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(wall_img, (width * 2, height * 2))
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.center = (self.x, self.y)


class Hero(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        self.speed_x = 0
        self.speed_y = 0
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)

    def move(self, i):
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0:
            if i == 1:
                self.speed_x = 5
            elif i == 2:
                self.speed_x = -5
            elif i == 3:
                self.speed_y = 5
            elif i == 4:
                self.speed_y = -5

    def update(self):
        """if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0:
            if random.randint(0, 1):
                if random.randint(0, 1):
                    self.speed_x = 5
                else:
                    self.speed_x = -5
                self.speed_y = 0
            else:
                if random.randint(0, 1):
                    self.speed_y = 5
                else:
                    self.speed_y -= 5
                self.speed_x = 0"""
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y


class Forest(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)


class Mine(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)


class Score(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)


def init_Grass():
    w = 32
    h = 16
    start = width * 2 + 20
    for i in range(w):
        for j in range(h):
            all_sprites.add(Grass(start + i * 40, start + j * 40))


def init_Wall():
    w = 18
    h = 8
    start = 40
    for i in range(0, w):
        all_sprites.add(Wall(start + i * 80, start))
    for i in range(0, w):
        all_sprites.add(Wall(start + i * 80, start + 9 * 80))

    for i in range(1, h + 1):
        all_sprites.add(Wall(start, start + i * 80))
    for i in range(1, h + 1):
        all_sprites.add(Wall(start + 17 * 80, start + i * 80))


def init_forest(board, h, w, start):
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if board[i][j] == 'f':
                all_sprites.add(Forest(start, j - 1, i - 1, forest_img))


def init_mine(board, h, w, start):
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if board[i][j] == 'm':
                all_sprites.add(Mine(start, j - 1, i - 1, mine_img))


def init_score():
    start = 20
    for i in range(4):
        all_sprites.add(Score(start, 36, i, coin_img))


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]

    # Normalize Pixel Values
    # normalized_frame = gray % 255.0
    normalized_frame = gray
    normalized_frame *= 255.0 / normalized_frame.max()
    # Resize
    # Thanks to Miko??aj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [156, 80])

    return preprocessed_frame


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        n_actions = 2
        self.action_space = gym.spaces.Discrete(n_actions)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        observation = preprocess_frame(screen1)
        reward = 0
        if action == 1:
            Hero.move(a,1)
        elif action == 2:
            Hero.move(a,1)
        elif action == 3:
            Hero.move(a,1)
        elif action == 4:
            Hero.move(a,1)
        if 1:
            reward = -1
        info = {}
        if 1 != 0:
            done = True
        else:
            done = False
        return observation, reward, done, info

    def reset(self):
        init_Wall()
        init_Grass()
        a = Hero(100, 0, 0, hero1_img)
        all_sprites.add(a)
        # all_sprites.add(Hero(100, 0, 0, hero1_img))
        all_sprites.add(Hero(100, 31, 0, hero2_img))
        all_sprites.add(Hero(100, 0, 15, hero3_img))
        all_sprites.add(Hero(100, 31, 15, hero4_img))
        sys.stdin = open('input.txt', 'r')
        board = [input().split() for i in range(18)]
        init_forest(board, 16, 32, 100)
        init_mine(board, 16, 32, 100)
        init_score()
        observation = np.zeros(5)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pygame.display.flip()



sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (182, 182, 182)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

x = 80
y = 80
WIDTH = 1561  # ???????????? ???????????????? ????????
HEIGHT = 801  # ???????????? ???????????????? ????????
FPS = 30  # ?????????????? ???????????? ?? ??????????????
width = 40
height = 40
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=1)
pygame.init()
infoObject = pygame.display.Info()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("success_code")
pygame.font.SysFont('Arial', 25)

# ?????????????????? ?????????? ??????????????
game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, 'img')
grass_img = pygame.image.load(os.path.join(img_folder, 'grass.png')).convert()
wall_img = pygame.image.load(os.path.join(img_folder, 'wall.jpg')).convert()
forest_img = pygame.image.load(os.path.join(img_folder, 'forest.png')).convert()
mine_img = pygame.image.load(os.path.join(img_folder, 'mine.jpg')).convert()
coin_img = pygame.image.load(os.path.join(img_folder, 'coin.png')).convert()

hero1_img = pygame.image.load(os.path.join(img_folder, 'hero1.jpg')).convert()
hero2_img = pygame.image.load(os.path.join(img_folder, 'hero2.jpg')).convert()
hero3_img = pygame.image.load(os.path.join(img_folder, 'hero3.jpg')).convert()
hero4_img = pygame.image.load(os.path.join(img_folder, 'hero4.png')).convert()

pygame.mixer.init()
music_folder = os.path.join(game_folder, 'music')
pygame.mixer.music.load(os.path.join(music_folder, 'hero.mp3'))
# pygame.mixer.music.play(loops=-1)

clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()

init_Wall()
init_Grass()
a = Hero(100, 0, 0, hero1_img)
all_sprites.add(a)
# all_sprites.add(Hero(100, 0, 0, hero1_img))
all_sprites.add(Hero(100, 31, 0, hero2_img))
all_sprites.add(Hero(100, 0, 15, hero3_img))
all_sprites.add(Hero(100, 31, 15, hero4_img))

board = [input().split() for i in range(18)]
init_forest(board, 16, 32, 100)
init_mine(board, 16, 32, 100)
init_score()
running = True

while running:
    # ???????????? ???????? ???? ???????????????????? ????????????????
    clock.tick(FPS)
    # ???????? ???????????????? (??????????????)
    for event in pygame.event.get():
        # check for closing window
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    # ????????????????????
    all_sprites.update()
    # ??????????????????
    screen.fill(BLACK)
    all_sprites.draw(screen)
    screen1 = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = preprocess_frame(screen1)
    for i in range(0, WIDTH, 40):
        pygame.draw.line(screen, RED, [i, 0], [i, HEIGHT], 1)
    for i in range(0, HEIGHT, 40):
        pygame.draw.line(screen, RED, [0, i], [WIDTH, i], 1)
    # ?????????? ?????????????????? ??????????, ???????????????????????????? ??????????
    pygame.display.flip()
print(frame.astype(int).tolist())
pygame.quit()

sys.stdin.close()
sys.stdout.close()