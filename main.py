import sys
import random
import pygame
from pygame import *
import os

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

    def update(self):
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0:
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
                self.speed_x = 0
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
        self.image.set_colorkey(WHITE)
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
            if board[i][j][1] == 'f':
                all_sprites.add(Forest(start, j - 1, i - 1, forest_img))

def init_mine(board, h, w, start):
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if board[i][j][1] == 'm':
                all_sprites.add(Mine(start, j - 1, i - 1, mine_img))

def init_score():
    start = 20
    for i in range(4):
        all_sprites.add(Score(start, 36, i, coin_img))

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

x = 80
y = 80
WIDTH = 1561  # ширина игрового окна
HEIGHT = 801  # высота игрового окна
FPS = 30  # частота кадров в секунду
width = 40
height = 40


pygame.init()
infoObject = pygame.display.Info()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("success_code")
pygame.font.SysFont('Arial', 25)

# настройка папки ассетов
game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, 'img')
grass_img = pygame.image.load(os.path.join(img_folder, 'grass.png')).convert()
wall_img = pygame.image.load(os.path.join(img_folder, 'wall.jpg')).convert()
forest_img = pygame.image.load(os.path.join(img_folder, 'forest.png')).convert()
mine_img = pygame.image.load(os.path.join(img_folder, 'mine.png')).convert()
coin_img = pygame.image.load(os.path.join(img_folder, 'coin.png')).convert()

hero1_img = pygame.image.load(os.path.join(img_folder, 'hero1.jpg')).convert()
hero2_img = pygame.image.load(os.path.join(img_folder, 'hero2.jpg')).convert()
hero3_img = pygame.image.load(os.path.join(img_folder, 'hero3.jpg')).convert()
hero4_img = pygame.image.load(os.path.join(img_folder, 'hero4.png')).convert()

pygame.mixer.init()
music_folder = os.path.join(game_folder, 'music')
pygame.mixer.music.load(os.path.join(music_folder, 'hero.mp3'))
pygame.mixer.music.play(loops=-1)

clock = pygame.time.Clock()


all_sprites = pygame.sprite.Group()

init_Wall()
init_Grass()
all_sprites.add(Hero(100, 0, 0, hero1_img))
all_sprites.add(Hero(100, 31, 0, hero2_img))
all_sprites.add(Hero(100, 0, 15, hero3_img))
all_sprites.add(Hero(100, 31, 15, hero4_img))

board = [input().split() for i in range(18)]
init_forest(board, 16, 32, 100)
init_mine(board, 16, 32, 100)
init_score()

running = True
while running:
    # Держим цикл на правильной скорости
    clock.tick(FPS)
    # Ввод процесса (события)
    for event in pygame.event.get():
        # check for closing window
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


    #Обновление
    all_sprites.update()
    print()
    # Рендеринг
    screen.fill(BLACK)
    all_sprites.draw(screen)
    for i in range(0, WIDTH, 40):
            pygame.draw.line(screen, RED, [i, 0], [i, HEIGHT], 1)
    for i in range(0, HEIGHT, 40):
            pygame.draw.line(screen, RED, [0, i], [WIDTH, i], 1)
    # После отрисовки всего, переворачиваем экран
    pygame.display.flip()

pygame.quit()

sys.stdin.close()
sys.stdout.close()
