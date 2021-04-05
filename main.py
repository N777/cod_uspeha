import sys
import random
import pygame
from pygame import *
import numpy as np
import os

class States():
    def __init__(self):
        self.cod_action = {}
        self.cod_status = {}
        sost = ['Nw', 'Ng', 'Nf', 'Nm', 'Eh', 'Mm', 'Em']
        cnt = 0
        for i1 in range(len(sost)):
            for i2 in range(len(sost)):
                for i3 in range(len(sost)):
                    for i4 in range(len(sost)):
                        self.cod_status[sost[i1] + sost[i2] + sost[i3] + sost[i4]] = cnt
                        cnt += 1


        self.cod_action['speed_x_up'] = 0
        self.cod_action['speed_x_down'] = 1
        self.cod_action['speed_y_up'] = 2
        self.cod_action['speed_y_down'] = 3
        self.cod_action['Nm->Mm'] = 4
        self.cod_action['Em->Mm'] = 5
        self.cod_action['attack'] = 6

        self.weight = np.zeros((len(self.cod_status), len(self.cod_action)))


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
        self.money = 100
        self.speed_x = 0
        self.speed_y = 0
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)
        self.Hero_states = States()

    def move(self):
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0:
            sost = ''
            start = 40
            # верх
            if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] == \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0]:
                sost += 'Mm'
            elif board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] != \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0] and \
                    board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] != 'N':
                sost += 'E' + board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][1]
            else:
                sost += board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40]

            # право
            if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] == \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0]:
                sost += 'Mm'
            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] != \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0] and \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] != 'N':
                sost += 'E' + board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][1]
            else:
                sost += board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1]

            # низ
            if board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] == \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0]:
                sost += 'Mm'
            elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] != \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0] and \
                    board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] != 'N':
                sost += 'E' + board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][1]
            else:
                sost += board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40]

            # лево
            if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] == \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0]:
                sost += 'Mm'
            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] != \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40][0] and \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] != 'N':
                sost += 'E' + board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][1]
            else:
                sost += board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1]

            # AI

            self.money = 0

    def update(self):
        self.rect.x += self.speed_x


class Forest(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)

class Mine(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)

class Score(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)

class Back_ground_score(pygame.sprite.Sprite):
    def __init__(self, start, x, y, w, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width * w, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x + width * w // 2 - 20, self.y)

class Mini_hero(pygame.sprite.Sprite):
    def __init__(self, start, x, y, img):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((1, 1)), RLEACCEL)
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
        all_sprites.add(Back_ground_score(100, 35, i * 2 - 2, 2, rect_img))
        all_sprites.add(Back_ground_score(100, 33, i * 2 - 1, 4, rect_img))
        all_sprites.add(Score(start, 38, i * 2, coin_img))

def draw_text(surf, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

def show_go_screen():
    draw_text(screen, "AI!", 64, WIDTH / 2, HEIGHT / 4)
    #draw_text(screen, "Arrow keys move, Space to fire", 22, WIDTH / 2, HEIGHT / 2)
    draw_text(screen, "Press a key to begin", 18, WIDTH / 2, HEIGHT * 3 / 4)
    pygame.display.flip()
    waiting = True
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYUP:
                waiting = False

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (80, 200, 120)
BLUE = (0, 0, 255)
GRAY = (217, 217, 217)

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
font_name = pygame.font.match_font('arial')

# настройка папки ассетов
game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, 'img')
grass_img = pygame.image.load(os.path.join(img_folder, 'grass.png')).convert()
wall_img = pygame.image.load(os.path.join(img_folder, 'wall.jpg')).convert()
forest_img = pygame.image.load(os.path.join(img_folder, 'forest.png')).convert()
mine_img = pygame.image.load(os.path.join(img_folder, 'mine.png')).convert()
coin_img = pygame.image.load(os.path.join(img_folder, 'coin.png')).convert()
rect_img = pygame.image.load(os.path.join(img_folder, 'rect.png')).convert()

hero1_img = pygame.image.load(os.path.join(img_folder, 'hero1.jpg')).convert()
hero2_img = pygame.image.load(os.path.join(img_folder, 'hero2.jpg')).convert()
hero3_img = pygame.image.load(os.path.join(img_folder, 'hero3.jpg')).convert()
hero4_img = pygame.image.load(os.path.join(img_folder, 'hero4.png')).convert()

pygame.mixer.init()
music_folder = os.path.join(game_folder, 'music')
pygame.mixer.music.load(os.path.join(music_folder, 'hero.mp3'))

clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()

init_Wall()
init_Grass()
board = [input().split() for i in range(18)]

init_forest(board, 16, 32, 100)
init_mine(board, 16, 32, 100)
init_score()

hero1 = Hero(100, 0, 0, hero1_img)
hero2 = Hero(100, 31, 0, hero2_img)
hero3 = Hero(100, 0, 15, hero3_img)
hero4 = Hero(100, 31, 15, hero4_img)

all_sprites.add(hero1)
all_sprites.add(hero2)
all_sprites.add(hero3)
all_sprites.add(hero4)

all_sprites.add(Mini_hero(100, 35, -2, hero1_img))
all_sprites.add(Mini_hero(100, 35, 0, hero2_img))
all_sprites.add(Mini_hero(100, 35, 2, hero3_img))
all_sprites.add(Mini_hero(100, 35, 4, hero4_img))

show_go_screen()
pygame.mixer.music.play(loops=-1)

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
    screen.fill(GREEN)
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