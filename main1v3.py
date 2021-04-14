import os
import sys
import time as tm
import gym
import numpy as np
import pygame
from gym import spaces
from pygame import *
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, hero):
        self.hero = hero
        state = States()

        self.get_legal_actions = get_legal_actions
        self.qtable = np.zeros((state.get_nstates(), state.get_nactions()))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def update(self, state, action, reward, next_state):

        gamma = self.discount
        learning_rate = self.alpha

        self.qtable[state, action] = self.qtable[state, action] + learning_rate * (
                reward + gamma * np.max(self.qtable[next_state, :]) - self.qtable[state, action])

    def get_action(self, state):

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if random.random() > epsilon:
            action = np.argmax(self.qtable[state, :])
            # Else doing a random choice --> exploration
        else:
            action = random.choice(possible_actions)

        return action


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        state = States()
        self.n_action = state.get_nactions()
        self.action_space = spaces.Discrete(self.n_action)
        # Example for using image as input:
        self.n_spaces = state.get_nstates()
        self.observation_space = spaces.Discrete(self.n_spaces)

    def step(self, action, actor):
        reward = 0
        info = []
        done = True
        if action == 0:
            actor.move_right()
            obs = actor.states()
        elif action == 1:
            actor.move_left()
            obs = actor.states()
        elif action == 2:
            actor.move_up()
            obs = actor.states()
        elif action == 3:
            actor.move_down()
            obs = actor.states()
        elif action == 4:

            reward += actor.mine()
            obs = actor.states()
        elif action == 5:

            reward += actor.attack()
            obs = actor.states()
        return obs, reward, done, info

    def reset(self):

        sys.stdin = open('input.txt', 'r')
        board = [input().split() for i in range(18)]
        hero1.reset()
        hero2.reset()
        hero3.reset()
        hero4.reset()
        sys.stdin.close()
        agents.clear()
        agents.append(agent1)
        agents.append(agent2)
        agents.append(agent3)
        agents.append(agent4)
        return board  # reward, done, info can't be included

    def render(mode='human'):
        # Обновление
        all_sprites.update()
        # Рендеринг
        screen.fill(GREEN)
        all_sprites.draw(screen)
        for i in range(0, WIDTH, 40):
            pygame.draw.line(screen, RED, [i, 0], [i, HEIGHT], 1)
        for i in range(0, HEIGHT, 40):
            pygame.draw.line(screen, RED, [0, i], [WIDTH, i], 1)
        # После отрисовки всего, переворачиваем экран
        pygame.display.flip()


class States():
    def __init__(self):
        self.cod_action = {}
        self.cod_status = {}
        """ sost = ['Nw', 'Ng', 'Nf', 'Nm', 'Eh', 'Fm']
        dh = ['Dh', 'Ch']
        dw = ['Dw', 'Cw']
        cnt = 0
        for i1 in range(len(sost)):
            for i2 in range(len(sost)):
                for i3 in range(len(sost)):
                    for i4 in range(len(sost)):
                        for i5 in range(len(dh)):
                            for i6 in range(len(dw)):
                                self.cod_status[sost[i1] + sost[i2] + sost[i3] + sost[i4]+dh[i5]+dw[i6]] = cnt
                                cnt += 1"""
        enemy = ['Cl', 'Nl']
        cnt = 0
        for i1 in range(18):
            for i2 in range(34):
                for i3 in range(len(enemy)):
                    self.cod_status[str(i1) + str(i2) + enemy[i3]] = cnt
                    cnt += 1


        self.cod_action['speed_x_up'] = 0
        self.cod_action['speed_x_down'] = 1
        self.cod_action['speed_y_up'] = 2
        self.cod_action['speed_y_down'] = 3
        self.cod_action['mine'] = 4
        self.cod_action['attack'] = 5

        self.weight = np.zeros((len(self.cod_status), len(self.cod_action)))

    def get_nactions(self):
        return len(self.cod_action)

    def get_nstates(self):
        return len(self.cod_status)


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
        self.alive = True
        self.money = 100
        self.speed_x = 0
        self.speed_y = 0
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(img, (width, height))
        self.image.set_colorkey(self.image.get_at((0, 0)), RLEACCEL)
        self.rect = self.image.get_rect()
        self.bx = start + x * 40
        self.by = start + y * 40
        self.x = start + x * 40
        self.y = start + y * 40
        self.rect.center = (self.x, self.y)
        self.Hero_states = States()
        self.state = self.states()

    def reset(self):
        self.x = self.bx
        self.y = self.by
        self.alive = True
        self.money = 100
        self.speed_x = 0
        self.speed_y = 0
        self.rect.center = (self.x, self.y)

    def states(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            sost = ''
            start = 40
            sost += str((self.rect.y - start) // 40)
            sost += str((self.rect.x - start) // 40)


            # верх
            if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][1] == 'h':
                sost += 'Cl'

            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][1] == 'h':
                sost += 'Cl'

            # низ
            elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][1] == 'h':
                sost += 'Cl'

            # лево
            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][1] == 'h':
                sost += 'Cl'
            else:
                sost += 'Nl'

            return self.Hero_states.cod_status[sost]
        if not self.alive:
            self.speed_x = 0
            self.speed_y = 0
            self.rect.x = -1000
            self.rect.y = -1000
            return 0
        return 0

    def move_up(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40] == 'Ng':
                self.speed_y = -5
                self.speed_x = 0
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40], \
                board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40] = \
                    board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40], \
                    board[(self.rect.y - start) // 40][(self.rect.x - start) // 40]
            else:
                return -20
            return 0
        return -10

    def move_down(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40] == 'Ng':
                self.speed_y = 5
                self.speed_x = 0
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40], \
                board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40] \
                    = board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40], \
                      board[(self.rect.y - start) // 40][(self.rect.x - start) // 40]
            else:
                return -20
            return 0
        return -10

    def move_left(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1] == 'Ng':
                self.speed_y = 0
                self.speed_x = -5
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40], \
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1] \
                    = board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1], \
                      board[(self.rect.y - start) // 40][(self.rect.x - start) // 40]
            else:
                return -20
            return 0
        return -10

    def move_right(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1] == 'Ng':
                self.speed_y = 0
                self.speed_x = 5
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40], \
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1] \
                    = board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1], \
                      board[(self.rect.y - start) // 40][(self.rect.x - start) // 40]
            else:
                return -20
            return 0
        return -10

    def mine(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40] == 'Fm':
                self.money += 50
                board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40] = 'Nm'

            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1] == 'Fm':
                self.money += 50
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1] = 'Nm'

            elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40] == 'Fm':
                self.money += 50
                board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40] = 'Nm'

            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1] == 'Fm':
                self.money += 50
                board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1] = 'Nm'

            else:
                return -1
            return 100
        return 0

    def attack(self):
        global board
        if (self.rect.x - 80) % 40 == 0 and (self.rect.y - 80) % 40 == 0 and self.alive:
            self.speed_x = 0
            self.speed_y = 0
            start = 40
            if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][1] == 'h':
                if board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] == 'A':
                    if self.money > hero1.money:
                        self.money += hero1.money // 4
                        hero1.money = 0
                        board[(hero1.rect.y - start) // 40][(hero1.rect.x - start) // 40] = 'Ng'
                        hero1.alive = False
                        hero1.speed_x = 0
                        hero1.speed_y = 0
                        hero1.rect.x = -1000
                        hero1.rect.y = -1000
                        return -50
                    elif self.money < hero1.money:
                        hero1.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero1.money -= hero1.money // 4
                        self.money -= self.money // 4
                        return -1000

                elif board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] == 'B':
                    if self.money > hero2.money:
                        self.money += hero2.money // 4
                        hero2.money = 0
                        board[(hero2.rect.y - start) // 40][(hero2.rect.x - start) // 40] = 'Ng'
                        hero2.alive = False
                        hero2.speed_x = 0
                        hero2.speed_y = 0
                        hero2.rect.x = -1000
                        hero2.rect.y = -1000
                        return -50
                    elif self.money < hero2.money:
                        hero2.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero2.money -= hero2.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] == 'C':
                    if self.money > hero3.money:
                        self.money += hero3.money // 4
                        hero3.money = 0
                        board[(hero3.rect.y - start) // 40][(hero3.rect.x - start) // 40] = 'Ng'
                        hero3.alive = False
                        hero3.speed_x = 0
                        hero3.speed_y = 0
                        hero3.rect.x = -1000
                        hero3.rect.y = -1000
                        return -50
                    elif self.money < hero3.money:
                        hero3.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero3.money -= hero3.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40 - 1][(self.rect.x - start) // 40][0] == 'D':
                    if self.money > hero4.money:
                        self.money += hero4.money // 4
                        hero4.money = 0
                        board[(hero4.rect.y - start) // 40][(hero4.rect.x - start) // 40] = 'Ng'
                        hero4.alive = False
                        hero4.speed_x = 0
                        hero4.speed_y = 0
                        hero4.rect.x = -1000
                        hero4.rect.y = -1000
                        return -50
                    elif self.money < hero4.money:
                        hero4.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero4.money -= hero4.money // 4
                        self.money -= self.money // 4
                        return -100

            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][1] == 'h':
                if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] == 'A':
                    if self.money > hero1.money:
                        self.money += hero1.money // 4
                        hero1.money = 0
                        board[(hero1.rect.y - start) // 40][(hero1.rect.x - start) // 40] = 'Ng'
                        hero1.alive = False
                        hero1.speed_x = 0
                        hero1.speed_y = 0
                        hero1.rect.x = -1000
                        hero1.rect.y = -1000
                        return -50
                    elif self.money < hero1.money:
                        hero1.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero1.money -= hero1.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] == 'B':
                    if self.money > hero2.money:
                        self.money += hero2.money // 4
                        hero2.money = 0
                        board[(hero2.rect.y - start) // 40][(hero2.rect.x - start) // 40] = 'Ng'
                        hero2.alive = False
                        hero2.speed_x = 0
                        hero2.speed_y = 0
                        hero2.rect.x = -1000
                        hero2.rect.y = -1000
                        return -50
                    elif self.money < hero2.money:
                        hero2.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero2.money -= hero2.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] == 'C':
                    if self.money > hero3.money:
                        self.money += hero3.money // 4
                        hero3.money = 0
                        board[(hero3.rect.y - start) // 40][(hero3.rect.x - start) // 40] = 'Ng'
                        hero3.alive = False
                        hero3.speed_x = 0
                        hero3.speed_y = 0
                        hero3.rect.x = -1000
                        hero3.rect.y = -1000
                        return -50
                    elif self.money < hero3.money:
                        hero3.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero3.money -= hero3.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 + 1][0] == 'D':
                    if self.money > hero4.money:
                        self.money += hero4.money // 4
                        hero4.money = 0
                        board[(hero4.rect.y - start) // 40][(hero4.rect.x - start) // 40] = 'Ng'
                        hero4.alive = False
                        hero4.speed_x = 0
                        hero4.speed_y = 0
                        hero4.rect.x = -1000
                        hero4.rect.y = -1000
                        return -50
                    elif self.money < hero4.money:
                        hero4.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero4.money -= hero4.money // 4
                        self.money -= self.money // 4
                        return -100

            elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][1] == 'h':
                if board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] == 'A':
                    if self.money > hero1.money:
                        self.money += hero1.money // 4
                        hero1.money = 0
                        board[(hero1.rect.y - start) // 40][(hero1.rect.x - start) // 40] = 'Ng'
                        hero1.alive = False
                        hero1.speed_x = 0
                        hero1.speed_y = 0
                        hero1.rect.x = -1000
                        hero1.rect.y = -1000
                        return -50
                    elif self.money < hero1.money:
                        hero1.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero1.money -= hero1.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] == 'B':
                    if self.money > hero2.money:
                        self.money += hero2.money // 4
                        hero2.money = 0
                        board[(hero2.rect.y - start) // 40][(hero2.rect.x - start) // 40] = 'Ng'
                        hero2.alive = False
                        hero2.speed_x = 0
                        hero2.speed_y = 0
                        hero2.rect.x = -1000
                        hero2.rect.y = -1000
                        return -50
                    elif self.money < hero2.money:
                        hero2.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero2.money -= hero2.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] == 'C':
                    if self.money > hero3.money:
                        self.money += hero3.money // 4
                        hero3.money = 0
                        board[(hero3.rect.y - start) // 40][(hero3.rect.x - start) // 40] = 'Ng'
                        hero3.alive = False
                        hero3.speed_x = 0
                        hero3.speed_y = 0
                        hero3.rect.x = -1000
                        hero3.rect.y = -1000
                        return -50
                    elif self.money < hero3.money:
                        hero3.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero3.money -= hero3.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40 + 1][(self.rect.x - start) // 40][0] == 'D':
                    if self.money > hero4.money:
                        self.money += hero4.money // 4
                        hero4.money = 0
                        board[(hero4.rect.y - start) // 40][(hero4.rect.x - start) // 40] = 'Ng'
                        hero4.alive = False
                        hero4.speed_x = 0
                        hero4.speed_y = 0
                        hero4.rect.x = -1000
                        hero4.rect.y = -1000
                        return -50
                    elif self.money < hero4.money:
                        hero4.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero4.money -= hero4.money // 4
                        self.money -= self.money // 4
                        return -100

            elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][1] == 'h':
                if board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] == 'A':
                    if self.money > hero1.money:
                        self.money += hero1.money // 4
                        hero1.money = 0
                        board[(hero1.rect.y - start) // 40][(hero1.rect.x - start) // 40] = 'Ng'
                        hero1.alive = False
                        hero1.speed_x = 0
                        hero1.speed_y = 0
                        hero1.rect.x = -1000
                        hero1.rect.y = -1000
                        return -50
                    elif self.money < hero1.money:
                        hero1.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero1.money -= hero1.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] == 'B':
                    if self.money > hero2.money:
                        self.money += hero2.money // 4
                        hero2.money = 0
                        board[(hero2.rect.y - start) // 40][(hero2.rect.x - start) // 40] = 'Ng'
                        hero2.alive = False
                        hero2.speed_x = 0
                        hero2.speed_y = 0
                        hero2.rect.x = -1000
                        hero2.rect.y = -1000
                        return -50
                    elif self.money < hero2.money:
                        hero2.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero2.money -= hero2.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] == 'C':
                    if self.money > hero3.money:
                        self.money += hero3.money // 4
                        hero3.money = 0
                        board[(hero3.rect.y - start) // 40][(hero3.rect.x - start) // 40] = 'Ng'
                        hero3.alive = False
                        hero3.speed_x = 0
                        hero3.speed_y = 0
                        hero3.rect.x = -1000
                        hero3.rect.y = -1000
                        return -50
                    elif self.money < hero3.money:
                        hero3.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero3.money -= hero3.money // 4
                        self.money -= self.money // 4
                        return -100

                elif board[(self.rect.y - start) // 40][(self.rect.x - start) // 40 - 1][0] == 'D':
                    if self.money > hero4.money:
                        self.money += hero4.money // 4
                        hero4.money = 0
                        board[(hero4.rect.y - start) // 40][(hero4.rect.x - start) // 40] = 'Ng'
                        hero4.alive = False
                        hero4.speed_x = 0
                        hero4.speed_y = 0
                        hero4.rect.x = -1000
                        hero4.rect.y = -1000
                        return -50
                    elif self.money < hero4.money:
                        hero4.money += self.money // 4
                        self.money = 0
                        board[(self.rect.y - start) // 40][(self.rect.x - start) // 40] = 'Ng'
                        self.alive = False
                        self.speed_x = 0
                        self.speed_y = 0
                        self.rect.x = -1000
                        self.rect.y = -1000
                        return -200
                    else:
                        hero4.money -= hero4.money // 4
                        self.money -= self.money // 4
                        return -100

            else:
                return -100
        return -100

    def update(self):
        if not self.alive:
            self.speed_x = 0
            self.speed_y = 0
            self.rect.x = -1000
            self.rect.y = -1000

        self.rect.x += self.speed_x
        self.rect.y += self.speed_y


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
                mine.append([i, j])


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


def draw_score(surf, text, size, x, y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, GOLD)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


def show_go_screen(games):
    draw_text(screen, "Game %d" % games, 64, WIDTH / 2, HEIGHT / 4)
    # draw_text(screen, "Arrow keys move, Space to fire", 22, WIDTH / 2, HEIGHT / 2)
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

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (80, 200, 120)
BLUE = (0, 0, 255)
GRAY = (217, 217, 217)
GOLD = (255, 215, 0)

games = 0

x = 80
y = 80
WIDTH = 1561  # ширина игрового окна
HEIGHT = 801  # высота игрового окна
FPS = 240  # частота кадров в секунду
width = 40
height = 40

pygame.init()
infoObject = pygame.display.Info()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("success_code+w")
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

hero1_img = pygame.image.load(os.path.join(img_folder, 'hero1.png')).convert()
hero2_img = pygame.image.load(os.path.join(img_folder, 'hero2.png')).convert()
hero3_img = pygame.image.load(os.path.join(img_folder, 'hero3.png')).convert()
hero4_img = pygame.image.load(os.path.join(img_folder, 'hero4.png')).convert()

pygame.mixer.init()
music_folder = os.path.join(game_folder, 'music')
# pygame.mixer.music.load(os.path.join(music_folder, 'hero.mp3'))

clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()

init_Wall()
init_Grass()
board = [input().split() for i in range(18)]

mine = []

init_forest(board, 16, 32, 100)
init_mine(board, 16, 32, 100)
init_score()
agents = []
hero1 = Hero(100, 3, 3, hero1_img)
hero2 = Hero(100, 28, 3, hero2_img)
hero3 = Hero(100, 28, 12, hero3_img)
hero4 = Hero(100, 3, 12, hero4_img)

all_sprites.add(hero1)
all_sprites.add(hero2)
all_sprites.add(hero3)
all_sprites.add(hero4)
time_fill_mine = 0

all_sprites.add(Mini_hero(100, 35, -2, hero1_img))
all_sprites.add(Mini_hero(100, 35, 0, hero2_img))
all_sprites.add(Mini_hero(100, 35, 2, hero3_img))
all_sprites.add(Mini_hero(100, 35, 4, hero4_img))
env = CustomEnv()
GAMMA = 0.99
ALPHA = 0.6
EPS = 0.71
# pygame.mixer.music.play(loops=-1)
agent1 = QLearningAgent(alpha=ALPHA, epsilon=EPS, discount=GAMMA,
                        get_legal_actions=lambda s: range(env.n_action), hero=hero1)
agent2 = QLearningAgent(alpha=ALPHA, epsilon=EPS, discount=GAMMA,
                        get_legal_actions=lambda s: range(env.n_action), hero=hero2)
agent3 = QLearningAgent(alpha=ALPHA, epsilon=EPS, discount=GAMMA,
                        get_legal_actions=lambda s: range(env.n_action), hero=hero3)
agent4 = QLearningAgent(alpha=ALPHA, epsilon=EPS, discount=GAMMA,
                        get_legal_actions=lambda s: range(env.n_action), hero=hero4)
agents.append(agent1)
agents.append(agent2)
agents.append(agent3)
agents.append(agent4)

running = True
States()


def run(running, t, iterp):
    global time_fill_mine
    if time_fill_mine == 0:
        time_fill_mine = 8
        for el in mine:
            board[el[0]][el[1]] = "Fm"
    time_fill_mine -= 1
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

        # Обновление
    all_sprites.update()
    # Рендеринг
    screen.fill(GREEN)
    all_sprites.draw(screen)
    for i in range(80, 35 * 40, 40):
        pygame.draw.line(screen, GREEN, [i, 80], [i, 18 * 40], 1)
    for i in range(80, 19 * 40, 40):
        pygame.draw.line(screen, GREEN, [80, i], [34 * 40, i], 1)
    draw_score(screen, "%d" % (hero1.money), 42, WIDTH - 80, HEIGHT / 4 - 165)
    draw_score(screen, "%d" % (hero2.money), 42, WIDTH - 80, HEIGHT / 4 - 85)
    draw_score(screen, "%d" % (hero3.money), 42, WIDTH - 80, HEIGHT / 4 - 5)
    draw_score(screen, "%d" % (hero4.money), 42, WIDTH - 80, HEIGHT / 4 + 75)
    draw_score(screen, "номер", 42, WIDTH - 80, HEIGHT / 4 + 130)
    draw_score(screen, "итерации", 42, WIDTH - 80, HEIGHT / 4 + 165)
    draw_score(screen, "%d" % t, 42, WIDTH - 80, HEIGHT / 4 + 205)
    draw_score(screen, "номер игры", 42, WIDTH - 90, HEIGHT / 4 + 250)
    draw_score(screen, "%d" % iterp, 42, WIDTH - 80, HEIGHT / 4 + 290)
    # После отрисовки всего, переворачиваем экран
    pygame.display.flip()


def play_and_train(env, agents, iterp, t_max=10 ** 2*5):
    total_reward = 0.0
    global board
    board = env.reset()
    #t_max = 10 ** 1 + 10 * iterp // 10
    for t in range(t_max):
        if t % 50 == 0:
            np.savetxt('test_1.txt', agent3.qtable)
        prevagent = agents[-1]
        for agent_n in agents:
            if not agent_n.hero.alive:
                s = agent_n.hero.state
                run(done, t, iterp)
                continue
            for i in range(18):
                for j in range(34):
                    if board[i][j] == 'Ah' and not hero1.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Bh' and not hero2.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Ch' and not hero3.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Dh' and not hero4.alive:
                        board[i][j] = 'Ng'

            s = agent_n.hero.states()
            a = agent_n.get_action(s)
            b = np.array(board)

            if prevagent == agent_n:
                continue
            prevagent = agent_n

            next_s, r, done, _ = env.step(a, agent_n.hero)

            run(done, t, iterp)
            agent_n.update(s, a, r, next_s)

            total_reward += r
            if not done:
                break

    return total_reward


def play_not_train(env, agents, iterp, t_max=10 ** 2*2):
    total_reward = 0.0
    global board
    board = env.reset()
    for t in range(t_max):
        prevagent = agents[-1]
        for agent_n in agents:
            if not agent_n.hero.alive:
                s = agent_n.hero.state
                run(done, t, iterp)
                continue
            for i in range(18):
                for j in range(34):
                    if board[i][j] == 'Ah' and not hero1.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Bh' and not hero2.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Ch' and not hero3.alive:
                        board[i][j] = 'Ng'
                    if board[i][j] == 'Dh' and not hero4.alive:
                        board[i][j] = 'Ng'

            s = agent_n.hero.states()
            a = agent_n.get_action(s)
            if prevagent == agent_n:
                continue
            prevagent = agent_n
            next_s, r, done, _ = env.step(a, agent_n.hero)

            run(done, t, iterp)

            total_reward += r
            if not done:
                break

    return total_reward


rewards = []
train = False
if not train:
    game = 50
    agent1.qtable = np.loadtxt(f'weight3/wsg{game}agent0.txt')
    agent2.qtable = np.loadtxt(f'weight3/wsg{game}agent1.txt')
    agent3.qtable = np.loadtxt(f'weight3/wsg{game}agent2.txt')
    agent4.qtable = np.loadtxt(f'weight3/wsg{game}agent3.txt')
    for agent in agents:
        agent.epsilon *= (0.99 ** game)
    print(play_not_train(env, agents, game))
else:
    for i in range(1000):
        rewards.append(play_and_train(env, agents, i))
        for agent in agents:
            agent.epsilon *= 0.99
        if i % 10 == 0:
            for k in range(len(agents)):
                np.savetxt(f'weight3/wsg{i}agent{k}.txt', agents[k].qtable)
        print(rewards[i])
# до сюда
# делает лево право хотя не суетолог

sys.stdin.close()
sys.stdout.close()
