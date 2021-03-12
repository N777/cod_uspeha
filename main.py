import sys
import random

def write_board_in_file():
    for el in board:
        for i in el:
            print('0' * int(len(str(i)) < 2) + str(i), end = ' ')
        print()

def check_place_for_smth(i, j):
    global board, mark, board_size
    step = []
    step.append([0, 0])
    step.append([0, 1])
    step.append([1, 0])
    step.append([0, -1])
    step.append([-1, 0])
    step.append([1, -1])
    step.append([-1, 1])
    step.append([1, 1])
    step.append([-1, -1])

    for l in range(9):
        if not(board[i + step[l][0]][j + step[l][1]] == 1 or board[i + step[l][0]][j + step[l][1]] == 0):
            return False
    return True

def bild_castle(ind, a, b):
    global board, mark
    for i in range(a, a + 3):
        for j in range(b, b + 3):
            board[i][j] = mark['castle' + str(ind)]

def generation_sawmill():
    global board, mark,  board_size

    sawmill_num = int(input())

    val = 0
    while val < sawmill_num:
        random.seed()
        i = random.randint(1,  board_size - 2)
        random.seed()
        j = random.randint(1,  board_size - 2)
        if check_place_for_smth(i, j):
            board[i][j] = mark['sawmill']
            val += 1

def generation_mine():
    global board, mark, board_size

    mine_num = int(input())

    val = 0
    while val < mine_num:
        random.seed()
        i = random.randint(1, board_size - 2)
        random.seed()
        j = random.randint(1, board_size - 2)
        if check_place_for_smth(i, j):
            board[i][j] = mark['mine']
            val += 1

def dfs_forest(i, j):
    global board, mark, board_size, bot_num, use

    step = []
    step.append([0, 1])
    step.append([1, 0])
    step.append([0, -1])
    step.append([-1, 0])
    step.append([1, -1])
    step.append([-1, 1])
    step.append([1, 1])
    step.append([-1, -1])

    if use[i][j] or (not use[i][j]) and board[i][j] != mark['forest']:
        return

    use[i][j] = 1
    for l in range(8):
        if 0 < i + step[l][0] < board_size - 1 and 0 < j + step[l][1] < board_size - 1:
                dfs_forest(i + step[l][0], j + step[l][1])

    return

def generation_forest():
    global board, mark, board_size, use

    forest_num = (board_size - 1) * (board_size - 1) // 4

    val = 0

    while val < forest_num:
        random.seed()
        i = random.randint(1, board_size - 2)
        random.seed()
        j = random.randint(1, board_size - 2)
        if board[i][j] == 0 or board[i][j] == 1:
            board[i][j] = mark['forest']
            val += 1

def main():
    global board, mark, board_size, bot_num
    sys.stdin = open('input.txt', 'r')
    sys.stdout = open('output.txt', 'w')

    bot_num = int(input())
    mark = {}
    mark['forest'] = 4
    mark['sawmill'] = 3
    mark['mine'] = 2
    mark['pain'] = 1
    mark['borders'] = 0

    for i in range(1, bot_num + 1):
        mark['warrior' + str(i)] = i * 10 + 7
        mark['worker' + str(i)] = i * 10 + 8
        mark['castle' + str(i)] = i * 10 + 9

    board_size = int(input())
    board = [[1 for i in range(board_size)] for i in range(board_size)]
    for i in range(board_size):
        board[0][i] = 0
        board[board_size - 1][i] = 0
        board[i][0] = 0
        board[i][board_size - 1] = 0

    if bot_num == 4:
        bild_castle(1, 1, 1)
        bild_castle(2, 1, board_size - 4)
        bild_castle(3, board_size - 4, board_size - 4)
        bild_castle(4, board_size - 4, 1)

    generation_mine()
    generation_sawmill()
    generation_forest()

    write_board_in_file()

    sys.stdin.close()
    sys.stdout.close()
    return


if __name__ == "__main__":
    main()