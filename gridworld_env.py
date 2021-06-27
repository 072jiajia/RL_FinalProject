import numpy as np

class Env:
    def __init__(self, grid=(5, 5), seed=0):
        assert len(grid) == 2, "input x and y"
        assert grid[0] > 0 and grid[1] > 0, "input positive number"
        np.random.seed(seed)
        self.gridsize = grid
        self.goal = (grid[0] - 1, grid[1] - 1)
        self.position = [0, 0]

        # initialize grid
        self.grid = np.zeros((grid[0], grid[1]))
        self.grid[self.goal] = 2
        self.done = 0
        self.action_space = np.array([0,1,2,3])

    def reset(self, random_loc=True):
        '''
            observation: player: 1
                         reward: 2
                         else: 0
        '''

        self.grid = np.zeros(self.grid.shape)
        if random_loc is True:
            # random player position
            random_position = np.random.randint(self.gridsize[0] * self.gridsize[1] - 1)
            self.position = [random_position // self.gridsize[0], random_position % self.gridsize[1]]
        else:
            self.position = [0, 0]

        self.grid[self.position[0], self.position[1]] = 1
        self.grid[self.goal] = 2
        self.done = 0

        return self.grid

    def step(self, action, test=False):
        # action(up:0, down:1, right:2, left:3)
        original_position = self.position.copy()
        out_of_boundary = False
        # state are 2D array
        if action == 0: # go up
            if self.position[0] - 1 >= 0:
                self.position[0] = self.position[0] - 1
            else:
                out_of_boundary = True
        elif action == 1:  # go down
            if self.position[0] + 1 < self.gridsize[0]:
                self.position[0] = self.position[0] + 1
            else:
                out_of_boundary = True
        elif action == 2:  # go right
            if self.position[1] + 1 < self.gridsize[1]:
                self.position[1] = self.position[1] + 1
            else:
                out_of_boundary = True
        elif action == 3:  # go left
            if self.position[1] - 1 >= 0:
                self.position[1] = self.position[1] - 1
            else:
                out_of_boundary = True

        if self.position[0] == self.gridsize[0] - 1 and self.position[1] == self.gridsize[1] - 1:
            reward = 1
            done = 1
            self.position = original_position
        elif out_of_boundary:
            # print(self.position[0])
            reward = -1
            done = 1
        else:
            reward = 0
            done = 0
            self.grid[self.position[0], self.position[1]] = 1
            self.grid[original_position[0], original_position[1]] = 0

        return self.grid.copy(), reward, done

    def set_agent_loc(self, row, col):
        assert row < self.gridsize[0]
        assert col < self.gridsize[1]

        orig_agent_loc = self.position.copy()
        self.grid[orig_agent_loc[0], orig_agent_loc[1]] = 0
        self.position = [row, col]
        self.grid[row, col] = 1
        return self.grid.copy()


if __name__ == '__main__':
    env = Env()
    state = env.reset(random_loc=False)
    state = env.step(1)
    state = env.step(2)
    state = env.reset(random_loc=False)
    print(state)