import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

def simulate(input, game_step=1):
    assert input.dtype == torch.bool
    live_neighbors_weights = torch.ones(1, 1, 3, 3, dtype=torch.int8)
    live_neighbors_weights[0, 0, 1, 1] = 0

    x = input
    for i in range(game_step):
        live_neighbors_count = F.conv2d(x.type(torch.int8), live_neighbors_weights, padding=1)
        assert x.shape == live_neighbors_count.shape

        live = live_neighbors_count == 3
        survive = torch.logical_and(x, live_neighbors_count == 2)
        x = torch.logical_or(live, survive)

    return x

class RandomDataset(IterableDataset):
    def __init__(self, game_step, board_size, density, batch_size, batch_count=None):
        super().__init__()

        self.game_step = game_step
        self.board_size = board_size
        self.density = density
        self.batch_size = batch_size
        self.batch_count = batch_count

    def __iter__(self):
        remained_batch_count = self.batch_count

        while remained_batch_count is None or remained_batch_count > 0:
            if remained_batch_count is not None:
                remained_batch_count -= 1

            random_inputs = torch.rand(self.batch_size, 1, self.board_size, self.board_size) < self.density
            simulation_outputs = simulate(random_inputs, game_step=self.game_step)
            yield random_inputs, simulation_outputs

class MinimalDataset(Dataset):
    def __init__(self):
        super().__init__()

        board_size = 3
        minimal_inputs = list(itertools.product([False, True], repeat=board_size * board_size))
        self.inputs = torch.tensor(minimal_inputs).view(len(minimal_inputs), 1, board_size, board_size)
        self.simulation_outputs = simulate(self.inputs)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index == 0:
            return self.inputs, self.simulation_outputs
        else:
            raise IndexError
