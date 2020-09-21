import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalModel(nn.Module):
    def __init__(self, game_step=1, channel_scale=1, initialization='random'):
        super().__init__()

        self.game_step = game_step

        if game_step != 1 and channel_scale != 1:
            self.input_conv = nn.Conv2d(1, 1 * channel_scale, 1)
        else:
            self.input_conv = None

        self.conv1 = nn.Conv2d(1 * channel_scale, 2 * channel_scale, 3, padding=1)
        self.conv2 = nn.Conv2d(2 * channel_scale, 1 * channel_scale, 1)
        self.output_conv = nn.Conv2d(1 * channel_scale, 1, 1)

        if initialization == 'ground_truth':
            assert channel_scale == 1, 'ground truth weight initialization does not support channel scale'

            with torch.no_grad():
                if self.input_conv is not None:
                    self.input_conv.weight.fill_(1.0)
                    self.input_conv.bias.fill_(0.0)

                assert self.conv1.weight.shape == (2, 1, 3, 3)
                self.conv1.weight.fill_(1.0)
                self.conv1.weight[0, 0, 1, 1] = 0.1

                assert self.conv1.bias.shape == (2,)
                self.conv1.bias[0] = -3.0
                self.conv1.bias[1] = -2.0

                assert self.conv2.weight.shape == (1, 2, 1, 1)
                self.conv2.weight[0, 0, 0, 0] = -10.0
                self.conv2.weight[0, 1, 0, 0] = 1.0

                self.conv2.bias.zero_()

                assert self.output_conv.weight.shape == (1, 1, 1, 1)
                self.output_conv.weight[0, 0, 0, 0] = 40.0

                assert self.output_conv.bias.shape == (1,)
                self.output_conv.bias[0] = -20.0

    def forward(self, x):
        if self.input_conv is not None:
            x = self.input_conv(x)

        for i in range(self.game_step):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)

        x = self.output_conv(x)
        return x
