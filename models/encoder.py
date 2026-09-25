import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
   def __init__(self, cnn_depth: int, image_channels: int = 3, non_linearity: str = 'relu') -> None:
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.output_size = 32 * cnn_depth
    self.conv1 = nn.Conv2d(image_channels, cnn_depth, 4, stride=2)
    self.conv2 = nn.Conv2d(cnn_depth, 2 * cnn_depth, 4, stride=2)
    self.conv3 = nn.Conv2d(2 * cnn_depth, 4 * cnn_depth, 4, stride=2)
    self.conv4 = nn.Conv2d(4 * cnn_depth, 8 * cnn_depth, 4, stride=2)

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    return hidden.reshape(-1, self.output_size)
