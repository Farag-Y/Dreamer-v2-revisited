import torch
from torch import nn
from torch.nn import functional as F


class ObservationModel(nn.Module):
    def __init__(
        self,
        belief_size: int,
        state_size: int,
        cnn_depth: int,
        image_channels: int = 3,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, non_linearity)
        self.embedding_size = 32 * cnn_depth
        self.fc1 = nn.Linear(belief_size + state_size, self.embedding_size)
        self.conv1 = nn.ConvTranspose2d(self.embedding_size, 4 * cnn_depth, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(4 * cnn_depth, 2 * cnn_depth, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(2 * cnn_depth, cnn_depth, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(cnn_depth, image_channels, 6, stride=2)

    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation
