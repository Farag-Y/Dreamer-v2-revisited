import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
  def __init__(
      self,
      belief_size: int,
      state_size: int,
      hidden_size: int,
      action_size: int,
      non_linearity: str = 'relu',
  ) -> None:
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.action_size = action_size
    self.fc1 = nn.Linear(belief_size+state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)
    self.fc4 = nn.Linear(hidden_size,hidden_size)
    self.final_layer = nn.Linear(hidden_size,action_size)
  def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
      hidden = self.act_fn(self.fc1(torch.concat((belief,state),dim=1)))
      hidden = self.act_fn(self.fc2(hidden))
      hidden = self.act_fn(self.fc3(hidden))
      hidden = self.act_fn(self.fc4(hidden))
      logits = self.final_layer(hidden)
      return logits
  def sample(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
     logits = self.forward(belief,state)
     sample = self.draw(logits)
     probs = F.softmax(logits, dim=-1)
     # straight-through: value is the hard sample, gradient flows through probs only
     return sample + probs - probs.detach()
  def mode(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    logits = self.forward(belief,state)
    index = torch.argmax(logits, dim=-1)
    return F.one_hot(index, num_classes=self.action_size).float()
  def draw(self,logits: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.OneHotCategorical(logits=logits)
    return dist.sample()  # already a one-hot vector per batch element, e.g. tensor([0, 0, 1, 0, 0])




class Critic(nn.Module):
  def __init__(
      self,
      belief_size: int,
      state_size: int,
      hidden_size: int,
      non_linearity: str = 'relu',
  ) -> None:
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.fc1 = nn.Linear(belief_size+state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)
    self.fc4 = nn.Linear(hidden_size,1)


  def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    hidden = self.act_fn(self.fc1(torch.concat((belief,state),dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    value = self.fc4(hidden)
    return value.squeeze(-1)

