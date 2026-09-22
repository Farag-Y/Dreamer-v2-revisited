from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RSSMOutput:
    det_hidden_states:  torch.Tensor
    prior_states:       torch.Tensor
    prior_logits:        torch.Tensor
    posterior_states:   torch.Tensor | None = None
    posterior_logits:    torch.Tensor | None = None


class RSSM(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        belief_size: int,
        action_size: int,
        num_categorical:int,
        num_classes:int,
        obs_size: int,
        non_linearity: str = 'relu',
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, non_linearity)
        self.num_categorical = num_categorical
        self.num_classes=num_classes
        self.fc_embed_state_action     = nn.Linear(state_size + action_size, belief_size)
        self.rnn                       = nn.GRUCell(input_size=belief_size, hidden_size=belief_size)
        self.fc_embed_belief_prior     = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior            = nn.Linear(hidden_size, num_categorical*num_classes)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + obs_size, hidden_size)
        self.fc_state_posterior        = nn.Linear(hidden_size, num_categorical*num_classes)

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: torch.Tensor | None = None,
        nonterminals: torch.Tensor | None = None,
    ) -> RSSMOutput:
        sequence_length = actions.shape[0] +1
        num_classes = self.num_classes
        num_categorical = self.num_categorical
        (det_hidden_states, prior_states,prior_logits,
            posterior_states,posterior_logits) = (
                [[torch.empty(0)] * sequence_length for _ in range(5)]
            )
        #Belifes is the detemnistic hidden state
        det_hidden_states[0],prior_states[0],posterior_states[0] = prev_belief,prev_state,prev_state
        for t in range(actions.shape[0]):
            prev_state = prior_states[t] if observations is None else posterior_states[t]
            prev_state = prev_state if nonterminals is None else prev_state*nonterminals[t]
            hidden_input = self.act_fn(self.fc_embed_state_action(torch.concat((prev_state,actions[t]),dim=1)))## TODO: Why dimension 1 ?
            det_hidden_states[t+1]= self.rnn(hidden_input,det_hidden_states[t])
            #TODO: Also empty out hidden state on episode end ? 
            #det_hidden_states[t+1]= self.rnn(hidden_input, det_hidden_states[t] if nonterminals is None else det_hidden_states[t]*nonterminals[t])

            ## Prior
            hidden_prior = self.act_fn(self.fc_embed_belief_prior(det_hidden_states[t+1]))
            prior_logits[t+1] = self.fc_state_prior(hidden_prior).reshape(-1,num_categorical,num_classes)
            prior_sample = self.draw(prior_logits[t+1])
            prior_probs = F.softmax(prior_logits[t+1], dim=-1)
            prior_states[t+1] = prior_sample + prior_probs - prior_probs.detach()
            prior_states[t+1]=prior_states[t+1].reshape(-1,num_classes*num_categorical)

            ##Posterior 
            if observations is not None:
                hidden_posterior = self.act_fn(self.fc_embed_belief_posterior(torch.concat((det_hidden_states[t+1],observations[t]),dim=1)))
                posterior_logits[t+1] = self.fc_state_posterior(hidden_posterior).reshape(-1,num_categorical,num_classes)
                posterior_sample = self.draw(posterior_logits[t+1])
                posterior_probs = F.softmax(posterior_logits[t+1], dim=-1)
                posterior_states[t+1] = posterior_sample + posterior_probs - posterior_probs.detach()
                posterior_states[t+1]=posterior_states[t+1].reshape(-1,num_classes*num_categorical)
        return RSSMOutput(
            det_hidden_states=torch.stack(det_hidden_states[1:], dim=0),
            prior_states=torch.stack(prior_states[1:], dim=0),
            prior_logits=torch.stack(prior_logits[1:], dim=0),
            posterior_states=torch.stack(posterior_states[1:], dim=0) if observations is not None else None,
            posterior_logits=torch.stack(posterior_logits[1:], dim=0) if observations is not None else None,
        )
    #TODO: draw can move into utils alongside the actor_critic one too.
    def draw(self,logits: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.OneHotCategorical(logits=logits)
        return dist.sample()
