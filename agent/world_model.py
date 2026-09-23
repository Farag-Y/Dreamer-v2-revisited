import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.distributions import Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

from env_wrapper import TERMINATING_ENVS
from models.discount_model import DiscountModel
from models.encoder import Encoder
from models.observation_model import ObservationModel
from models.reward_model import RewardModel
from models.rssm import RSSM, RSSMOutput
from utils import model_wrapper


class WorldModel(nn.Module):
    def __init__(self, cfg: DictConfig, action_size: int, device: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.train_discount = cfg.env in TERMINATING_ENVS
        self.rssm = RSSM(
            state_size=cfg.state_size,
            hidden_size=cfg.hidden_size,
            belief_size=cfg.belief_size,
            num_categorical=cfg.num_categorical,
            num_classes=cfg.num_classes,
            action_size=action_size,
            obs_size=cfg.embedding_size,
            non_linearity=cfg.activation_function,
        ).to(device=device)
        self.decoder = ObservationModel(
            belief_size=cfg.belief_size,
            state_size=cfg.state_size,
            embedding_size=cfg.embedding_size,
        ).to(device=device)
        self.reward_model = RewardModel(
            belief_size=cfg.belief_size,
            state_size=cfg.state_size,
            hidden_size=cfg.dense_hidden_size,
            non_linearity=cfg.dense_activation_function,
        ).to(device=device)
        self.encoder = Encoder(embedding_size=cfg.embedding_size).to(device=device)
        self.discount_model = DiscountModel(
            state_size=cfg.state_size,
            belief_size=cfg.belief_size,
            hidden_size=cfg.dense_hidden_size,
            non_linearity=cfg.dense_activation_function,
        ).to(device=device)
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    def observe(
        self,
        actions: torch.Tensor,
        encoded_obs: torch.Tensor,
        init_belief: torch.Tensor,
        init_state: torch.Tensor,
        nonterminals: torch.Tensor | None = None,
    ) -> RSSMOutput:
        return self.rssm(init_state, actions, init_belief, encoded_obs, nonterminals)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        nonterminals: torch.Tensor,
        true_nonterminals: torch.Tensor,
    ) -> tuple[torch.Tensor, RSSMOutput, dict[str, float]]:
        cfg, device = self.cfg, self.device
        init_belief = torch.zeros(cfg.batch_size, cfg.belief_size, device=device)
        init_state = torch.zeros(cfg.batch_size, cfg.state_size, device=device)
        free_nats = torch.full((1,), cfg.free_nats, dtype=torch.float32, device=device)

        encoded_obs = model_wrapper(self.encoder, obs[1:])
        rssm_output: RSSMOutput = self.observe(actions[:-1], encoded_obs, init_belief, init_state, nonterminals[:-1])
        predicted_reward = model_wrapper(
            self.reward_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1
        )

        # New KL Divergence
        post_dist = Independent(OneHotCategorical(logits=rssm_output.posterior_logits), 1)
        post_dist_detached = Independent(OneHotCategorical(logits=rssm_output.posterior_logits.detach()), 1)
        prior_dist = Independent(OneHotCategorical(logits=rssm_output.prior_logits), 1)
        prior_dist_detached = Independent(OneHotCategorical(logits=rssm_output.prior_logits.detach()), 1)
        prior_kl_term = kl_divergence(post_dist_detached, prior_dist).mean()
        post_kl_term = kl_divergence(post_dist, prior_dist_detached).mean()
        prior_kl_term = torch.max(prior_kl_term, free_nats.squeeze())
        post_kl_term = torch.max(post_kl_term, free_nats.squeeze())
        kl_loss = cfg.kl_scale * (cfg.kl_balance_alpha * prior_kl_term + (1 - cfg.kl_balance_alpha) * post_kl_term)

        if self.train_discount:
            discount_logits = model_wrapper(
                self.discount_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1
            )
            discount_loss = F.binary_cross_entropy_with_logits(
                discount_logits, true_nonterminals[:-1].squeeze(-1), reduction="none"
            ).mean()
            discount_loss = discount_loss * cfg.discount_loss_scale  # Increasing the scale of discount loss.
        else:
            discount_loss = torch.tensor(0.0, device=device)

        decoded_obs = model_wrapper(
            self.decoder, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1
        )
        obs_loss = 0.5 * F.mse_loss(decoded_obs, obs[1:], reduction="none").sum((2, 3, 4)).mean()
        reward_loss = 0.5 * F.mse_loss(predicted_reward, rewards[:-1], reduction="none").mean()

        total_loss = kl_loss + obs_loss + reward_loss + discount_loss
        loss_components = {
            "kl_loss": kl_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "discount_loss": discount_loss.item(),
        }
        return total_loss, rssm_output, loss_components

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        nonterminals: torch.Tensor,
        true_nonterminals: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        self.optimizer.zero_grad()
        total_loss, rssm_output, loss_components = self.compute_loss(
            obs, actions, rewards, nonterminals, true_nonterminals
        )
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.cfg.grad_clip_norm)
        self.optimizer.step()
        return {
            **loss_components,
            "belief": rssm_output.det_hidden_states[-1],
            "state": rssm_output.posterior_states[-1],
        }
