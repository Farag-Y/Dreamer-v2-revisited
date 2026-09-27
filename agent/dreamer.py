import torch
from omegaconf import DictConfig
from tqdm import tqdm

from agent.behavior import ActorCritic
from agent.world_model import WorldModel
from env_wrapper import BaseEnv
from experience_replay import ExperienceReplay


class Dreamer(torch.nn.Module):
    def __init__(
        self, world_model: WorldModel, behavior: ActorCritic, cfg: DictConfig, action_size: int, device: str,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.behavior = behavior
        self.cfg = cfg
        self.action_size = action_size
        self.device = device
        self.collected_steps = 0

    @classmethod
    def from_config(cls, cfg: DictConfig, env: BaseEnv, device: str) -> "Dreamer":
        world_model = WorldModel(
            cfg, action_size=env.action_size, image_channels=env.observation_size[0], device=device
        )
        behavior = ActorCritic(cfg, action_size=env.action_size, device=device)
        return cls(world_model, behavior, cfg, env.action_size, device)

    def act(
        self,
        env: BaseEnv,
        observation: torch.Tensor,
        belief: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        explore: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, bool, bool]:
        with torch.no_grad():
            encoded = self.world_model.encoder(observation.to(self.device))
            rssm_out = self.world_model.observe(action.unsqueeze(0), encoded.unsqueeze(0), belief, state)
            belief = rssm_out.det_hidden_states[-1]
            state = rssm_out.posterior_states[-1]
            action = self.behavior.act(belief, state, explore)
            if not env.discrete_actions:  # one-hot discrete actions need no clamping
                min_action, max_action = env.action_range
                action = action.clamp(min_action, max_action)
            next_obs, reward, done, terminated = env.step(action[0].cpu())
        return belief, state, action, next_obs, reward, done, terminated

    def collect_episode(
        self, env: BaseEnv, replay: ExperienceReplay, explore: bool = True,
    ) -> tuple[float, list[dict[str, float]]]:
        cfg = self.cfg
        belief = torch.zeros(1, cfg.belief_size, device=self.device)
        state  = torch.zeros(1, cfg.state_size,  device=self.device)
        action = torch.zeros(1, self.action_size, device=self.device)
        observation = env.reset()
        episode_reward = 0.0
        results = []
        for _ in tqdm(range(cfg.max_episode_length // cfg.action_repeat)):
            belief, state, action, next_obs, reward, done, terminated = self.act(
                env, observation, belief, state, action, explore)
            replay.append(observation, reward, action.squeeze(0).cpu(), terminated)
            episode_reward += reward
            observation = next_obs
            self.collected_steps += 1
            if self.collected_steps % cfg.train_every == 0:
                results.append(self.train_on_batch(replay))
            if done:
                replay.end_episode(observation)
                break
        return episode_reward, results

    def train_on_batch(self, replay: ExperienceReplay) -> dict[str, float]:
        cfg = self.cfg
        obs, actions, rewards, nonterminals, true_nonterminals = replay.sample(cfg.batch_size, cfg.chunk_size)
        wm_result = self.world_model.train_step(obs, actions, rewards, nonterminals, true_nonterminals)
        state, belief = wm_result.pop('state'), wm_result.pop('belief')
        behavior_result = self.behavior.train_step(state, belief, true_nonterminals[:-1], self.world_model)
        return {**wm_result, **behavior_result}
