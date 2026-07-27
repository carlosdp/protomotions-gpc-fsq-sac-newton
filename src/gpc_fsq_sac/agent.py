"""ProtoMotions lifecycle adapter for the released rsl_rl SAC implementation."""

from __future__ import annotations

import json
import logging
import time

import torch
from rsl_rl.algorithms import SAC
from rsl_rl.storage import ReplayBuffer
from tensordict import TensorDict

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.utils.training import aggregate_scalar_metrics

from .config import FSQSACAgentConfig
from .model import FSQSACModel

log = logging.getLogger(__name__)


class FSQSACAgent(BaseAgent):
    """Run released SAC updates while retaining ProtoMotions' environment lifecycle."""

    config: FSQSACAgentConfig
    require_reward_norm_on_load = False

    def _before_create_model(self) -> None:
        if self.fabric.world_size != 1:
            raise ValueError("The initial FSQ-SAC integration supports exactly one GPU.")
        obs = self.add_agent_info_to_obs(self.env.get_obs())
        self._setup_obs_td = self.obs_dict_to_tensordict(obs)

    def create_model(self) -> FSQSACModel:
        return FSQSACModel(
            config=self.config.model,
            obs=self._setup_obs_td,
            action_dim=self.env.robot_config.number_of_actions,
        )

    @staticmethod
    def replay_bytes_per_transition(obs: TensorDict, action_dim: int) -> int:
        # The released buffer stores float32 obs, action, reward, next_obs, done,
        # and bootstrap tensors.
        obs_elements = sum(value[0].numel() for value in obs.values())
        elements = (2 * obs_elements) + action_dim + 3
        return elements * torch.tensor([], dtype=torch.float32).element_size()

    def _profile_replay(self, obs: TensorDict, requested_size: int) -> dict:
        bytes_per_transition = self.replay_bytes_per_transition(
            obs,
            self.env.robot_config.number_of_actions,
        )
        budget_bytes = int(self.config.replay_memory_limit_gib * (1024**3))
        budget_transitions = budget_bytes // bytes_per_transition
        per_env = max(
            self.config.n_steps,
            min(requested_size, budget_transitions) // self.num_envs,
        )
        effective_size = per_env * self.num_envs
        allocation_bytes = effective_size * bytes_per_transition
        return {
            "requested_transitions": requested_size,
            "effective_transitions": effective_size,
            "transitions_per_env": per_env,
            "bytes_per_transition": bytes_per_transition,
            "allocation_bytes": allocation_bytes,
            "allocation_gib": allocation_bytes / (1024**3),
            "memory_limit_gib": self.config.replay_memory_limit_gib,
            "observation_shapes": {
                key: list(value.shape[1:]) for key, value in obs.items()
            },
            "action_dim": self.env.robot_config.number_of_actions,
            "checkpoint_replay_buffer": self.config.checkpoint_replay_buffer,
        }

    def create_optimizers(self, model: FSQSACModel) -> None:
        self.replay_profile = self._profile_replay(
            self._setup_obs_td,
            self.config.replay_buffer_size,
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.root_dir / "replay_profile.json"
        profile_path.write_text(json.dumps(self.replay_profile, indent=2) + "\n")
        log.info("Replay profile: %s", json.dumps(self.replay_profile, sort_keys=True))

        replay_buffer = ReplayBuffer(
            num_envs=self.num_envs,
            num_transitions_per_env=self.config.num_steps,
            obs=self._setup_obs_td,
            actions_shape=[self.env.robot_config.number_of_actions],
            device=self.device,
            buffer_size=self.replay_profile["effective_transitions"],
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
        )
        self.algorithm = SAC(
            actor=model.actor,
            critic=model.critic,
            replay_buffer=replay_buffer,
            replay_buffer_size=self.replay_profile["effective_transitions"],
            num_learning_epochs=self.config.num_learning_epochs,
            num_mini_batches=self.config.num_mini_batches,
            mini_batch_size=self.config.mini_batch_size,
            actor_learning_rate=self.config.actor_learning_rate,
            critic_learning_rate=self.config.critic_learning_rate,
            alpha_learning_rate=self.config.alpha_learning_rate,
            auto_alpha=self.config.auto_alpha,
            alpha=self.config.initial_alpha,
            tau=self.config.tau,
            gamma=self.config.gamma,
            target_entropy_scale=self.config.target_entropy_scale,
            device=str(self.device),
            max_grad_norm=self.config.max_grad_norm,
            policy_frequency=self.config.policy_frequency,
            n_steps=self.config.n_steps,
        )
        self._diagnostic_batch = None

    @staticmethod
    def _gradient_norm(parameters) -> torch.Tensor:
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return torch.tensor(0.0)
        return torch.stack(gradients).norm(2)

    def _maybe_create_diagnostic_batch(self) -> None:
        if self._diagnostic_batch is not None:
            return
        replay_buffer = self.algorithm.replay_buffer
        transitions = replay_buffer.num_transitions * self.num_envs
        minimum = max(
            self.config.diagnostic_batch_size,
            self.config.n_steps * self.num_envs,
        )
        if transitions < minimum:
            return
        valid_indices = replay_buffer._generate_valid_indices()
        self._diagnostic_batch = replay_buffer._generate_batch(
            valid_indices,
            self.config.diagnostic_batch_size,
        )
        cpu_batch = [
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else value.detach().cpu()
            for value in self._diagnostic_batch
        ]
        torch.save(cpu_batch, self.root_dir / "diagnostic_batch.pt")

    def _sac_diagnostics(self) -> dict[str, torch.Tensor]:
        self._maybe_create_diagnostic_batch()
        if self._diagnostic_batch is None:
            return {}

        (
            obs_batch,
            actions_batch,
            rewards_batch,
            next_obs_batch,
            dones_batch,
            bootstrap_batch,
            effective_n_steps,
        ) = self._diagnostic_batch
        with torch.no_grad():
            new_actions, log_prob = self.model.actor.sample_action_logp(obs_batch)
            q1, q2 = self.model.critic.evaluate_all_q(obs_batch, new_actions)

            next_actions, next_log_prob = self.model.actor.sample_action_logp(
                next_obs_batch
            )
            target_q1, target_q2 = self.model.critic.evaluate_all_target_q(
                next_obs_batch,
                next_actions,
            )
            bootstrap_mask = bootstrap_batch + 1 - dones_batch
            discount = torch.pow(
                self.config.gamma,
                effective_n_steps.to(dtype=target_q1.dtype),
            )
            target_q = rewards_batch + discount * bootstrap_mask * (
                torch.minimum(target_q1, target_q2)
                - self.algorithm.log_alpha.exp() * next_log_prob
            )
            data_q1, data_q2 = self.model.critic.evaluate_all_q(
                obs_batch,
                actions_batch,
            )
            td_error = 0.5 * (
                (data_q1 - target_q).abs() + (data_q2 - target_q).abs()
            )

        return {
            "sac_diag/log_prob_mean": log_prob.mean(),
            "sac_diag/entropy_mean": -log_prob.mean(),
            "sac_diag/q1_mean": q1.mean(),
            "sac_diag/q2_mean": q2.mean(),
            "sac_diag/q_abs_max": torch.maximum(q1.abs().max(), q2.abs().max()),
            "sac_diag/critic_disagreement": (q1 - q2).abs().mean(),
            "sac_diag/target_q_mean": target_q.mean(),
            "sac_diag/target_q_std": target_q.std(),
            "sac_diag/td_error_mean": td_error.mean(),
            "sac_diag/td_error_max": td_error.max(),
            "sac_diag/sampled_action_abs_mean": new_actions.abs().mean(),
            "sac_diag/sampled_action_saturation": (
                new_actions.abs() > 0.95
            ).float().mean(),
            "sac_diag/actor_reference_mse": (
                self.model.critic.actor_reference_mse(
                    self.model.actor,
                    obs_batch,
                )
            ),
            "grad/actor_global_norm": self._gradient_norm(
                self.algorithm.actor_parameters
            ),
            "grad/critic_global_norm": self._gradient_norm(
                self.algorithm.critic_parameters
            ),
        }

    def max_num_batches(self) -> int:
        # BaseAgent uses this during construction only to validate DDP parity.
        return 1

    def perform_optimization_step(self, batch_dict, batch_idx):
        raise NotImplementedError("FSQSACAgent uses the released SAC update loop.")

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["sac"] = self.algorithm.save()
        state_dict["sac_update_step"] = self.algorithm.update_step
        state_dict["replay_profile"] = self.replay_profile
        state_dict["replay_contents_saved"] = False
        return state_dict

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        if "sac" in state_dict:
            self.algorithm.load(state_dict["sac"], load_cfg=None, strict=True)
            self.algorithm.update_step = state_dict.get("sac_update_step", 0)
        # The replay buffer is deliberately not restored. This is recorded in the
        # checkpoint and replay_profile.json so a resume never implies otherwise.
        self.algorithm.clear_storage()

    def train(self):
        self.model.train()
        self.algorithm.train_mode()

    def eval(self):
        self.model.eval()
        self.algorithm.eval_mode()

    def _record_environment_metrics(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        done_indices: torch.Tensor,
        extras: dict,
    ) -> None:
        self.current_rewards += rewards
        self.current_lengths += 1
        if done_indices.numel() > 0:
            self.episode_reward_meter.add(
                {"episode_reward": self.current_rewards[done_indices]}
            )
            self.episode_length_meter.add(
                {"episode_length": self.current_lengths[done_indices]}
            )
        alive = 1.0 - dones.float()
        self.current_rewards *= alive
        self.current_lengths *= alive.long()
        extra_stats = {}
        for key, value in extras.items():
            if key.startswith("raw/") or not isinstance(value, torch.Tensor):
                continue
            value = value.float()
            if value.numel() == 1:
                extra_stats[key] = value.flatten()
            else:
                extra_stats[f"{key}_mean"] = value.mean()
                extra_stats[f"{key}_std"] = value.std()
        self.episode_env_tensors.add(extra_stats)

    def _log_epoch(
        self,
        losses: dict,
        rewards: list[torch.Tensor],
        obs_td: TensorDict,
        epoch_counts: dict[str, int],
        updated: bool,
        actor_updated: bool,
    ) -> None:
        end_time = time.time()
        episode_rewards = self.episode_reward_meter.mean_and_clear()
        episode_lengths = self.episode_length_meter.mean_and_clear()
        self.last_episode_reward = episode_rewards.get(
            "episode_reward",
            self.last_episode_reward,
        )
        self.last_episode_length = episode_lengths.get(
            "episode_length",
            self.last_episode_length,
        )
        mean_reward = torch.stack(rewards).mean() if rewards else torch.tensor(0.0)
        log_dict = {
            "info/episode_length": self.last_episode_length,
            "info/episode_reward": self.last_episode_reward,
            "info/frames": float(self.step_count),
            "info/gframes": self.step_count / 1e9,
            "rewards/task_rewards": mean_reward,
            "sac/alpha": float(self.algorithm.alpha),
            "sac/target_entropy": float(self.algorithm.target_entropy),
            "sac/update_step": float(self.algorithm.update_step),
            "replay/transitions": float(
                self.algorithm.replay_buffer.num_transitions * self.num_envs
            ),
            "replay/capacity": float(self.replay_profile["effective_transitions"]),
            "replay/occupancy": float(
                self.algorithm.replay_buffer.num_transitions * self.num_envs
            )
            / self.replay_profile["effective_transitions"],
            "replay/configured_utd": (
                self.config.num_learning_epochs
                * self.config.num_mini_batches
                * self.config.mini_batch_size
            )
            / (self.num_steps * self.num_envs),
            "replay/warmup_transitions": float(
                self.config.replay_warmup_transitions
            ),
            "optimization/enabled": float(updated),
            "optimization/actor_enabled": float(actor_updated),
            "terminations/done_rate": epoch_counts["dones"]
            / (self.num_steps * self.num_envs),
            "terminations/failure_rate": epoch_counts["terminated"]
            / (self.num_steps * self.num_envs),
            "terminations/timeout_rate": epoch_counts["timeouts"]
            / (self.num_steps * self.num_envs),
            "times/fps_last_epoch": (
                self.num_steps * self.get_step_count_increment()
            )
            / max(end_time - self.epoch_start_time, 1e-6),
            "times/training_minutes": (end_time - self.fit_start_time) / 60,
        }
        log_dict.update(
            {
                "losses/critic1_loss": losses.get("critic1", 0.0),
                "losses/critic2_loss": losses.get("critic2", 0.0),
                "losses/actor_loss": losses.get("actor", 0.0),
                "losses/alpha_loss": losses.get("alpha", 0.0),
            }
        )
        log_dict.update(self.model.actor.fsq_diagnostics(obs_td))
        log_dict.update(self.model.actor.policy_diagnostics(obs_td))
        if self.current_epoch % self.config.diagnostic_every == 0:
            log_dict.update(self._sac_diagnostics())
        log_dict.update(
            {f"env/{key}": value for key, value in self.episode_env_tensors.mean_and_clear().items()}
        )
        aggregated = aggregate_scalar_metrics(log_dict, self.fabric, weight=self.num_envs)
        self.fabric.log_dict(aggregated, step=self.current_epoch)

    def _evaluate_and_persist(self, name: str) -> tuple[dict, float | None]:
        eval_log, score, num_items = self.evaluator.evaluate()
        serializable = {
            key: float(value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in eval_log.items()
        }
        serializable.update(
            {
                "score": score,
                "num_evaluated": num_items,
                "fixed_order_motion_ids": (
                    list(self.evaluator.config.evaluation_motion_ids)
                    if getattr(
                        self.evaluator.config,
                        "evaluation_motion_ids",
                        None,
                    )
                    else list(range(num_items))
                ),
                "epoch": self.current_epoch,
                "step_count": self.step_count,
            }
        )
        (self.root_dir / name).write_text(json.dumps(serializable, indent=2) + "\n")
        return eval_log, score

    def fit(self):
        obs, _ = self.env.reset()
        obs_td = self.obs_dict_to_tensordict(self.add_agent_info_to_obs(obs))
        done_indices = torch.arange(self.num_envs, device=self.device)
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)
        self.train()

        while self.current_epoch < self.max_epochs:
            self.epoch_start_time = time.time()
            rewards_for_log: list[torch.Tensor] = []
            epoch_counts = {"dones": 0, "terminated": 0, "timeouts": 0}
            for step in range(self.num_steps):
                obs, _ = self.env.reset(done_indices)
                self.pre_collect_step(step)
                obs_td = self.obs_dict_to_tensordict(self.add_agent_info_to_obs(obs))

                with torch.no_grad():
                    actions = self.algorithm.act(obs_td.clone())
                next_obs, rewards, dones, terminated, extras = self.env.step(actions)
                if not torch.isfinite(rewards).all():
                    raise FloatingPointError("non-finite reward from ProtoMotions environment")

                next_obs_td = self.obs_dict_to_tensordict(
                    self.add_agent_info_to_next_obs(next_obs)
                )
                dones, terminated, extras = self.post_env_step_modifications(
                    dones,
                    terminated,
                    extras,
                )
                done_indices = dones.nonzero(as_tuple=False).flatten()
                timeouts = dones.bool() & ~terminated.bool()
                epoch_counts["dones"] += int(dones.sum().item())
                epoch_counts["terminated"] += int(terminated.sum().item())
                epoch_counts["timeouts"] += int(timeouts.sum().item())
                sac_extras = {
                    "time_outs": timeouts.unsqueeze(-1),
                    "time_outs_obs": next_obs_td.clone(),
                }
                self.algorithm.process_env_step(
                    next_obs_td.clone(),
                    rewards,
                    dones,
                    sac_extras,
                )
                self._record_environment_metrics(rewards, dones, done_indices, extras)
                rewards_for_log.append(rewards.detach().mean())
                self.step_count += self.get_step_count_increment()
                obs_td = next_obs_td

            losses = {}
            replay_transitions = (
                self.algorithm.replay_buffer.num_transitions * self.num_envs
            )
            updated = (
                self.current_epoch >= self.config.start_training_epoch
                and replay_transitions >= self.config.replay_warmup_transitions
            )
            actor_updated = (
                updated
                and self.current_epoch
                >= self.config.actor_start_training_epoch
            )
            if updated:
                original_policy_frequency = self.algorithm.policy_frequency
                original_auto_alpha = self.algorithm.auto_alpha
                if not actor_updated:
                    if self.algorithm.update_step == 0:
                        self.algorithm.update_step = 1
                    self.algorithm.policy_frequency = 1_000_000_000
                    self.algorithm.auto_alpha = False
                try:
                    losses = self.algorithm.update()
                finally:
                    self.algorithm.policy_frequency = original_policy_frequency
                    self.algorithm.auto_alpha = original_auto_alpha
                if actor_updated:
                    self.model.critic.update_actor_reference(
                        self.model.actor,
                        self.config.model.actor_reference_tau,
                    )

            self.current_epoch += 1
            self._log_epoch(
                losses,
                rewards_for_log,
                obs_td,
                epoch_counts,
                updated,
                actor_updated,
            )
            self.fabric.call("after_train", self)

            if (
                self.config.save_epoch_checkpoint_every is not None
                and self.current_epoch % self.config.save_epoch_checkpoint_every == 0
            ):
                self.save(f"epoch_{self.current_epoch}.ckpt")
            if self.current_epoch % self.config.save_last_checkpoint_every == 0:
                self.save("last.ckpt")

            eval_every = self.evaluator.config.eval_metrics_every
            if eval_every is not None and self.current_epoch % eval_every == 0:
                eval_log, score = self._evaluate_and_persist(
                    f"evaluation_epoch_{self.current_epoch}.json"
                )
                if score is not None and (
                    self.best_evaluated_score is None
                    or score >= self.best_evaluated_score
                ):
                    self.best_evaluated_score = score
                    self.save("last.ckpt", new_high_score=True)
                self.fabric.log_dict(eval_log, step=self.current_epoch)
                self.train()

            self.env.on_epoch_end(self.current_epoch)
            if self.should_stop:
                self.save("last.ckpt")
                self._evaluate_and_persist("evaluation_final.json")
                self.fabric.call("on_training_stop", self)
                return

        self.save("last.ckpt")
        self._evaluate_and_persist("evaluation_final.json")
        self.fabric.call("on_fit_end", self)
