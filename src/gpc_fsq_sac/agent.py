"""ProtoMotions lifecycle adapter for the released rsl_rl SAC implementation."""

from __future__ import annotations

import json
import logging
import math
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


def one_sided_conservative_q_loss(
    data_q: torch.Tensor,
    policy_q: torch.Tensor,
    random_q: torch.Tensor,
) -> torch.Tensor:
    """Penalize the most overestimated sampled OOD action family."""
    largest_ood_mean = torch.maximum(policy_q.mean(), random_q.mean())
    return torch.relu(largest_ood_mean - data_q.mean())


class FSQSACAgent(BaseAgent):
    """Run released SAC updates while retaining ProtoMotions' environment lifecycle."""

    config: FSQSACAgentConfig
    require_reward_norm_on_load = False

    def _before_create_model(self) -> None:
        self.config.save_inference_checkpoint = self._as_bool(
            self.config.save_inference_checkpoint
        )
        if self.fabric.world_size != 1:
            raise ValueError("The initial FSQ-SAC integration supports exactly one GPU.")
        obs = self.add_agent_info_to_obs(self.env.get_obs())
        self._setup_obs_td = self.obs_dict_to_tensordict(obs)

    def create_model(self) -> FSQSACModel:
        model = FSQSACModel(
            config=self.config.model,
            obs=self._setup_obs_td,
            action_dim=self.env.robot_config.number_of_actions,
        )
        if self._as_bool(self.config.model.action_bounds_from_motion):
            self._set_motion_action_bounds(model)
        if self.config.model.physical_action_std is not None:
            model.actor.set_physical_action_std(
                self.config.model.physical_action_std
            )
            with torch.no_grad():
                model.critic.actor_reference.log_std.copy_(
                    model.actor.log_std
                )
        return model

    @staticmethod
    def _as_bool(value: bool | str) -> bool:
        return value.lower() == "true" if isinstance(value, str) else value

    def add_agent_info_to_obs(self, obs: dict) -> dict:
        """Record motion identity for SAC replay bookkeeping only.

        The actor and critics' configured observation groups do not consume
        this field. It lets the SAC-only curriculum apply a fixed behavioral
        anchor to already-solved source motions without constraining newly
        introduced motions.
        """
        obs = dict(obs)
        obs["sac_motion_id"] = self.env.motion_manager.motion_ids.detach().clone()[
            :, None
        ]
        if self._as_bool(self.config.model.reference_residual_actions):
            motion_ids = self.env.motion_manager.motion_ids
            motion_times = self.env.motion_manager.motion_times
            time_offset = (
                self.config.model.reference_action_time_offset_steps
                * self.env.motion_manager.env_dt
            )
            target_times = motion_times + time_offset
            motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
            target_times = torch.minimum(target_times, motion_lengths)
            reference_state = self.env.motion_lib.get_motion_state(
                motion_ids,
                target_times,
            )
            action_config = self.env.config.action_config
            offset = action_config["pd_action_offset"].to(
                device=reference_state.dof_pos.device,
                dtype=reference_state.dof_pos.dtype,
            )
            scale = action_config["pd_action_scale"].to(
                device=reference_state.dof_pos.device,
                dtype=reference_state.dof_pos.dtype,
            )
            obs["sac_reference_action"] = (
                reference_state.dof_pos - offset
            ) / scale
        return obs

    @staticmethod
    def target_entropy_for_action_scale(
        canonical_target_entropy: float,
        log_action_range: float,
    ) -> float:
        """Transform a canonical SAC entropy target into environment units."""
        return canonical_target_entropy + log_action_range

    @torch.no_grad()
    def _set_motion_action_bounds(self, model: FSQSACModel) -> None:
        """Calibrate SAC's squashed action space to reachable reference poses.

        ProtoMotions' generic ball-joint action map expands every axis to
        ``[-pi, pi]``. That is safe for PPO, but it makes a unit SAC action span
        a much larger physical displacement than the released SAC recipe
        assumes. Use only the licensed tracker's own reference DOF range, plus
        explicit headroom. No policy checkpoint or expert action is involved.
        """
        action_config = self.env.config.action_config
        offset = action_config["pd_action_offset"].to(
            device=self.device,
            dtype=self.env.motion_lib.dps.dtype,
        )
        scale = action_config["pd_action_scale"].to(
            device=self.device,
            dtype=self.env.motion_lib.dps.dtype,
        )
        dof_pos = self.env.motion_lib.dps
        motion_id = self.config.model.action_bounds_motion_id
        motion_ids = self.config.model.action_bounds_motion_ids
        if motion_id is not None and motion_ids is not None:
            raise ValueError(
                "action_bounds_motion_id and action_bounds_motion_ids "
                "are mutually exclusive"
            )
        selected_motion_ids = (
            [motion_id]
            if motion_id is not None
            else list(motion_ids) if motion_ids is not None else None
        )
        if selected_motion_ids is not None:
            if not selected_motion_ids:
                raise ValueError("action_bounds_motion_ids must not be empty")
            selected_frames = []
            for selected_motion_id in selected_motion_ids:
                start = int(
                    self.env.motion_lib.length_starts[selected_motion_id].item()
                )
                count = int(
                    self.env.motion_lib.motion_num_frames[selected_motion_id].item()
                )
                selected_frames.append(dof_pos[start : start + count])
            dof_pos = torch.cat(selected_frames, dim=0)

        normalized = (dof_pos - offset) / scale
        lower = normalized.amin(dim=0)
        upper = normalized.amax(dim=0)
        margin = self.config.model.action_bounds_margin
        if margin < 0:
            raise ValueError("action_bounds_margin must be non-negative")
        symmetric = self._as_bool(self.config.model.action_bounds_symmetric)
        if symmetric:
            action_range = torch.maximum(lower.abs(), upper.abs()) + margin
            action_bias = torch.zeros_like(action_range)
        else:
            action_bias = 0.5 * (upper + lower)
            action_range = 0.5 * (upper - lower) + margin
        action_range = action_range.clamp(min=1e-3, max=1.0)
        action_bias = action_bias.clamp(
            min=-1.0 + action_range,
            max=1.0 - action_range,
        )
        model.actor.action_bias.copy_(action_bias)
        model.actor.action_range.copy_(action_range)
        model.actor.log_action_range.copy_(torch.log(action_range).sum())
        model.critic.set_action_scaling(action_bias, action_range)
        self.motion_action_bounds = {
            "motion_id": motion_id,
            "motion_ids": selected_motion_ids,
            "margin": margin,
            "symmetric": symmetric,
            "range_min": float(action_range.min().item()),
            "range_mean": float(action_range.mean().item()),
            "range_max": float(action_range.max().item()),
            "bias_abs_mean": float(action_bias.abs().mean().item()),
            "bias_abs_max": float(action_bias.abs().max().item()),
        }
        log.info(
            "Motion-calibrated SAC action bounds: %s",
            json.dumps(self.motion_action_bounds, sort_keys=True),
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
            "motion_action_bounds": getattr(self, "motion_action_bounds", None),
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
            # CLI/dataclass overrides can arrive as the strings "true" and
            # "false". Passing "false" directly is truthy in Python and
            # silently enables temperature learning.
            auto_alpha=self._as_bool(self.config.auto_alpha),
            alpha=self.config.initial_alpha,
            tau=self.config.tau,
            gamma=self.config.gamma,
            target_entropy_scale=self.config.target_entropy_scale,
            device=str(self.device),
            max_grad_norm=self.config.max_grad_norm,
            policy_frequency=self.config.policy_frequency,
            n_steps=self.config.n_steps,
        )
        # ``sample_action_logp`` reports density in the calibrated environment
        # action coordinates. The released SAC target entropy is defined for
        # canonical ``[-1, 1]`` actions. Entropy changes by
        # ``sum(log(action_range))`` under this affine scaling, so apply the
        # same change of variables to the target. Without this correction,
        # automatic alpha tuning chases a coordinate-system artifact whenever
        # motion-calibrated bounds are active.
        self.algorithm.target_entropy = self.target_entropy_for_action_scale(
            self.algorithm.target_entropy,
            float(self.model.actor.log_action_range.item()),
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
            shuffled_target_obs = obs_batch.detach().clone()
            shuffled_target_obs["mimic_target_poses"] = torch.roll(
                shuffled_target_obs["mimic_target_poses"],
                shifts=1,
                dims=0,
            )
            shuffled_target_q1, shuffled_target_q2 = (
                self.model.critic.evaluate_all_q(
                    shuffled_target_obs,
                    actions_batch,
                )
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
            "sac_diag/q_target_sensitivity": 0.5
            * (
                (data_q1 - shuffled_target_q1).abs().mean()
                + (data_q2 - shuffled_target_q2).abs().mean()
            ),
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

    def _conservative_critic_step(self) -> dict[str, float]:
        """Penalize critic values for policy/uniform actions outside replay support."""
        coef = self.config.conservative_q_coef
        if coef <= 0:
            return {}
        replay_buffer = self.algorithm.replay_buffer
        valid_indices = replay_buffer._generate_valid_indices()
        batch = replay_buffer._generate_batch(
            valid_indices,
            self.config.conservative_q_batch_size,
        )
        obs_batch, data_actions = batch[0], batch[1]
        with torch.no_grad():
            policy_actions, _ = self.model.actor.sample_action_logp(obs_batch)
            actor = self.model.actor
            random_actions = actor.action_center(obs_batch) + actor.action_range * (
                2 * torch.rand_like(data_actions) - 1
            )

        data_q1, data_q2 = self.model.critic.evaluate_all_q(
            obs_batch,
            data_actions,
        )
        policy_q1, policy_q2 = self.model.critic.evaluate_all_q(
            obs_batch,
            policy_actions,
        )
        random_q1, random_q2 = self.model.critic.evaluate_all_q(
            obs_batch,
            random_actions,
        )

        # This online variant is a guardrail, not an offline-data objective.
        # Averaging policy and uniform alternatives can hide an exploitable
        # policy gap behind very low random-action Q values.
        cql1 = one_sided_conservative_q_loss(data_q1, policy_q1, random_q1)
        cql2 = one_sided_conservative_q_loss(data_q2, policy_q2, random_q2)
        loss = 0.5 * coef * (cql1 + cql2)
        self.algorithm.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.algorithm.critic_parameters,
            self.config.max_grad_norm,
        )
        self.algorithm.critic_optimizer.step()
        with torch.no_grad():
            self.model.critic.soft_update_target_networks(self.config.tau)
        return {
            "conservative_q_loss": float(loss.item()),
            "conservative_q1_gap": float(
                (policy_q1.detach() - data_q1.detach()).mean().item()
            ),
            "conservative_q2_gap": float(
                (policy_q2.detach() - data_q2.detach()).mean().item()
            ),
            "conservative_random_q1_gap": float(
                (random_q1.detach() - data_q1.detach()).mean().item()
            ),
            "conservative_random_q2_gap": float(
                (random_q2.detach() - data_q2.detach()).mean().item()
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

    def _apply_configured_optimizer_learning_rates(self) -> None:
        """Apply the active stage's LRs after restoring optimizer moments.

        The released SAC loader restores each optimizer's complete state dict,
        including its old parameter-group learning rates. Preserve the useful
        moment estimates, then honor the explicitly configured rates for the
        current SAC-only curriculum stage.
        """
        optimizer_rates = (
            (self.algorithm.actor_optimizer, self.config.actor_learning_rate),
            (self.algorithm.critic_optimizer, self.config.critic_learning_rate),
            (self.algorithm.alpha_optimizer, self.config.alpha_learning_rate),
        )
        for optimizer, learning_rate in optimizer_rates:
            if optimizer is None:
                continue
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        if "sac" in state_dict:
            self.algorithm.load(state_dict["sac"], load_cfg=None, strict=True)
            self._apply_configured_optimizer_learning_rates()
            self.algorithm.update_step = state_dict.get("sac_update_step", 0)
            if self._as_bool(self.config.reset_alpha_on_load):
                initial_alpha = float(self.config.initial_alpha)
                if initial_alpha <= 0:
                    raise ValueError("initial_alpha must be positive")
                with torch.no_grad():
                    self.algorithm.log_alpha.fill_(math.log(initial_alpha))
                self.algorithm.alpha = initial_alpha
                if self.algorithm.alpha_optimizer is not None:
                    self.algorithm.alpha_optimizer.state.clear()
        # The replay buffer is deliberately not restored. This is recorded in the
        # checkpoint and replay_profile.json so a resume never implies otherwise.
        self.algorithm.clear_storage()

    def _after_load_model_state_dict(self, state_dict) -> None:
        super()._after_load_model_state_dict(state_dict)
        # Checkpoints contain action-scaling buffers, but a SAC-only curriculum
        # may intentionally widen from one motion to the complete corpus. The
        # current stage's declared bounds must win over the previous stage's
        # buffers so the actor, online critics, target critics, and behavior
        # reference all use the same coordinates.
        if self._as_bool(self.config.model.action_bounds_from_motion):
            self._set_motion_action_bounds(self.model)
        if self._as_bool(self.config.model.reset_std_on_load):
            with torch.no_grad():
                if self.config.model.physical_action_std is None:
                    log_std = math.log(self.config.model.initial_std)
                    self.model.actor.log_std.fill_(log_std)
                else:
                    self.model.actor.set_physical_action_std(
                        self.config.model.physical_action_std
                    )
                self.model.critic.actor_reference.log_std.copy_(
                    self.model.actor.log_std
                )

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
                "losses/conservative_q_loss": losses.get(
                    "conservative_q_loss",
                    0.0,
                ),
                "sac_diag/conservative_q1_gap": losses.get(
                    "conservative_q1_gap",
                    0.0,
                ),
                "sac_diag/conservative_q2_gap": losses.get(
                    "conservative_q2_gap",
                    0.0,
                ),
                "sac_diag/conservative_random_q1_gap": losses.get(
                    "conservative_random_q1_gap",
                    0.0,
                ),
                "sac_diag/conservative_random_q2_gap": losses.get(
                    "conservative_random_q2_gap",
                    0.0,
                ),
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
        model_selection_score = self.model_selection_score(eval_log, score)
        serializable = {
            key: float(value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in eval_log.items()
        }
        serializable.update(
            {
                "score": score,
                "model_selection_score": model_selection_score,
                "num_evaluated": num_items,
                "fixed_order_motion_ids": (
                    list(self.evaluator._last_evaluated_motion_ids)
                    if getattr(
                        self.evaluator,
                        "_last_evaluated_motion_ids",
                        None,
                    )
                    else (
                        list(self.evaluator.config.evaluation_motion_ids)
                        if getattr(
                            self.evaluator.config,
                            "evaluation_motion_ids",
                            None,
                        )
                        else list(range(num_items))
                    )
                ),
                "epoch": self.current_epoch,
                "step_count": self.step_count,
            }
        )
        (self.root_dir / name).write_text(json.dumps(serializable, indent=2) + "\n")
        if model_selection_score is not None:
            eval_log["eval/model_selection_score"] = model_selection_score
        return eval_log, model_selection_score

    @staticmethod
    def model_selection_score(
        eval_log: dict,
        success_rate: float | None,
    ) -> float | None:
        """Rank strict success first and mean GT error second.

        Strict success is intentionally binary per motion. During early
        scratch training it is commonly tied at zero for many evaluations;
        using it alone repeatedly overwrites the checkpoint with an arbitrary
        tied model. A small negative GT-error tie-breaker preserves the best
        deterministic tracker without allowing error scale to outrank even one
        additional successful motion.
        """
        if success_rate is None:
            return None
        progressive_stage = eval_log.get("eval/progressive_stage", 0.0)
        if isinstance(progressive_stage, torch.Tensor):
            progressive_stage = progressive_stage.item()
        gt_error = eval_log.get("eval/gt_error/mean")
        if gt_error is None:
            return 2.0 * float(progressive_stage) + float(success_rate)
        if isinstance(gt_error, torch.Tensor):
            gt_error = gt_error.item()
        return (
            2.0 * float(progressive_stage)
            + float(success_rate)
            - 1e-3 * float(gt_error)
        )

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
                and self.current_epoch
                >= getattr(
                    self,
                    "_curriculum_actor_resume_epoch",
                    0,
                )
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
                losses.update(self._conservative_critic_step())
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
