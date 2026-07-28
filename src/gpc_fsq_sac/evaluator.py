"""Newton-safe fixed-order evaluation adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator


@dataclass
class NewtonMimicEvaluatorConfig(MimicEvaluatorConfig):
    """Allow quick probes to evaluate only their training motions."""

    evaluation_motion_ids: list[int] = field(default_factory=list)
    continuous_error_curriculum: bool = False
    continuous_error_curriculum_start_epoch: int = 0
    continuous_error_curriculum_min_success_rate: float = 0.0
    continuous_error_curriculum_alpha: float = 0.25
    continuous_error_curriculum_min_relative_weight: float = 0.5
    continuous_error_curriculum_max_relative_weight: float = 2.0
    progressive_seed_motion_ids: list[int] = field(default_factory=list)
    progressive_expand_success_rate: float = 1.0
    progressive_expand_consecutive_evals: int = 1
    progressive_actor_pause_epochs: int = 0


class NewtonMimicEvaluator(MimicEvaluator):
    """Keep inactive Newton worlds valid instead of parking them below the terrain.

    ProtoMotions' generic evaluator parks unused environments to reduce PhysX
    broadphase pressure. Newton advances every vectorized world, and parking can
    make those unused worlds non-finite. Leaving them at their valid pre-evaluation
    states is safe; metrics still use each of the 61 motion IDs exactly once.
    """

    def __init__(self, agent, fabric, config):
        super().__init__(agent, fabric, config)
        self._progressive_expanded = False
        self._progressive_success_streak = 0
        self._progressive_expansion_epoch = None
        self._progressive_actor_resume_epoch = None
        self._last_evaluated_motion_ids: list[int] = []
        self._last_evaluation_progressive_stage = 0
        self.agent._curriculum_actor_resume_epoch = 0
        self._initialize_progressive_curriculum()

    def _progressive_seed_motion_ids(self) -> list[int]:
        config = getattr(self, "config", None)
        return list(
            getattr(
                config,
                "progressive_seed_motion_ids",
                [],
            )
        )

    def _initialize_progressive_curriculum(self) -> None:
        """Restrict initial sampling without persistent exclusion masks.

        MotionManager exclusions are deliberately persistent: excluded IDs are
        forced back to zero after every weight update. A progressive curriculum
        therefore has to own the sampling weights directly so later stages can
        make all motions sampleable in the same uninterrupted SAC process.
        """
        seed_ids = self._progressive_seed_motion_ids()
        if not seed_ids:
            return
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError(
                "progressive_seed_motion_ids must contain unique IDs"
            )
        num_motions = self.motion_lib.num_motions()
        if any(motion_id < 0 or motion_id >= num_motions for motion_id in seed_ids):
            raise ValueError(
                "progressive_seed_motion_ids contains an out-of-range ID"
            )
        if getattr(self.motion_manager, "excluded_motion_ids", None) is not None:
            raise ValueError(
                "progressive_seed_motion_ids cannot be combined with persistent "
                "motion exclusions"
            )
        if getattr(self.config, "progressive_expand_consecutive_evals", 1) < 1:
            raise ValueError(
                "progressive_expand_consecutive_evals must be at least 1"
            )
        if getattr(self.config, "progressive_actor_pause_epochs", 0) < 0:
            raise ValueError(
                "progressive_actor_pause_epochs cannot be negative"
            )
        success_rate = getattr(
            self.config,
            "progressive_expand_success_rate",
            1.0,
        )
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError(
                "progressive_expand_success_rate must be in [0, 1]"
            )

        weights = torch.zeros_like(self.motion_manager.motion_weights)
        weights[seed_ids] = 1.0
        self.motion_manager.update_sampling_weights(weights)
        self.config.evaluation_motion_ids = seed_ids

    def _maybe_expand_progressive_curriculum(self) -> bool:
        """Unlock the full corpus after the scratch policy solves its seed set."""
        seed_ids = self._progressive_seed_motion_ids()
        if not seed_ids or self._progressive_expanded:
            return False

        evaluated_ids = torch.nonzero(
            self._eval_mask,
            as_tuple=False,
        ).flatten()
        if evaluated_ids.tolist() != sorted(seed_ids):
            raise RuntimeError(
                "progressive seed evaluation did not cover the configured "
                "motion IDs exactly"
            )
        success_rate = float(
            (~self._motion_failed[evaluated_ids]).float().mean().item()
        )
        required_rate = self.config.progressive_expand_success_rate
        if success_rate >= required_rate:
            self._progressive_success_streak += 1
        else:
            self._progressive_success_streak = 0
        if (
            self._progressive_success_streak
            < self.config.progressive_expand_consecutive_evals
        ):
            return False

        self._progressive_expanded = True
        self._progressive_expansion_epoch = self.agent.current_epoch
        self._progressive_actor_resume_epoch = (
            self.agent.current_epoch
            + self.config.progressive_actor_pause_epochs
        )
        self.agent._curriculum_actor_resume_epoch = (
            self._progressive_actor_resume_epoch
        )
        self.config.evaluation_motion_ids = []
        self.motion_manager.update_sampling_weights(
            torch.ones_like(self.motion_manager.motion_weights)
        )
        return True

    def _park_inactive_envs(self, active_env_ids):
        del active_env_ids

    def _inactive_env_ids(self, active_env_ids: torch.Tensor) -> torch.Tensor:
        active_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        active_mask[active_env_ids] = True
        return torch.nonzero(~active_mask, as_tuple=False).flatten()

    def _stabilize_inactive_actions(
        self,
        actions: torch.Tensor,
        obs,
        active_env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Keep metric-inactive Newton worlds on their reference trajectories.

        ProtoMotions steps every vectorized Newton world even when a focused
        evaluation only scores a few motion IDs. Letting a random scratch
        policy control the other hundreds of worlds can make an unscored world
        numerically non-finite and abort the scored evaluation. SAC observations
        already contain the task's next reference DOF target; applying it only
        to inactive worlds isolates evaluator bookkeeping without changing any
        scored action, training transition, reward, or replay item.
        """
        inactive_env_ids = self._inactive_env_ids(active_env_ids)
        if (
            inactive_env_ids.numel() == 0
            or "sac_reference_action" not in obs.keys()
        ):
            return actions
        stabilized = actions.clone()
        stabilized[inactive_env_ids] = obs["sac_reference_action"][
            inactive_env_ids
        ]
        return stabilized

    def evaluate_episode(
        self,
        env_ids: torch.Tensor,
        max_steps: int,
    ) -> None:
        """Evaluate active clips while keeping all Newton worlds finite."""
        ema_alpha = self.config.eval_action_ema_alpha
        self._on_episode_start(env_ids)

        inactive_env_ids = self._inactive_env_ids(env_ids)
        if inactive_env_ids.numel() > 0:
            self.env.reset(
                inactive_env_ids,
                sample_flat=True,
                disable_motion_resample=True,
            )
        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        self.agent.pre_collect_step(0)
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None
        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get(
                "mean_action",
                model_outs.get("action"),
            )
            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = (
                    ema_alpha * actions
                    + (1.0 - ema_alpha) * prev_actions
                )
                prev_actions = actions.clone()
            actions = self._stabilize_inactive_actions(
                actions,
                obs_td,
                env_ids,
            )

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            self.agent.pre_collect_step(step_idx + 1)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def _build_eval_batches(self):
        motion_ids = self.config.evaluation_motion_ids
        if not motion_ids:
            return super()._build_eval_batches()
        if len(motion_ids) > self.num_envs:
            raise ValueError(
                "evaluation_motion_ids cannot contain more entries than num_envs"
            )
        motion_ids_tensor = torch.tensor(
            motion_ids,
            dtype=torch.long,
            device=self.device,
        )
        if (motion_ids_tensor < 0).any() or (
            motion_ids_tensor >= self.motion_lib.num_motions()
        ).any():
            raise ValueError("evaluation_motion_ids contains an out-of-range ID")
        # Match the environment assignment used by the full fixed-order evaluation
        # whenever possible. Newton environments are independent, but their reset
        # streams are keyed by environment ID, so moving motion 43 to environment 0
        # is not an exact focused replay of the all-motion evaluation.
        if int(motion_ids_tensor.max().item()) < self.num_envs:
            env_ids = motion_ids_tensor.clone()
        else:
            env_ids = torch.arange(len(motion_ids), device=self.device)
        return [(env_ids, motion_ids_tensor)]

    def _update_motion_sampling_weights(self) -> None:
        """Optionally prioritize clips by continuous mean tracking error.

        ProtoMotions' native curriculum is intentionally based on a binary
        per-motion success threshold. During scratch SAC training, a clip whose
        mean error has fallen substantially can still be classified as failed
        because of one bad frame. If every clip is still binary-failed, the
        native rule scales every weight equally and cannot focus replay on the
        remaining hard motions.

        This alternative keeps the native motion manager and evaluator
        lifecycle, but smoothly moves evaluated weights toward their relative
        mean ground-tracking error. It consumes only the current SAC policy's
        own evaluation metrics; it does not use demonstrations, another policy,
        or checkpoint-derived data.
        """
        # Avoid reporting the previous evaluation's continuous weights when
        # this evaluation falls back to the native binary curriculum.
        self._continuous_sampling_weights = {}
        if self._maybe_expand_progressive_curriculum():
            return
        if not self.config.continuous_error_curriculum:
            return super()._update_motion_sampling_weights()
        if (
            self.agent.current_epoch
            < self.config.continuous_error_curriculum_start_epoch
        ):
            return super()._update_motion_sampling_weights()
        min_success_rate = (
            self.config.continuous_error_curriculum_min_success_rate
        )
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError(
                "continuous_error_curriculum_min_success_rate must be "
                "in [0, 1]"
            )
        evaluated_count = self._eval_mask.sum().clamp(min=1)
        success_rate = (
            (~self._motion_failed & self._eval_mask).sum()
            / evaluated_count
        )
        if success_rate < min_success_rate:
            return super()._update_motion_sampling_weights()
        if self._motion_failed is None or self._eval_mask is None:
            return
        if "gt_error" not in self._component_value_sum:
            raise KeyError(
                "continuous_error_curriculum requires the gt_error "
                "evaluation component"
            )

        alpha = self.config.continuous_error_curriculum_alpha
        min_relative = (
            self.config.continuous_error_curriculum_min_relative_weight
        )
        max_relative = (
            self.config.continuous_error_curriculum_max_relative_weight
        )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "continuous_error_curriculum_alpha must be in [0, 1]"
            )
        if min_relative <= 0 or max_relative < min_relative:
            raise ValueError(
                "continuous curriculum relative weights must satisfy "
                "0 < min <= max"
            )

        evaluated_ids = torch.nonzero(
            self._eval_mask,
            as_tuple=False,
        ).flatten()
        step_counts = self._component_step_count["gt_error"][evaluated_ids]
        valid = step_counts > 0
        evaluated_ids = evaluated_ids[valid]
        if evaluated_ids.numel() == 0:
            return
        step_counts = step_counts[valid]
        mean_errors = (
            self._component_value_sum["gt_error"][evaluated_ids]
            / step_counts
        )
        relative = mean_errors / mean_errors.mean().clamp(min=1e-6)
        relative = relative.sqrt().clamp(
            min=min_relative,
            max=max_relative,
        )
        relative /= relative.mean().clamp(min=1e-6)

        failed_motions = torch.nonzero(
            self._motion_failed,
            as_tuple=False,
        ).flatten().tolist()
        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        new_weights = self.env.motion_manager.motion_weights.clone()
        current_mean = new_weights[evaluated_ids].mean()
        target_weights = current_mean * relative
        new_weights[evaluated_ids] = torch.lerp(
            new_weights[evaluated_ids],
            target_weights,
            alpha,
        )
        self.env.motion_manager.update_sampling_weights(new_weights)
        self._continuous_sampling_weights = {
            int(motion_id.item()): float(weight.item())
            for motion_id, weight in zip(
                evaluated_ids,
                new_weights[evaluated_ids],
                strict=True,
            )
        }

    def process_eval_results(self):
        """Retain per-motion tracking metrics alongside aggregate metrics.

        ProtoMotions intentionally aggregates evaluator buffers before returning
        them. Aggregate scores are appropriate for model selection, but they hide
        whether a small-corpus SAC experiment is improving every clip or merely
        averaging one easy and one catastrophic motion. Read the native buffers
        before evaluator cleanup and expose the same means/maxima per motion.
        """
        self._last_evaluated_motion_ids = torch.nonzero(
            self._eval_mask,
            as_tuple=False,
        ).flatten().tolist()
        self._last_evaluation_progressive_stage = int(
            getattr(self, "_progressive_expanded", False)
        )
        to_log, success_rate, num_eval_items = super().process_eval_results()
        if self._motion_failed is None or self._eval_mask is None:
            return to_log, success_rate, num_eval_items

        if self._progressive_seed_motion_ids():
            to_log["eval/progressive_stage"] = float(
                self._last_evaluation_progressive_stage
            )
            to_log["eval/progressive_expanded"] = float(
                self._progressive_expanded
            )
            to_log["eval/progressive_success_streak"] = float(
                self._progressive_success_streak
            )
            if self._progressive_expansion_epoch is not None:
                to_log["eval/progressive_expansion_epoch"] = float(
                    self._progressive_expansion_epoch
                )
            if self._progressive_actor_resume_epoch is not None:
                to_log["eval/progressive_actor_resume_epoch"] = float(
                    self._progressive_actor_resume_epoch
                )

        evaluated_ids = torch.nonzero(self._eval_mask, as_tuple=False).flatten()
        for motion_id_tensor in evaluated_ids:
            motion_id = int(motion_id_tensor.item())
            prefix = f"eval/motion_{motion_id}"
            to_log[f"{prefix}/success"] = float(
                not self._motion_failed[motion_id].item()
            )
            for name in self._component_value_sum:
                step_count = int(self._component_step_count[name][motion_id].item())
                if step_count == 0:
                    continue
                to_log[f"{prefix}/{name}_mean"] = float(
                    (
                        self._component_value_sum[name][motion_id] / step_count
                    ).item()
                )
                to_log[f"{prefix}/{name}_max"] = float(
                    self._component_value_max[name][motion_id].item()
                )
            sampling_weights = getattr(
                self,
                "_continuous_sampling_weights",
                {},
            )
            if motion_id in sampling_weights:
                to_log[f"{prefix}/sampling_weight"] = sampling_weights[
                    motion_id
                ]

        return to_log, success_rate, num_eval_items

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict.update(
            {
                "progressive_expanded": self._progressive_expanded,
                "progressive_success_streak": self._progressive_success_streak,
                "progressive_expansion_epoch": self._progressive_expansion_epoch,
                "progressive_actor_resume_epoch": (
                    self._progressive_actor_resume_epoch
                ),
            }
        )
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._progressive_expanded = state_dict.get(
            "progressive_expanded",
            False,
        )
        self._progressive_success_streak = state_dict.get(
            "progressive_success_streak",
            0,
        )
        self._progressive_expansion_epoch = state_dict.get(
            "progressive_expansion_epoch",
        )
        self._progressive_actor_resume_epoch = state_dict.get(
            "progressive_actor_resume_epoch",
        )
        self.agent._curriculum_actor_resume_epoch = (
            self._progressive_actor_resume_epoch or 0
        )
        seed_ids = self._progressive_seed_motion_ids()
        if not seed_ids:
            return
        if self._progressive_expanded:
            self.config.evaluation_motion_ids = []
            weights = torch.ones_like(self.motion_manager.motion_weights)
        else:
            self.config.evaluation_motion_ids = seed_ids
            weights = torch.zeros_like(self.motion_manager.motion_weights)
            weights[seed_ids] = 1.0
        self.motion_manager.update_sampling_weights(weights)
