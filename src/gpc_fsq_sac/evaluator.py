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


class NewtonMimicEvaluator(MimicEvaluator):
    """Keep inactive Newton worlds valid instead of parking them below the terrain.

    ProtoMotions' generic evaluator parks unused environments to reduce PhysX
    broadphase pressure. Newton advances every vectorized world, and parking can
    make those unused worlds non-finite. Leaving them at their valid pre-evaluation
    states is safe; metrics still use each of the 61 motion IDs exactly once.
    """

    def _park_inactive_envs(self, active_env_ids):
        del active_env_ids

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
