"""Newton-safe fixed-order evaluation adapter."""

from __future__ import annotations

from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator


class NewtonMimicEvaluator(MimicEvaluator):
    """Keep inactive Newton worlds valid instead of parking them below the terrain.

    ProtoMotions' generic evaluator parks unused environments to reduce PhysX
    broadphase pressure. Newton advances every vectorized world, and parking can
    make those unused worlds non-finite. Leaving them at their valid pre-evaluation
    states is safe; metrics still use each of the 61 motion IDs exactly once.
    """

    def _park_inactive_envs(self, active_env_ids):
        del active_env_ids

