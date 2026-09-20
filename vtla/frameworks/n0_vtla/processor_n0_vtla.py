from dataclasses import dataclass

import torch.nn.functional as F

from vtla.engine.processor import ProcessorStepRegistry
from vtla.engine.types import TransitionKey
from vtla.engine.utils.constants import OBS_STATE
from vtla.frameworks.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)


@ProcessorStepRegistry.register(name="n0_vtla_prepare_state")
@dataclass
class N0VTLAPrepareStateStep(Pi05PrepareStateTokenizerProcessorStep):
    """Match native PI0.5's padded state prompt; retain StarVTLA normalization."""

    def __call__(self, transition):
        if self.state_mode != "none":
            transition = dict(transition)
            observation = dict(transition[TransitionKey.OBSERVATION])
            state = observation[OBS_STATE]
            if state.ndim != 2 or state.shape[-1] > self.max_state_dim:
                raise ValueError("N0-VTLA state must be [B,D] with D <= max_state_dim.")
            observation[OBS_STATE] = F.pad(state, (0, self.max_state_dim - state.shape[-1]))
            transition[TransitionKey.OBSERVATION] = observation
        return super().__call__(transition)


def make_n0_vtla_pre_post_processors(config, dataset_stats=None):
    pre, post = make_pi05_pre_post_processors(config, dataset_stats)
    for index, step in enumerate(pre.steps):
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep):
            pre.steps[index] = N0VTLAPrepareStateStep(max_state_dim=config.max_state_dim, state_mode=config.state_mode)
    return pre, post
