"""Backward-compatible imports for the split vocoder model modules.

Prefer importing generator classes from :mod:`generator` and discriminator
classes/losses from :mod:`discriminators` in new code.
"""

from generator import Generator, ResBlock1, ResBlock2
from discriminators import (
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)

__all__ = [
    "Generator",
    "ResBlock1",
    "ResBlock2",
    "DiscriminatorP",
    "DiscriminatorS",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "discriminator_loss",
    "feature_loss",
    "generator_loss",
]
