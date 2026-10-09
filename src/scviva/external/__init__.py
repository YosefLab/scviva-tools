"""External models for scviva-tools.

These models are ported from scvi-tools and adapted to use scviva base classes
and mixins.
"""

from __future__ import annotations

from scviva.external.amici import AMICI
from scviva.external.diagvi import DIAGVI
from scviva.external.starfysh import Starfysh
from scviva.external.stereoscope._model import RNAStereoscope, SpatialStereoscope
from scviva.external.tangram import Tangram

__all__ = ["AMICI", "RNAStereoscope", "SpatialStereoscope", "Starfysh", "Tangram", "DIAGVI"]
