"""Validated V2 configuration for hierarchical latent-geometry monitoring.

Sprint 11 Task 9. Task 13+ (aggregation detail), Task 14 (tracker state),
Task 15 (risk layer), and Task 16 (trainer integration) build on these
contracts but MUST NOT live here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth.config import SUPPORTED_CHANNEL_COUNTS
from synth.schema import DIAGNOSTIC_ONLY_FIELD_NAMES

#: The only permitted reference fallback chain, finest to coarsest.
#: There is deliberately no program-only entry: the same program on a
#: different robot is never a positive (methodology section 8.1).
FALLBACK_ORDER: tuple[str, ...] = (
    "robot_program_regime",
    "robot_program",
    "robot",
    "fleet_low_confidence",
)

#: Encoder-visible conditioning vocabulary for the V2 patch path. Regime ids
#: are per-patch operating context; robot/program are per-file context.
V2_ENCODER_CONTEXT_FIELDS: frozenset[str] = frozenset(
    {"robot_idx", "program_idx", "regime_ids", "operating_context"}
)

#: Fields that MUST NEVER enter encoder inputs. Masks shape the loss only.
V2_ENCODER_FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    DIAGNOSTIC_ONLY_FIELD_NAMES
    | {"file_labels", "file_label", "label", "severity", "mask"}
)


def assert_encoder_inputs_clean(inputs: object) -> None:
    """Raise if a would-be encoder input mapping carries supervision."""
    names = set(inputs) if isinstance(inputs, dict) else set(vars(inputs))
    leaked = names & set(V2_ENCODER_FORBIDDEN_FIELDS)
    if leaked:
        raise ValueError(f"encoder inputs leak supervision: {sorted(leaked)}")


class V2Config(BaseModel):
    """Hyperparameters for pair/regime patch energy and longitudinal outputs.

    ``T`` and the patch count stay runtime properties of each batch; this
    model only fixes computational geometry, fallback/regularization, file
    boundary summaries, trajectory fields, and risk horizons.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n_channels: int = Field(default=6, description="Supported sensor count.")
    patch_size: int = Field(default=32, description="Timesteps per patch.")
    stride: int = Field(default=16, description="Patch start spacing.")
    d_model: int = Field(default=128, description="Patch latent width.")
    sequence_layers: int = Field(default=4, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    n_robots: int = Field(default=5, ge=1)
    n_programs: int = Field(default=8, ge=1)
    n_regimes: int = Field(default=7, ge=1)
    n_prototypes: int = Field(default=2, ge=1, description="Mixture components per (robot, program).")

    shrinkage: float = Field(default=0.2, ge=0.0, le=1.0)
    covariance_eps: float = Field(default=1e-4, gt=0.0)
    min_group_samples: int = Field(default=8, ge=2)
    diag_min_samples: int = Field(default=32, ge=2)

    top_q_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    elevated_threshold: float = Field(default=3.0, gt=0.0)

    boundary_margin: float = Field(default=1.0, ge=0.0)
    boundary_alpha_max: float = Field(default=1.0, ge=0.0)
    boundary_warmup_steps: int = Field(default=500, ge=0)
    boundary_ramp_steps: int = Field(default=2_000, ge=0)
    background_weight: float = Field(default=1.0, ge=0.0)
    variance_weight: float = Field(default=1.0, ge=0.0)
    covariance_weight: float = Field(default=1.0, ge=0.0)

    calibration_min_samples: int = Field(default=32, ge=1)
    risk_horizons_days: tuple[int, int] = Field(default=(1, 7))
    allow_program_only_fallback: bool = Field(default=False)

    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_v2_contract(self) -> "V2Config":
        """Validate cross-field invariants individual bounds cannot express."""
        if self.n_channels not in SUPPORTED_CHANNEL_COUNTS:
            raise ValueError(
                f"n_channels must be one of {SUPPORTED_CHANNEL_COUNTS}, "
                f"got {self.n_channels}"
            )
        if self.stride > self.patch_size:
            raise ValueError("stride must be less than or equal to patch_size")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.diag_min_samples < self.min_group_samples:
            raise ValueError("diag_min_samples must cover min_group_samples")
        if sorted(self.risk_horizons_days) != [1, 7]:
            raise ValueError("risk_horizons_days must be exactly (1, 7)")
        if self.allow_program_only_fallback:
            raise ValueError("program-only cross-robot fallback is forbidden")
        return self

    def fallback_order(self) -> tuple[str, ...]:
        """Return the fixed hierarchical fallback chain."""
        return FALLBACK_ORDER

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable configuration metadata for checkpoints."""
        return self.model_dump(mode="json")
