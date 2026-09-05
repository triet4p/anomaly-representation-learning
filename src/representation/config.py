"""Validated configuration for the V1 representation-learning model."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from synth.config import SUPPORTED_CHANNEL_COUNTS


class V1Config(BaseModel):
    """Model hyperparameters with no fixed semantic file length.

    ``T`` and the resulting patch count are runtime properties of each batch.
    This configuration only describes the computational patch geometry and
    model capacity.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n_channels: int = Field(default=6, description="Supported sensor count.")
    patch_size: int = Field(default=32, description="Timesteps per patch.")
    stride: int = Field(default=16, description="Patch start spacing.")
    d_model: int = Field(default=128, description="Patch latent width.")
    sequence_layers: int = Field(default=4, description="Context encoder depth.")
    attention_heads: int = Field(default=4, description="Context attention heads.")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    n_robots: int = Field(default=5, ge=1, description="Fleet robot count for conditional norm.")
    n_programs: int = Field(default=8, ge=1, description="Fleet program count for conditional norm.")
    use_conditional_norm: bool = Field(default=True, description="Enable conditional sequence batch normalization.")
    min_bucket_samples: int = Field(default=32, ge=1, description="Min normal samples before bucket activates.")


    total_mask_ratio: float = Field(default=0.40, ge=0.0, le=1.0)
    random_fraction: float = Field(default=0.34, ge=0.0, le=1.0)
    info_fraction: float = Field(default=0.33, ge=0.0, le=1.0)
    block_fraction: float = Field(default=0.33, ge=0.0, le=1.0)

    ema_decay: float = Field(default=0.996, gt=0.0, lt=1.0)
    prediction_weight: float = Field(default=1.0, ge=0.0)
    contrastive_weight_max: float = Field(
        default=0.1,
        ge=0.0,
        validation_alias=AliasChoices("contrastive_weight_max", "lambda_max"),
        description="Max contrastive weight (lambda_max).",
    )
    contrastive_warmup_steps: int = Field(default=0, ge=0)
    contrastive_ramp_steps: int = Field(default=1_000, ge=0)
    contrastive_temperature: float = Field(default=0.2, gt=0.0)
    contrastive_gain_std: float = Field(default=0.05, ge=0.0)
    contrastive_offset_std: float = Field(default=0.03, ge=0.0)
    contrastive_noise_std: float = Field(default=0.015, ge=0.0)
    contrastive_max_shift: int = Field(default=4, ge=0)
    knn_k: int = Field(default=5, ge=1)
    seed: int = Field(default=0)
    @property
    def lambda_max(self) -> float:
        """Return the maximum contrastive weight."""
        return self.contrastive_weight_max


    @model_validator(mode="after")
    def validate_model_contract(self) -> "V1Config":
        """Validate cross-field invariants that individual bounds cannot express."""
        if self.n_channels not in SUPPORTED_CHANNEL_COUNTS:
            raise ValueError(
                f"n_channels must be one of {SUPPORTED_CHANNEL_COUNTS}, "
                f"got {self.n_channels}"
            )
        if self.stride > self.patch_size:
            raise ValueError("stride must be less than or equal to patch_size")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        composition = self.random_fraction + self.info_fraction + self.block_fraction
        if abs(composition - 1.0) > 1e-6:
            raise ValueError("masking composition fractions must sum to 1")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable configuration metadata for checkpoints."""
        return self.model_dump(mode="json")
