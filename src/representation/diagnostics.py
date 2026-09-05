"""Public CPU-only latent geometry diagnostics API."""

from representation.geometry_analysis import (
    analyze_geometry_cache,
    compute_health_metrics,
    compute_neighbors,
    compute_projections,
    compute_separation_metrics,
    load_geometry_cache,
    neighbor_group_rates,
    write_neighbors,
    write_projection_figures,
)

__all__ = [
    "analyze_geometry_cache",
    "compute_health_metrics",
    "compute_neighbors",
    "compute_projections",
    "compute_separation_metrics",
    "load_geometry_cache",
    "neighbor_group_rates",
    "write_neighbors",
    "write_projection_figures",
]
