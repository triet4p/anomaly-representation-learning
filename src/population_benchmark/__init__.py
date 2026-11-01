"""Sprint 24 Task 7-M20 population-movement benchmark package.

Modules are split so an offline replay can import the frozen spec, durable
storage and pure-numpy evaluation without importing torch or the encoder
package:

- ``spec``: frozen design, namespaces, membership, counts, gates, decision;
- ``evaluation``: the one MMD statistic, calibration, aggregation, gates;
- ``storage``: hash-verified chunk custody;
- ``capture``: real generation + frozen-encoder capture (torch import here);
- ``runner``: design/contract/smoke/official/replay orchestration.
"""

from __future__ import annotations

__all__ = ["capture", "evaluation", "runner", "spec", "storage"]
