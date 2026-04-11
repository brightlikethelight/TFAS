"""Interactive analysis dashboard for exploring s1s2 results.

Provides a Gradio-based UI with tabs for behavioral outcomes, probing
results, SAE features, attention entropy, representational geometry,
and hypothesis evaluation. Runs on real results or synthetic demo data.

Usage::

    python scripts/run_dashboard.py                # real results
    python scripts/run_dashboard.py --synthetic    # demo mode with synthetic data
"""

from s1s2.dashboard.app import create_app

__all__ = ["create_app"]
