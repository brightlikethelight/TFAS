"""Driver script for the probing workstream.

Thin wrapper around :func:`s1s2.probes.cli.main` so you can run it directly::

    python scripts/run_probes.py
    python scripts/run_probes.py targets=[task_type] layers=[0,8,16,24,31]

All configuration flows through ``configs/probe.yaml``.
"""

from __future__ import annotations

from s1s2.probes.cli import main

if __name__ == "__main__":
    main()
