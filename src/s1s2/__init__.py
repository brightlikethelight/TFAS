"""s1s2: Mechanistic Signatures of System 1 vs System 2 Processing in LLMs.

A multi-method mechanistic interpretability study testing whether LLMs exhibit
internally distinct processing modes analogous to System 1 (heuristic) and
System 2 (deliberative) cognition, and whether reasoning training amplifies
this distinction.

Five analysis workstreams operate on a shared activation cache:

- :mod:`s1s2.probes` -- linear probing
- :mod:`s1s2.sae` -- SAE feature analysis
- :mod:`s1s2.attention` -- attention entropy
- :mod:`s1s2.geometry` -- representational geometry
- :mod:`s1s2.causal` -- causal interventions

Plus:

- :mod:`s1s2.benchmark` -- problem set
- :mod:`s1s2.extract` -- activation extraction
- :mod:`s1s2.metacog` -- metacognitive monitoring stretch goal
- :mod:`s1s2.utils` -- shared utilities (IO, seeding, stats)
- :mod:`s1s2.viz` -- plotting helpers
"""

__version__ = "0.1.0"
