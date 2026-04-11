"""Attention entropy analysis pipeline.

Reads precomputed attention metrics from HDF5, performs per-head Mann-Whitney
U tests with BH-FDR correction, classifies heads as S1-/S2-/mixed/non-
specialized following Fartale et al. (2025), handles GQA non-independence
and Gemma-2 sliding window layers.
"""
