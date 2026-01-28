## TIA-Framework for Cross-Paradigm Logic Translation

This repository contains the implementation of the Tiered Isomorphic Alignment (TIA) framework for auto-formalizing imperative code (Python) into formal logic (Isabelle/HOL). This work introduces a hierarchical decomposition strategy to bridge the semantic gap in cross-paradigm logic translation.


## Project Overview
Auto-formalization often fails due to the structural and semantic mismatch between imperative languages and functional formal logics. The TIA Framework addresses this by decomposing computational specifications into four distinct complexity tiers:

- Tier 1: Lexical Grounding (Constants, word sizes, masks)
- Tier 2: Functional Units (Core operations, round functions, S-Boxes)
- Tier 3: Structural Logic (Iteration, recursion, key schedules)
- Tier 4: Top-level Orchestration (End-to-end encryption/decryption)

Requirements
- requirements.txt
- Python 3.9+
- Isabelle 2025


