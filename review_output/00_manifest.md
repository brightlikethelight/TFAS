# Pre-Deadline Review Manifest

Generated: 2026-05-03

## Project P1: Readable but Not Writable

| Field | Value |
|---|---|
| Repo | `/Users/brightliu/School_Work/TFAS/s1s2/` |
| Title | Readable but Not Writable: Reasoning Training Decouples Bias Directions from Behavior in LLMs |
| Last commit | `d0abe63` Round 22 (2026-05-03) |
| Result JSONs | 74 files |
| Figure PDFs | 15 files |
| Bib entries | 41 active keys |

### Papers

| Paper | File | Pages | Body pp | Abstract | Style | Author | Figs | Tables | Cites | Type 3 Fonts |
|---|---|---|---|---|---|---|---|---|---|---|
| NeurIPS anonymous | `neurips_paper_anonymous.tex` | 22 | 9 (Discussion p.9) | 238w | `neurips_2026` (anon) | Anonymous Authors | 3 | 6 | 34 unique keys | 0 |
| NeurIPS preprint | `neurips_paper.tex` | 23 | N/A (preprint) | 238w | `neurips_2026` [preprint] | Anonymous Authors | 3 | 7+ | 34 | 0 |
| ICML Workshop | `workshop_paper_icml.tex` | 9 | ~7 (Discussion p.6) | 174w | `article` (not icml2026) | Anonymous | 1 | 2 | 31 | 0 |

### Target Venues

| Venue | Deadline | Page limit | Style | Status |
|---|---|---|---|---|
| NeurIPS 2026 Main | Abstract May 4 AOE, Paper May 6 AOE | 9pp body + refs + appendix | `neurips_2026.sty` | **READY** — PDF compiled, zero errors, checklist included |
| ICML MechInterp Workshop | May 8 AOE | 9pp (NeurIPS fmt) / 8pp (ICML fmt) | NeurIPS OK for submission, ICML for camera-ready | **READY** — uses `article` class (acceptable per CFP), real figures |
| EMNLP 2026 ARR May | May 25 | 8pp (ACL format) | ACL style | NOT PREPARED — dual-sub with NeurIPS forbidden |
| TMLR | Rolling | No limit | TMLR style | NOT PREPARED — dual-sub with NeurIPS forbidden |

### Blockers

| Issue | Severity | Notes |
|---|---|---|
| Compiled PDFs present | OK | All 3 compiled, zero errors, zero undefined refs |
| `.bib` present | OK | `references.bib`, 41 entries |
| Style files | OK | `neurips_2026.sty` present; ICML workshop uses NeurIPS format (allowed) |
| `humanizer-academic` skill | NOT INSTALLED | No matching deferred tool found. Manual AI-writing pass was done in Rounds 18-19. |
| Disk space | WARNING | 6.9 GB free — limits subagent count to ~4 concurrent |
| Co-author list | HUMAN ACTION | Still "pending advisor confirmation" per metadata |

### Subagent Plan

Given disk constraints, I will run **4 subagents** (not 6) in parallel, combining some roles:

| Agent | Covers | Files needed |
|---|---|---|
| A: Format + Compliance | Subagent A scope for both venues | Both .tex files, style file, checklist.tex |
| B: Writing + Humanizer | Subagent B scope — prose quality, AI-tells, clarity | Both .tex files |
| D: Citations + Experiments | Subagents D+E combined — citations, claims, gaps | Both .tex files, references.bib, all result JSONs |
| F: Red Team Reviews | Subagent F — simulated reviews for both venues | Both compiled PDFs |
| (Figures audit done in Rounds 14-20 already — skip C, report carried forward) |
