# opendde-colab

Google Colab pipeline around OpenDDE (an AlphaFold3-style structure
predictor). The structural biology focus is an EGFR-kinase-domain + RAD51
fusion protein that forms a homodimer, plus various EGFR TKIs (erlotinib,
afatinib, osimertinib, alflutinib, lazertinib) and cetuximab bound to EGFR.

Two notebooks:

- `opendde.ipynb` — runs OpenDDE predictions.
- `visualize_opendde_run.ipynb` — the actively-developed one. Upload a run's
  `output.zip`, then a chain of cells: py2Dmol structure+PAE viewer, an
  interactive Ribbon/cartoon viewer with a clickable ipSAE/ipTM matrix
  (custom JS canvas, styled like the AF3 server), ipSAE ranking, ipTM/pTM/
  ipSAE range summary, alignment against reference PDB structures (EGFR
  active 2GS2 / inactive 2GS7, RAD51 1N0W, TKI-bound structures, cetuximab
  5GZ0) with per-model conformation comparison, and a contact-map cell
  analyzing the RAD51-RAD51 vs EGFR-C-terminal contribution to dimerization.

## Working conventions for `visualize_opendde_run.ipynb`

The file is several MB (embedded Colab outputs) — too large for the
Read/NotebookEdit tools. Don't try them directly on this file. Instead:

- Write the new cell as a standalone `.py` file, then insert it via a small
  Python script that loads the notebook as JSON, finds the right cell by
  matching its `#@title` first line, inserts a new cell dict after it, and
  re-dumps with `json.dumps(nb, indent=2, ensure_ascii=False)` /
  `open(path, "w", encoding="utf-8", newline="")` to preserve exact
  formatting. See git history for examples of this pattern (commits
  `e224660`, `38da2f9`, `960a13f`).
- Before every edit: `git fetch` and confirm HEAD matches `origin/main`
  (someone may have pushed from Colab).
- Verify new cell source with `ast.parse` across Python 3.10, 3.11, and 3.12
  — the Colab runtime version isn't fixed.
- Test new cells headlessly against real extracted run data before
  inserting: `exec()` the relevant upstream cell(s) plus the new cell in one
  namespace, with fake `ipywidgets` (Dropdown/Output/HBox stubs) and fake
  `IPython.display` (display/HTML stubs) modules injected into
  `sys.modules`, using a real extracted `.cif`/`.json` from a previously
  downloaded run's `output.zip`. Watch out: if the test harness's
  `RUNS_DIR` collides with the `reference_pdbs/` cache directory the
  alignment cell creates, glob picks up cached reference PDBs as if they
  were real predicted models — use a `RUNS_DIR` that's a sibling of, not a
  parent of, `reference_pdbs/` (matches production, where
  `RUNS_DIR="./runs"`).
- After inserting: `git diff --stat` to confirm the change is purely
  additive, re-verify syntax by loading the notebook JSON and re-running
  `ast.parse` on the target cell, then commit with a detailed rationale
  (why, not just what) and `git fetch` + `git push`.
- Reuse established helpers rather than reimplementing: `find_all_cifs`,
  `find_summary_json`, `find_full_data_json`, `model_label` (shared helpers
  cell); `compute_alignments(cif_path)` (returns per-model EGFR/RAD51
  domain matches with per-residue resnum ranges — use this for domain
  boundaries instead of hardcoding residue numbers); `GROUP_COLOR` for
  consistent EGFR (`#0053D6`) / RAD51 (`#FF7D45`) coloring.
- No new pip dependencies without checking they're free. numpy/pandas are
  already relied on transitively; matplotlib is Colab-preinstalled and used
  with `matplotlib.use("Agg")`-safe code (avoid assuming a live IPython
  display hook).

## Key domain/debugging facts

- OpenDDE's `runner/inference.py` `update_inference_configs()` forces fp32
  (silently overriding `--dtype bf16`) for the confidence_head and
  sample_diffusion phases whenever total tokens <= 2560, which can cause
  surprise CUDA OOMs on large jobs even with bf16 + sample=1 requested.
  Override with env vars `OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1` and
  `OPENDDE_FORCE_CONFIDENCE_AMP=1`. Also
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set via `os.environ`
  before CUDA/the opendde subprocess starts) helps with
  fragmentation-driven OOMs.
- The fusion construct's EGFR kinase domain spans roughly residues 1-279
  and RAD51 spans roughly 300-541 within each fusion chain, with an
  unaligned ~20-residue linker between them — don't hardcode this, read it
  per-model from `compute_alignments()` since it can shift.
- Contact-map finding (seed 103 fusion-dimer prediction): the dimer
  interface involves RAD51-RAD51 self-contact, EGFR C-terminal contact,
  AND the linker region — all three, not just one mechanism.
