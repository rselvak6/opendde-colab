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

- `scripts/download_opendde_data.sh` must be run with `--skip-search-database`
  on Colab. `search_database/` is only the template + RNA-MSA search data, and
  the pred command uses `--use_template false --use_rna_msa false` with protein
  MSAs coming from the remote MSA service (api.colabfold.com), so nothing reads
  it — but it decompresses to ~89 GB (nt_rna ~75 GB, rnacentral ~13 GB) and
  blows the Colab disk. On *any* search-database failure (disk, stalled
  multi-GB transfer, missing `zstd` CLI) the script falls back to
  `$DATA_BASE_URL/search_database.tar.gz` and reports
  `Failed to download/extract .../search_database.tar.gz`; that tarball returns
  HTTP 403, so the 403 is a masked symptom, never the real cause.

- Templates: OpenDDE takes **no structure file** as input. There is no AF3-style
  `mmcifPath` / `templates` array — the only hook is `templatesPath` on a
  `proteinChain`, pointing at an hmmsearch `.a3m` or HHsearch `.hhr` *hit* file.
  The structure is then resolved by the 4-char PDB ID in the hit name from
  `$OPENDDE_ROOT_DIR/search_database/mmcif/<id>.cif`, else downloaded from PDBe
  (`data.template.fetch_remote` defaults true). Template *inference* therefore
  does not need the skipped search database — only `kalign` (3.x CLI:
  `-i/-o/-format fasta`) and the `common/` files. Four gates apply, and three
  fail **silently** (the run just logs `Found 0 templates`):
  1. hit name must be `[a-zA-Z\d]{4}_<chain>` (`get_pdb_id_and_chain`);
  2. the ID must exist in `common/release_date_cache.json`, else
     "`<id> date unknown`" in `TemplateHitFilter.prefilter`;
  3. the CIF's own `min(_pdbx_audit_revision_history.revision_date)` must be
     <= `2021-09-30` — hardcoded as `max_template_date` in
     `infer_dataloader.py` (the checkpoint's data cutoff). No revision date is
     read as 9999-12-31 and dropped. This rejects 7UKV (2022-11-23) and 9NIS
     (absent from the cache) as templates;
  4. some monomer's `_chem_comp.type` must contain lowercase `"peptide"`
     (`_get_protein_chains`). The PDB writes `L-peptide linking`; **AlphaFold3
     writes `L-PEPTIDE LINKING`**, which fails and gives "No protein chains
     found" — AF3-exported CIFs must have that casing normalized.
  All 219,912 real PDB IDs in the date cache start with a digit, so a 4-char ID
  starting with a letter (we use `egfa`/`egfi`) can never collide.
  `process()` re-aligns the query to the CIF's seqres with kalign and rebuilds
  the hit indices (`_update_realigned_hit`), so the alignment inside the `.a3m`
  only feeds the prefilter's coverage ratio. Note
  `search_database/template_cache/<id>_<chain>.pkl` caches the parsed CIF and
  will shadow a changed file staged under the same ID.

- The two supplied templates in `egfr/opendde/templates/` are 268-residue EGFR
  kinase domains (auth numbering 1-268, chain A) that already carry a
  `2021-09-30` revision date. Relative to the 541-residue fusion chain
  (residue 1 = EGFR 706, so DFG = 855), the active one is an exact match and
  the **inactive one is a T790M + V948R double mutant** — V948R blocks the
  asymmetric activating dimer interface.

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
