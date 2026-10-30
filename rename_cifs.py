#!/usr/bin/env python3
"""Rename and move OpenDDE prediction .cif files out of a run's output.zip.

For the homodimer/fusion runs with an inhibitor and a judged conformation. The
monomer sweeps from opendde_monomer.ipynb have no conformation call to make, so
they use the simpler rename_cifs_monomer.py, which imports the shared helpers
from here.

Standalone Tkinter GUI (standard library only). Pick a run's output.zip, an
output folder, the inhibitor, which supplied template the run used, and the
construct; the .cif files nested in the zip are listed in ascending
(seed, sample) order and numbered 0, 1, 2, ...  Drag-select the ones in the
ACTIVE conformation (anything left unselected is treated as inactive). On
"Rename & move", each model's .cif AND both of its confidence sidecars
(*_full_data_sample_*.json and *_summary_confidence_sample_*.json) are copied to
the output folder, renamed to

  egfr_rad51_{construct}_{inhibitor}_{template}_{active|inactive}_{PDB}_{model_no}.cif
  ..._{model_no}_full_data.json
  ..._{model_no}_summary_confidences.json

where {PDB} is the closest reference for that conformation (INHIBITOR_PDBS,
e.g. afatinib -> 4G5J active / 2GS7 inactive) and {template} records whether
the prediction was seeded with a supplied template (TEMPLATE_TOKENS:
msa_nt / msa_tactive / msa_tinactive). Choosing "none" (no inhibitor)
drops the inhibitor segment, e.g. egfr_rad51_dimer_msa_nt_active_2GS2_0.cif.
A manifest.csv records every file's new names, conformation, inhibitor,
template, construct, PDB, model number, and its original paths in the zip.

Note the two independent "active/inactive" axes, which the name keeps in this
order: {template} is the conformation of the template the model was PREDICTED
with, and the following segment is the conformation the model was JUDGED to be
in. So ..._msa_tactive_inactive_2GS7_3 means "predicted with the active
template, came out inactive".

Run it with:  python rename_cifs.py

This is a starting point - the GUI logic is kept thin and the file handling
lives in the pure functions below (find_cifs_in_zip / order_cifs / process) so
it is easy to test and build on.
"""
import csv
import os
import re
import zipfile

# Inhibitor options for the dropdown. "none" = no inhibitor; when chosen the
# inhibitor segment is dropped from the output filename (see build_new_basename).
INHIBITORS = ["none", "erlotinib", "afatinib", "osimertinib", "alflutinib", "lazertinib",
              "bh30643", "cetuximab"]

# Closest-matching reference PDB per inhibitor and conformation, used in the
# output filename. Mirrors the activation-loop viewer's template pairs: the
# drug's own co-crystal for the state(s) it has, the wild-type EGFR kinase
# reference (active 2GS2 / inactive 2GS7) for the opposite state. "none" uses
# the wild-type references for both states; cetuximab is an extracellular-domain
# antibody with no kinase co-crystal, so it too falls back to wild-type, as does
# bh30643 (no co-crystal deposited) - for those the PDB in the filename records
# the wild-type conformation matched, not a drug-bound template.
INHIBITOR_PDBS = {
    "none":        {"active": "2GS2", "inactive": "2GS7"},
    "erlotinib":   {"active": "1M17", "inactive": "4HJO"},
    "afatinib":    {"active": "4G5J", "inactive": "2GS7"},
    "osimertinib": {"active": "4ZAU", "inactive": "2GS7"},
    "alflutinib":  {"active": "9NIS", "inactive": "2GS7"},
    "lazertinib":  {"active": "7UKV", "inactive": "2GS7"},
    "bh30643":     {"active": "2GS2", "inactive": "2GS7"},
    "cetuximab":   {"active": "2GS2", "inactive": "2GS7"},
}

# Which supplied template the run was seeded with, and the filename segment for
# each. "none" keeps the original msa_nt ("MSA, no template") token so existing
# output folders stay consistent; the other two mirror the notebook's run-name
# suffixes (<project>-tactive / -tinactive) so a zip can be traced back to the
# run that produced it.
TEMPLATES = ["none", "active", "inactive"]
TEMPLATE_TOKENS = {
    "none": "msa_nt",
    "active": "msa_tactive",
    "inactive": "msa_tinactive",
}

# The construct the models are of, as the second filename segment. The fusion
# runs this was written for are homodimers; monomer runs (opendde_monomer.ipynb,
# inputs named egfr-rad51-monomer-*) must be "monomer" or every output file is
# mislabelled. Note egfr-rad51-fusion-monomer is the fusion chain predicted on
# its own, so it is a monomer too.
CONSTRUCTS = ["dimer", "monomer"]

_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)
_SAMPLE_RE = re.compile(r"sample[_-]?(\d+)", re.IGNORECASE)
# The notebook's template sweep names its runs <project>-tactive / -tinactive
# and the download cell zips ./output/<run_name>, so the zip name usually says
# which template was used.
_TEMPLATE_IN_NAME_RE = re.compile(r"-t(active|inactive)\b", re.IGNORECASE)


def template_from_zip_name(zip_path):
    """Guess the template from a zip produced by the notebook's sweep.

    Returns "active"/"inactive" when the name carries the sweep's suffix, else
    "none". Only a default for the dropdown - the user can always override it.
    """
    m = _TEMPLATE_IN_NAME_RE.search(os.path.basename(zip_path))
    return m.group(1).lower() if m else "none"


def construct_from_zip_name(zip_path):
    """Guess the construct from the zip name; a default for the dropdown only.

    "monomer" anywhere in the name wins, which also covers
    egfr-rad51-fusion-monomer (the fusion chain predicted on its own);
    everything else falls back to "dimer".
    """
    return "monomer" if "monomer" in os.path.basename(zip_path).lower() else "dimer"


def find_cifs_in_zip(zip_path):
    """Return [{path, seed, sample}, ...] for every .cif entry in the zip.

    seed/sample are parsed from the entry path following OpenDDE's convention
    (seed_{seed}/predictions/{name}_sample_{rank}.cif); either is None if the
    path does not carry it.
    """
    found = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".cif"):
                continue
            seed = _SEED_RE.search(name)
            sample = _SAMPLE_RE.search(name)
            found.append({
                "path": name,
                "seed": int(seed.group(1)) if seed else None,
                "sample": int(sample.group(1)) if sample else None,
            })
    return found


def order_cifs(cifs):
    """Sort by (seed, sample) ascending and assign 0-based model indices.

    Entries missing a seed and/or sample sort after the parsed ones (by path)
    but are still numbered, so nothing is silently dropped. Mutates and returns
    the given dicts with "index" (the model number) added.
    """
    def sort_key(c):
        return (c["seed"] is None, c["seed"] or 0,
                c["sample"] is None, c["sample"] or 0, c["path"])

    ordered = sorted(cifs, key=sort_key)
    for i, c in enumerate(ordered):
        c["index"] = i
    return ordered


def build_new_basename(inhibitor, conformation, pdb_id, model_no, template="none",
                       construct="dimer"):
    """The shared base name for a model's output .cif / .json (no extension).

    The inhibitor segment is omitted when inhibitor == "none" (no inhibitor),
    e.g. egfr_rad51_dimer_msa_nt_active_2GS2_0. `template` is the supplied
    template the run was seeded with and becomes the TEMPLATE_TOKENS segment,
    which sits immediately before the judged conformation. `construct` is
    "dimer" or "monomer".
    """
    drug = "" if inhibitor == "none" else f"{inhibitor}_"
    tpl = TEMPLATE_TOKENS[template]
    return f"egfr_rad51_{construct}_{drug}{tpl}_{conformation}_{pdb_id}_{model_no}"


# Per-model sidecars OpenDDE writes next to each .cif in the zip, and the
# suffix each gets on the way out. The output suffixes are the AF3-style names
# ipsae.py expects (it looks for "*_full_data*.json" and derives the summary
# path from it), so a generated folder can be fed to ipsae.py directly. Note
# OpenDDE writes "summary_confidence" (singular) in the zip while ipsae.py
# wants "summary_confidences" (plural) - hence the rename rather than a copy.
SIDECARS = (
    ("_full_data_sample_", "_full_data.json"),
    ("_summary_confidence_sample_", "_summary_confidences.json"),
)


def _sidecar_json_for(cif_path, zip_infix):
    """Expected zip entry of the {zip_infix}{rank}.json sidecar beside a cif.

    OpenDDE writes the cif as {name}_sample_{rank}.cif and its sidecars as
    {name}_full_data_sample_{rank}.json / {name}_summary_confidence_sample_
    {rank}.json in the same predictions/ directory. Returns None if the cif
    name does not match.
    """
    base = cif_path.rsplit("/", 1)[-1]
    prefix = cif_path[: len(cif_path) - len(base)]  # dir part incl. trailing '/'
    m = re.match(r"(.+)_sample_(\d+)\.cif$", base)
    if not m:
        return None
    return f"{prefix}{m.group(1)}{zip_infix}{m.group(2)}.json"


def process(zip_path, out_dir, ordered, active_indices, inhibitor, template="none",
            construct="dimer"):
    """Copy each model's .cif and both confidence sidecars to out_dir, renamed.

    Output names, sharing one base per model:
      egfr_rad51_dimer_{inhibitor}_{template}_{active|inactive}_{PDB}_{model_no}.cif
      ..._{model_no}_full_data.json            (token_pair_pae / atom_plddt)
      ..._{model_no}_summary_confidences.json  (iptm / ptm / chain_pair_iptm)
    Each sidecar is copied only when present in the zip (full_data needs the
    run to have used --need_atom_confidence true). {PDB} is the
    closest-matching reference for the chosen conformation (INHIBITOR_PDBS) and
    {template} the TEMPLATE_TOKENS segment for the supplied template the run
    used. active_indices is the set of ordered-list positions flagged active.
    Writes manifest.csv and returns its rows.
    """
    active_indices = set(active_indices)
    pdbs = INHIBITOR_PDBS[inhibitor]
    if template not in TEMPLATE_TOKENS:
        raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")
    if construct not in CONSTRUCTS:
        raise ValueError(f"unknown construct {construct!r}; expected one of {CONSTRUCTS}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for c in ordered:
            conformation = "active" if c["index"] in active_indices else "inactive"
            pdb_id = pdbs[conformation]
            base = build_new_basename(inhibitor, conformation, pdb_id, c["index"],
                                      template, construct)

            with zf.open(c["path"]) as src, open(os.path.join(out_dir, base + ".cif"), "wb") as dst:
                dst.write(src.read())

            copied = {}
            for zip_infix, out_suffix in SIDECARS:
                src_entry = _sidecar_json_for(c["path"], zip_infix)
                if src_entry and src_entry in names:
                    with zf.open(src_entry) as src, \
                            open(os.path.join(out_dir, base + out_suffix), "wb") as dst:
                        dst.write(src.read())
                    copied[out_suffix] = (base + out_suffix, src_entry)
                else:
                    copied[out_suffix] = ("", src_entry or "")

            full_name, full_src = copied["_full_data.json"]
            summary_name, summary_src = copied["_summary_confidences.json"]
            rows.append({
                "cif_name": base + ".cif",
                "json_name": full_name,
                "summary_json_name": summary_name,
                "conformation": conformation,
                "inhibitor": inhibitor,
                "template": template,
                "construct": construct,
                "pdb_id": pdb_id,
                "model_no": c["index"],
                "seed": c["seed"],
                "sample": c["sample"],
                "source_cif": c["path"],
                "source_json": full_src,
                "source_summary_json": summary_src,
            })
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cif_name", "json_name", "summary_json_name", "conformation", "inhibitor",
            "template", "construct", "pdb_id", "model_no", "seed", "sample",
            "source_cif", "source_json", "source_summary_json"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    class App:
        def __init__(self, root):
            self.root = root
            self.ordered = []
            root.title("OpenDDE .cif renamer")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)

            frm = ttk.Frame(root, padding=10)
            frm.grid(sticky="nsew")
            frm.columnconfigure(1, weight=1)

            # --- inputs ---
            ttk.Label(frm, text="Input .zip:").grid(row=0, column=0, sticky="w")
            self.zip_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.zip_var).grid(row=0, column=1, sticky="ew", padx=4)
            ttk.Button(frm, text="Browse...", command=self._browse_zip).grid(row=0, column=2)

            ttk.Label(frm, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.out_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
            ttk.Button(frm, text="Browse...", command=self._browse_out).grid(row=1, column=2, pady=(4, 0))

            ttk.Label(frm, text="Inhibitor:").grid(row=2, column=0, sticky="w", pady=(4, 0))
            self.inhibitor_var = tk.StringVar()
            cb = ttk.Combobox(frm, textvariable=self.inhibitor_var, values=INHIBITORS,
                              state="readonly")
            cb.grid(row=2, column=1, sticky="w", padx=4, pady=(4, 0))
            cb.bind("<<ComboboxSelected>>", lambda e: self._scan())
            ttk.Button(frm, text="Rescan .zip", command=self._scan).grid(row=2, column=2, pady=(4, 0))

            ttk.Label(frm, text="Supplied template:").grid(row=3, column=0, sticky="w", pady=(4, 0))
            self.template_var = tk.StringVar(value="none")
            tcb = ttk.Combobox(frm, textvariable=self.template_var, values=TEMPLATES,
                               state="readonly")
            tcb.grid(row=3, column=1, sticky="w", padx=4, pady=(4, 0))
            ttk.Label(frm, text="(auto-set from the zip name when it says -tactive/-tinactive)",
                      foreground="#555").grid(row=3, column=2, sticky="w", pady=(4, 0))

            ttk.Label(frm, text="Construct:").grid(row=4, column=0, sticky="w", pady=(4, 0))
            self.construct_var = tk.StringVar(value="dimer")
            ccb = ttk.Combobox(frm, textvariable=self.construct_var, values=CONSTRUCTS,
                               state="readonly")
            ccb.grid(row=4, column=1, sticky="w", padx=4, pady=(4, 0))
            ttk.Label(frm, text='(auto-set to monomer when the zip name says so)',
                      foreground="#555").grid(row=4, column=2, sticky="w", pady=(4, 0))

            # --- status + list ---
            self.count_var = tk.StringVar(value="Pick a .zip and an inhibitor to scan for .cif files.")
            ttk.Label(frm, textvariable=self.count_var).grid(
                row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
            ttk.Label(frm, text="Drag to select ACTIVE conformations "
                                "(anything unselected is treated as inactive):").grid(
                row=6, column=0, columnspan=3, sticky="w")

            lb_frame = ttk.Frame(frm)
            lb_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(2, 0))
            frm.rowconfigure(7, weight=1)
            lb_frame.rowconfigure(0, weight=1)
            lb_frame.columnconfigure(0, weight=1)
            self.listbox = tk.Listbox(lb_frame, selectmode="extended", height=16,
                                      activestyle="none", exportselection=False)
            sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
            self.listbox.configure(yscrollcommand=sb.set)
            self.listbox.grid(row=0, column=0, sticky="nsew")
            sb.grid(row=0, column=1, sticky="ns")

            # --- action ---
            ttk.Button(frm, text="Rename & move", command=self._process).grid(
                row=8, column=0, columnspan=3, pady=(10, 0))
            self.status_var = tk.StringVar(value="")
            ttk.Label(frm, textvariable=self.status_var, foreground="#555").grid(
                row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))

            root.minsize(640, 560)

        # --- callbacks ---
        def _browse_zip(self):
            path = filedialog.askopenfilename(
                title="Select a run's output.zip",
                filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")])
            if path:
                self.zip_var.set(path)
                self.template_var.set(template_from_zip_name(path))
                self.construct_var.set(construct_from_zip_name(path))
                if self.inhibitor_var.get():
                    self._scan()

        def _browse_out(self):
            path = filedialog.askdirectory(title="Select the output folder")
            if path:
                self.out_var.set(path)

        def _scan(self):
            zip_path = self.zip_var.get().strip()
            if not zip_path:
                return
            if not os.path.isfile(zip_path):
                messagebox.showerror("Error", f"Zip not found:\n{zip_path}")
                return
            try:
                cifs = find_cifs_in_zip(zip_path)
            except zipfile.BadZipFile:
                messagebox.showerror("Error", "That file is not a valid .zip archive.")
                return
            self.ordered = order_cifs(cifs)
            self.listbox.delete(0, "end")
            for c in self.ordered:
                seed = c["seed"] if c["seed"] is not None else "?"
                sample = c["sample"] if c["sample"] is not None else "?"
                self.listbox.insert(
                    "end",
                    f"{c['index']:>4}    seed {seed} / sample {sample}    "
                    f"({os.path.basename(c['path'])})")
            if self.ordered:
                self.count_var.set(f"{len(self.ordered)} .cif file(s) found - they will be "
                                   f"renamed 0 .. {len(self.ordered) - 1}.")
            else:
                self.count_var.set("No .cif files found inside that .zip.")
            self.status_var.set("")

        def _process(self):
            if not self.ordered:
                messagebox.showwarning("Nothing to do", "Scan a .zip containing .cif files first.")
                return
            out_dir = self.out_var.get().strip()
            if not out_dir:
                messagebox.showerror("Error", "Choose an output folder.")
                return
            inhibitor = self.inhibitor_var.get().strip()
            if not inhibitor:
                messagebox.showerror("Error", "Select an inhibitor.")
                return
            template = self.template_var.get().strip() or "none"
            construct = self.construct_var.get().strip() or "dimer"
            active = set(self.listbox.curselection())
            try:
                rows = process(self.zip_var.get().strip(), out_dir, self.ordered, active,
                               inhibitor, template, construct)
            except Exception as exc:  # surface any extraction/IO error to the user
                messagebox.showerror("Error", f"Failed while writing files:\n{exc}")
                return
            n_active = sum(1 for r in rows if r["conformation"] == "active")
            n_full = sum(1 for r in rows if r["json_name"])
            n_sum = sum(1 for r in rows if r["summary_json_name"])
            missing = (len(rows) - n_full) + (len(rows) - n_sum)
            summary = (f"Copied {len(rows)} .cif + {n_full} full_data + {n_sum} "
                       f"summary_confidences file(s) to:\n{out_dir}\n\n"
                       f"{n_active} active, {len(rows) - n_active} inactive.\n"
                       f"Inhibitor: {inhibitor}\n"
                       f"Template: {template} ({TEMPLATE_TOKENS[template]})\n"
                       f"Construct: {construct}\n"
                       f"Manifest: manifest.csv")
            if missing:
                summary += (f"\n\nNote: {missing} sidecar(s) were missing from the zip "
                            f"(full_data needs --need_atom_confidence true).")
            self.status_var.set(f"Done - {len(rows)} .cif + {n_full} full_data + {n_sum} "
                                f"summary + manifest.csv written to {out_dir}")
            messagebox.showinfo("Done", summary)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
