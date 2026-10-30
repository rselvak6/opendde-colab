#!/usr/bin/env python3
"""Rename and move the .cif files out of an opendde_monomer.ipynb run's zip.

The monomer counterpart to rename_cifs.py. The dimer/fusion runs need a
conformation call per model (hence that script's drag-select and its
INHIBITOR_PDBS lookup); a monomer sweep has no such call to make, so this one
asks for nothing but the project name and which supplied template the run used,
and numbers the models exactly the same way:

  {project_name}_{template}_{model_no}.cif
  {project_name}_{template}_{model_no}_full_data.json
  {project_name}_{template}_{model_no}_summary_confidences.json

e.g. egfr_rad51_monomer_gefitinib_msa_tactive_7.cif

Hyphens in the project name become underscores, so the notebook's
`egfr-rad51-monomer-gefitinib` lines up with the underscore-separated names the
dimer script writes. {template} is the TEMPLATE_TOKENS segment
(msa_nt / msa_tactive / msa_tinactive).

Two zip shapes are handled, and which one you have is decided by the CONTENTS,
never the filename:

- per-run, from the notebook's single-run download cell, which archives
  ./output/<run_name>, so entries begin at `seed_101/...`;
- combined, from "Download every output as one zip", which archives ./output
  itself, so every entry begins with its own run directory,
  `egfr-rad51-monomer-afatinib-tactive/seed_101/...`.

The filename cannot be trusted for this: a combined archive given
ARCHIVE_NAME="egfr-rad51-monomer-afatinib" looks exactly like a per-run zip of
that project while actually holding all three template variants. Treating it as
one run would number 75 models 0..74 across runs and label them all msa_nt.
So the .cif entries are grouped by the path above their `seed_*` component:
one group means one run, several groups mean a combined archive, each group
numbered from 0 on its own and taking its project and template from its own
directory name. Output filenames carry the project and the template, so every
run of a combined archive can be written into one folder without collision.

Models are numbered in ascending (seed, sample) order, so seed 101/sample_0 is
0, seed 101/sample_1 is 1, seed 102/sample_0 is 5, matching rename_cifs.py. A
manifest.csv records the new names, run, project, template, model number and
the original paths inside the zip.

Run it with:  python rename_cifs_monomer.py

The zip layout, model ordering and sidecar handling are imported from
rename_cifs rather than reimplemented, so OpenDDE's naming convention is
described in exactly one place.
"""
import csv
import os
import re
import zipfile

from rename_cifs import (
    SIDECARS,
    TEMPLATE_TOKENS,
    TEMPLATES,
    _sidecar_json_for,
    find_cifs_in_zip,
    order_cifs,
    template_from_zip_name,
)

MANIFEST_COLUMNS = [
    "cif_name", "json_name", "summary_json_name", "run_name", "project_name",
    "template", "model_no", "seed", "sample", "source_cif", "source_json",
    "source_summary_json",
]

# OpenDDE lays every prediction out as <...>/seed_<n>/predictions/<name>.cif, so
# whatever sits above the seed component is the run directory - empty for a
# per-run zip, the run name for a combined one.
_SEED_COMPONENT_RE = re.compile(r"(?:^|/)seed[_-]?\d+(?:/|$)", re.IGNORECASE)


def run_prefix_for_entry(entry_path):
    """The run directory an entry belongs to, or "" when there is none."""
    m = _SEED_COMPONENT_RE.search(entry_path)
    if m:
        return entry_path[: m.start()].strip("/")
    # No seed_* component: fall back to the entry's directory, so an unexpected
    # layout still groups rather than silently merging into one run.
    head, _, tail = entry_path.rpartition("/")
    return head.strip("/") if tail else ""


def project_from_zip_name(zip_path):
    """The project a zip or run directory belongs to, minus any sweep suffix.

    The sweep's run names are <project>-tactive / <project>-tinactive, so
    stripping that suffix recovers the project. Accepts a zip path or a bare
    run-directory name.
    """
    name = os.path.basename(zip_path)
    if name.lower().endswith(".zip"):
        name = name[:-4]
    for suffix in ("-tactive", "-tinactive"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize_project(project_name):
    """Underscore form of the project name, for use in a filename."""
    cleaned = project_name.strip().replace("-", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def build_new_basename(project_name, template, model_no):
    """The shared base name for a model's output .cif / .json (no extension)."""
    norm = normalize_project(project_name)
    if not norm:
        raise ValueError("project name is empty")
    if template not in TEMPLATE_TOKENS:
        raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")
    return f"{norm}_{TEMPLATE_TOKENS[template]}_{model_no}"


def find_runs_in_zip(zip_path):
    """Group a zip's .cif entries into runs, each numbered from 0.

    Returns [{"run_name", "project", "template", "cifs"}] ordered by run name.
    A single group means a per-run zip, in which case the run takes its name
    from the zip itself when the entries carry no directory of their own.
    """
    groups = {}
    for c in find_cifs_in_zip(zip_path):
        groups.setdefault(run_prefix_for_entry(c["path"]), []).append(c)
    zip_stem = os.path.splitext(os.path.basename(zip_path))[0]
    runs = []
    for prefix in sorted(groups):
        name = prefix or zip_stem
        runs.append({
            "run_name": name,
            "project": project_from_zip_name(name),
            "template": template_from_zip_name(name),
            "cifs": order_cifs(groups[prefix]),
        })
    return runs


def _copy_model(zf, names, cif, base, out_dir, run):
    """Copy one model's .cif and both sidecars; returns its manifest row."""
    with zf.open(cif["path"]) as src, \
            open(os.path.join(out_dir, base + ".cif"), "wb") as dst:
        dst.write(src.read())

    copied = {}
    for zip_infix, out_suffix in SIDECARS:
        src_entry = _sidecar_json_for(cif["path"], zip_infix)
        if src_entry and src_entry in names:
            with zf.open(src_entry) as src, \
                    open(os.path.join(out_dir, base + out_suffix), "wb") as dst:
                dst.write(src.read())
            copied[out_suffix] = (base + out_suffix, src_entry)
        else:
            copied[out_suffix] = ("", src_entry or "")

    full_name, full_src = copied["_full_data.json"]
    summary_name, summary_src = copied["_summary_confidences.json"]
    return {
        "cif_name": base + ".cif",
        "json_name": full_name,
        "summary_json_name": summary_name,
        "run_name": run["run_name"],
        "project_name": run["project"],
        "template": run["template"],
        "model_no": cif["index"],
        "seed": cif["seed"],
        "sample": cif["sample"],
        "source_cif": cif["path"],
        "source_json": full_src,
        "source_summary_json": summary_src,
    }


def process_runs(zip_path, out_dir, runs):
    """Copy every selected run's models to out_dir, renamed, numbered per run.

    Each sidecar is copied only when present in the zip (the full_data one needs
    the run to have used --need_atom_confidence true). Writes one manifest.csv
    covering every run and returns its rows.
    """
    if not runs:
        raise ValueError("no runs selected")
    # Validate before writing anything, so a bad project or template cannot
    # leave a half-renamed folder behind.
    seen = {}
    for run in runs:
        build_new_basename(run["project"], run["template"], 0)
        key = (normalize_project(run["project"]), run["template"])
        if key in seen:
            raise ValueError(
                "runs %r and %r both map to %s_%s_* - their files would "
                "overwrite each other; deselect one"
                % (seen[key], run["run_name"], key[0], TEMPLATE_TOKENS[key[1]]))
        seen[key] = run["run_name"]

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for run in runs:
            for cif in run["cifs"]:
                base = build_new_basename(run["project"], run["template"],
                                          cif["index"])
                rows.append(_copy_model(zf, names, cif, base, out_dir, run))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def process(zip_path, out_dir, ordered, project_name, template):
    """Single-run convenience wrapper around process_runs."""
    return process_runs(zip_path, out_dir, [{
        "run_name": os.path.splitext(os.path.basename(zip_path))[0],
        "project": project_name,
        "template": template,
        "cifs": ordered,
    }])


def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    class App:
        def __init__(self, root):
            self.root = root
            self.runs = []
            self.combined = False
            root.title("OpenDDE monomer .cif renamer")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)

            frm = ttk.Frame(root, padding=10)
            frm.grid(sticky="nsew")
            frm.columnconfigure(1, weight=1)

            # --- inputs ---
            ttk.Label(frm, text="Input .zip:").grid(row=0, column=0, sticky="w")
            self.zip_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.zip_var).grid(
                row=0, column=1, sticky="ew", padx=4)
            ttk.Button(frm, text="Browse...", command=self._browse_zip).grid(
                row=0, column=2)

            ttk.Label(frm, text="Output folder:").grid(
                row=1, column=0, sticky="w", pady=(4, 0))
            self.out_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.out_var).grid(
                row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
            ttk.Button(frm, text="Browse...", command=self._browse_out).grid(
                row=1, column=2, pady=(4, 0))

            ttk.Label(frm, text="Project name:").grid(
                row=2, column=0, sticky="w", pady=(4, 0))
            self.project_var = tk.StringVar()
            self.project_entry = ttk.Entry(frm, textvariable=self.project_var)
            self.project_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
            ttk.Label(frm, text="(from the zip)", foreground="#555").grid(
                row=2, column=2, sticky="w", pady=(4, 0))

            ttk.Label(frm, text="Supplied template:").grid(
                row=3, column=0, sticky="w", pady=(4, 0))
            self.template_var = tk.StringVar(value="none")
            self.template_cb = ttk.Combobox(
                frm, textvariable=self.template_var, values=TEMPLATES,
                state="readonly")
            self.template_cb.grid(row=3, column=1, sticky="w", padx=4, pady=(4, 0))
            ttk.Label(frm, text="(from the -tactive/-tinactive suffix)",
                      foreground="#555").grid(row=3, column=2, sticky="w", pady=(4, 0))

            self.project_var.trace_add("write", lambda *_: self._refresh())
            self.template_var.trace_add("write", lambda *_: self._refresh())

            # --- status + list ---
            self.count_var = tk.StringVar(
                value="Pick a run's .zip, or a combined one, to scan it.")
            ttk.Label(frm, textvariable=self.count_var).grid(
                row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
            self.list_label = tk.StringVar(value="Preview of what will be written:")
            ttk.Label(frm, textvariable=self.list_label).grid(
                row=5, column=0, columnspan=3, sticky="w")

            lb_frame = ttk.Frame(frm)
            lb_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(2, 0))
            frm.rowconfigure(6, weight=1)
            lb_frame.rowconfigure(0, weight=1)
            lb_frame.columnconfigure(0, weight=1)
            self.listbox = tk.Listbox(lb_frame, selectmode="none", height=16,
                                      activestyle="none", exportselection=False)
            sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
            self.listbox.configure(yscrollcommand=sb.set)
            self.listbox.grid(row=0, column=0, sticky="nsew")
            sb.grid(row=0, column=1, sticky="ns")

            # --- action ---
            ttk.Button(frm, text="Rename & move", command=self._process).grid(
                row=7, column=0, columnspan=3, pady=(10, 0))
            self.status_var = tk.StringVar(value="")
            ttk.Label(frm, textvariable=self.status_var, foreground="#555").grid(
                row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))

            root.minsize(760, 580)

        # --- callbacks ---
        def _browse_zip(self):
            path = filedialog.askopenfilename(
                title="Select a run's .zip, or a combined output .zip",
                filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")])
            if path:
                self.zip_var.set(path)
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
                self.runs = find_runs_in_zip(zip_path)
            except zipfile.BadZipFile:
                messagebox.showerror("Error", "That file is not a valid .zip archive.")
                return
            self.combined = len(self.runs) > 1
            n_cif = sum(len(r["cifs"]) for r in self.runs)
            if not self.runs:
                self.count_var.set("No .cif files found inside that .zip.")
            elif self.combined:
                self.count_var.set(
                    f"Combined archive: {len(self.runs)} runs, {n_cif} .cif total. "
                    f"Each run is numbered from 0 separately.")
            else:
                self.count_var.set(
                    f"{n_cif} .cif file(s) found - they will be numbered "
                    f"0 .. {n_cif - 1}.")

            # In combined mode the project/template come from each run's own
            # directory, so the single-value fields would be meaningless.
            if self.combined:
                self.project_entry.configure(state="disabled")
                self.template_cb.configure(state="disabled")
                self.listbox.configure(selectmode="extended")
                self.list_label.set(
                    "Runs found - all selected; drag or ctrl-click to narrow:")
            else:
                self.project_entry.configure(state="normal")
                self.template_cb.configure(state="readonly")
                self.listbox.configure(selectmode="none")
                self.list_label.set("Preview of what will be written:")
                if self.runs:
                    self.project_var.set(self.runs[0]["project"])
                    self.template_var.set(self.runs[0]["template"])
            self.status_var.set("")
            self._refresh()
            if self.combined:
                self.listbox.selection_set(0, "end")

        def _selected_runs(self):
            if not self.combined:
                if not self.runs:
                    return []
                run = dict(self.runs[0])
                run["project"] = self.project_var.get().strip()
                run["template"] = self.template_var.get().strip() or "none"
                return [run]
            idx = self.listbox.curselection()
            return [self.runs[i] for i in idx if i < len(self.runs)]

        def _refresh(self):
            self.listbox.delete(0, "end")
            if self.combined:
                for run in self.runs:
                    try:
                        sample = build_new_basename(
                            run["project"], run["template"], 0) + ".cif"
                    except ValueError as exc:
                        sample = f"<{exc}>"
                    self.listbox.insert(
                        "end",
                        f"{run['run_name']:<46} {len(run['cifs']):>3} models   "
                        f"template={run['template']:<9} -> {sample}")
                return
            project = self.project_var.get().strip()
            template = self.template_var.get().strip() or "none"
            for run in self.runs:
                for c in run["cifs"]:
                    seed = c["seed"] if c["seed"] is not None else "?"
                    sample = c["sample"] if c["sample"] is not None else "?"
                    try:
                        new = build_new_basename(project, template, c["index"]) + ".cif"
                    except ValueError as exc:
                        new = f"<{exc}>"
                    self.listbox.insert(
                        "end",
                        f"{c['index']:>4}   seed {seed} / sample {sample}   ->   {new}")

        def _process(self):
            if not self.runs:
                messagebox.showwarning("Nothing to do",
                                       "Scan a .zip containing .cif files first.")
                return
            out_dir = self.out_var.get().strip()
            if not out_dir:
                messagebox.showerror("Error", "Choose an output folder.")
                return
            runs = self._selected_runs()
            if not runs:
                messagebox.showerror("Error", "Select at least one run.")
                return
            if not self.combined and not normalize_project(runs[0]["project"]):
                messagebox.showerror("Error", "Enter a project name.")
                return
            try:
                rows = process_runs(self.zip_var.get().strip(), out_dir, runs)
            except Exception as exc:  # surface any extraction/IO error to the user
                messagebox.showerror("Error", f"Failed while writing files:\n{exc}")
                return
            n_full = sum(1 for r in rows if r["json_name"])
            n_sum = sum(1 for r in rows if r["summary_json_name"])
            missing = (len(rows) - n_full) + (len(rows) - n_sum)
            per_run = "\n".join(
                "  %s -> %s_%s_0 .. _%d"
                % (r["run_name"], normalize_project(r["project"]),
                   TEMPLATE_TOKENS[r["template"]], len(r["cifs"]) - 1)
                for r in runs)
            summary = (f"Copied {len(rows)} .cif + {n_full} full_data + {n_sum} "
                       f"summary_confidences file(s) to:\n{out_dir}\n\n"
                       f"{len(runs)} run(s):\n{per_run}\n\n"
                       f"Manifest: manifest.csv")
            if missing:
                summary += (f"\n\nNote: {missing} sidecar(s) were missing from the zip "
                            f"(full_data needs --need_atom_confidence true).")
            self.status_var.set(
                f"Done - {len(rows)} .cif + {n_full} full_data + {n_sum} summary "
                f"+ manifest.csv written to {out_dir}")
            messagebox.showinfo("Done", summary)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
