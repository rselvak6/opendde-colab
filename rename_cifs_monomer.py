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

Both are filled in from the zip name, because the notebook's download cell zips
./output/<run_name> and the sweep names its runs <project>-tactive /
-tinactive: picking egfr-rad51-monomer-gefitinib-tinactive.zip gives a project
of egfr-rad51-monomer-gefitinib and a template of inactive. Either can be
overridden before writing.

Models are numbered in ascending (seed, sample) order, so seed 101/sample_0 is
0, seed 101/sample_1 is 1, seed 102/sample_0 is 5, matching rename_cifs.py. A
manifest.csv records the new names, project, template, model number and the
original paths inside the zip.

Run it with:  python rename_cifs_monomer.py

The zip layout, model ordering and sidecar handling are imported from
rename_cifs rather than reimplemented, so OpenDDE's naming convention is
described in exactly one place.
"""
import csv
import os
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
    "cif_name", "json_name", "summary_json_name", "project_name", "template",
    "model_no", "seed", "sample", "source_cif", "source_json",
    "source_summary_json",
]


def project_from_zip_name(zip_path):
    """The project name a zip belongs to, with any sweep suffix removed.

    The sweep's run names are <project>-tactive / <project>-tinactive and the
    download cell zips ./output/<run_name>, so stripping that suffix recovers
    the project the user chose. A no-template run's zip is already the bare
    project name.
    """
    name = os.path.basename(zip_path)
    for ext in (".zip",):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
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


def process(zip_path, out_dir, ordered, project_name, template):
    """Copy each model's .cif and both confidence sidecars to out_dir, renamed.

    Each sidecar is copied only when it is present in the zip (the full_data one
    needs the run to have used --need_atom_confidence true). Writes manifest.csv
    and returns its rows.
    """
    # Validate before writing anything, so a bad template cannot leave a
    # half-renamed folder behind.
    build_new_basename(project_name, template, 0)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for c in ordered:
            base = build_new_basename(project_name, template, c["index"])

            with zf.open(c["path"]) as src, \
                    open(os.path.join(out_dir, base + ".cif"), "wb") as dst:
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
                "project_name": project_name,
                "template": template,
                "model_no": c["index"],
                "seed": c["seed"],
                "sample": c["sample"],
                "source_cif": c["path"],
                "source_json": full_src,
                "source_summary_json": summary_src,
            })
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
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
            ttk.Entry(frm, textvariable=self.project_var).grid(
                row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
            ttk.Label(frm, text="(from the zip name)", foreground="#555").grid(
                row=2, column=2, sticky="w", pady=(4, 0))

            ttk.Label(frm, text="Supplied template:").grid(
                row=3, column=0, sticky="w", pady=(4, 0))
            self.template_var = tk.StringVar(value="none")
            tcb = ttk.Combobox(frm, textvariable=self.template_var, values=TEMPLATES,
                               state="readonly")
            tcb.grid(row=3, column=1, sticky="w", padx=4, pady=(4, 0))
            ttk.Label(frm, text="(from the zip's -tactive/-tinactive suffix)",
                      foreground="#555").grid(row=3, column=2, sticky="w", pady=(4, 0))

            # Keep the preview honest as either field is edited.
            self.project_var.trace_add("write", lambda *_: self._refresh_preview())
            self.template_var.trace_add("write", lambda *_: self._refresh_preview())

            # --- status + preview ---
            self.count_var = tk.StringVar(
                value="Pick a run's .zip to scan it for .cif files.")
            ttk.Label(frm, textvariable=self.count_var).grid(
                row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
            ttk.Label(frm, text="Preview of what will be written:").grid(
                row=5, column=0, columnspan=3, sticky="w")

            lb_frame = ttk.Frame(frm)
            lb_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(2, 0))
            frm.rowconfigure(6, weight=1)
            lb_frame.rowconfigure(0, weight=1)
            lb_frame.columnconfigure(0, weight=1)
            # Display only - unlike the dimer renamer there is nothing to select.
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

            root.minsize(680, 560)

        # --- callbacks ---
        def _browse_zip(self):
            path = filedialog.askopenfilename(
                title="Select a run's output .zip",
                filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")])
            if path:
                self.zip_var.set(path)
                self.project_var.set(project_from_zip_name(path))
                self.template_var.set(template_from_zip_name(path))
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
            if self.ordered:
                self.count_var.set(
                    f"{len(self.ordered)} .cif file(s) found - they will be numbered "
                    f"0 .. {len(self.ordered) - 1}.")
            else:
                self.count_var.set("No .cif files found inside that .zip.")
            self.status_var.set("")
            self._refresh_preview()

        def _refresh_preview(self):
            self.listbox.delete(0, "end")
            project = self.project_var.get().strip()
            template = self.template_var.get().strip() or "none"
            for c in self.ordered:
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
            if not self.ordered:
                messagebox.showwarning("Nothing to do",
                                       "Scan a .zip containing .cif files first.")
                return
            out_dir = self.out_var.get().strip()
            if not out_dir:
                messagebox.showerror("Error", "Choose an output folder.")
                return
            project = self.project_var.get().strip()
            if not normalize_project(project):
                messagebox.showerror("Error", "Enter a project name.")
                return
            template = self.template_var.get().strip() or "none"
            try:
                rows = process(self.zip_var.get().strip(), out_dir, self.ordered,
                               project, template)
            except Exception as exc:  # surface any extraction/IO error to the user
                messagebox.showerror("Error", f"Failed while writing files:\n{exc}")
                return
            n_full = sum(1 for r in rows if r["json_name"])
            n_sum = sum(1 for r in rows if r["summary_json_name"])
            missing = (len(rows) - n_full) + (len(rows) - n_sum)
            summary = (f"Copied {len(rows)} .cif + {n_full} full_data + {n_sum} "
                       f"summary_confidences file(s) to:\n{out_dir}\n\n"
                       f"Project: {project}\n"
                       f"Template: {template} ({TEMPLATE_TOKENS[template]})\n"
                       f"Numbered 0 .. {len(rows) - 1}\n"
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
