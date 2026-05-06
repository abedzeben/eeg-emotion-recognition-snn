# Step 06.5 — Multi-subject DEAP Loading

## Goal

- Load **all** available DEAP subject files (`s*.dat`) from `data/raw/`, concatenate them into a single dataset, then run the existing pipeline (preprocessing → features → labels → baseline → SNN) on the combined data.

## Files modified

- `src/load_data.py`
- `main.py`

## What was implemented

- **`load_all_deap_files(folder_path)`** in `src/load_data.py`
  - Finds files matching `s*.dat` inside the provided folder (e.g., `data/raw/`).
  - Sorts them by filename.
  - Loads each file using existing `load_deap_file` logic and collects `X` and `y`.
  - Concatenates across subjects (trial axis):
    - `X_all`: `(num_subjects * 40, 40, 8064)`
    - `y_all`: `(num_subjects * 40, 4)`
  - Prints:
    - number of files loaded
    - final `X_all` shape
    - final `y_all` shape

- **`main.py`**
  - Uses `load_all_deap_files("data/raw")` instead of loading only `s01.dat`.
  - If no `.dat` files exist, prints:
    - `No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/`
  - Keeps the rest of the pipeline unchanged.

## How to run

```bash
python main.py
```

## Expected output

- If no subject files exist in `data/raw/`:
  - `No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/`
- Otherwise:
  - Per-file inspection prints from `load_deap_file`
  - A summary printout from `load_all_deap_files`:
    - number of files loaded
    - final concatenated shapes
  - Then the existing pipeline prints:
    - preprocessing completed + shape
    - feature extraction completed + shape
    - labels created + shape
    - baseline model trained + accuracy
    - SNN model trained + accuracy

## Notes / limitations

- This step **does not** change feature extraction or model architectures; it only extends data loading from single-subject to multi-subject.
- Loading many `.dat` files will increase runtime and memory usage; start with a few subject files if needed.
