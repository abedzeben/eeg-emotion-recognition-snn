# Step 02 — Load DEAP Dataset

## Goal

- Implement DEAP `.dat` file loading and print basic dataset inspection information (shapes and a sample label row).

## Files modified

- `src/load_data.py` (extended with DEAP loading)
- `main.py` (created)

## What was implemented

- `load_deap_file(file_path)` in `src/load_data.py`
  - Loads a DEAP subject file (e.g., `s01.dat`) via `pickle.load(..., encoding="latin1")`
  - Returns:
    - `X = data["data"]`
    - `y = data["labels"]`
  - Prints:
    - `X shape`
    - `y shape`
    - `first trial shape`
    - `first label row`
- `main.py`
  - Checks for `data/raw/s01.dat`
  - If missing, prints: `DEAP file not found. Please place s01.dat inside data/raw/`
  - If present, loads and inspects the file via `load_deap_file`

## How to run

```bash
python main.py
```

## Expected output

If `data/raw/s01.dat` exists, you should see output similar to:

- `X shape: (.., .., .., ..)`
- `y shape: (.., ..)`
- `first trial shape: (.., .., ..)`
- `first label row: [.. .. .. ..]`

If `data/raw/s01.dat` does not exist:

- `DEAP file not found. Please place s01.dat inside data/raw/`

## Notes / limitations

- This step **only loads and inspects** data. No preprocessing or modeling is implemented yet.
- The DEAP `.dat` structure is expected to contain `data` and `labels` keys.
