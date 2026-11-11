"""Load the S3231 dataset (3 folders = 1 dataset) into one table."""
import re
from pathlib import Path
import pandas as pd

DATA = Path("Data")
STRESS_IDS = {"p03", "p04", "p19"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
NAME_RE = re.compile(r"^(p\d+)-f?(\d+)-(scan|photo|hard|hardphoto)(\d*)$", re.I)
ANSWERS = list("ABCDE")


def _scan_images(root: Path):
    rows, skipped = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append(str(p))
            continue
        student, form, kind, extra = m.groups()
        student = student.lower()
        kind = kind.lower().replace("hardphoto", "hard")
        rows.append(dict(path=str(p), fname=p.name, student=student,
                         form_id=f"{student}-f{int(form):02d}", kind=kind,
                         extra=extra or "1"))
    return pd.DataFrame(rows), skipped


def _find_labels_csv() -> Path:
    for ds in sorted(DATA.glob("AI-for-Vision-Dataset*")):
        f = ds / "labels" / "labels.csv"
        if f.exists():
            return f
    raise FileNotFoundError("labels.csv not found")


def load_all():
    """Returns (main, stress) DataFrames: one row per image, joined to labels.
    Stress students are removed from `main` so there is no leakage."""
    labels = pd.read_csv(_find_labels_csv(), dtype=str, keep_default_na=False)

    frames, skipped = [], []
    for ds in sorted(DATA.glob("AI-for-Vision-Dataset*")):
        df, sk = _scan_images(ds / "images")
        df["dataset"] = ds.name
        frames.append(df)
        skipped += sk
    main = pd.concat(frames, ignore_index=True)
    n_dup = main.duplicated("fname").sum()
    main = main.drop_duplicates("fname")
    main = main.merge(labels, on="form_id", how="left", validate="m:1")

    stress_df, sk = _scan_images(DATA / "stress")
    skipped += sk
    stress = stress_df.merge(labels, on="form_id", how="left", validate="m:1")

    print(f"images: {len(main)} | duplicate filenames dropped: {n_dup} | "
          f"unparsed filenames: {len(skipped)} | "
          f"images without labels: {main['split'].isna().sum()}")
    main = main[~main["student"].isin(STRESS_IDS)].reset_index(drop=True)
    return main, stress


def truth(row):
    """Ground truth in the predict.py output format (invalid/blank -> None)."""
    sn = row["student_number"]
    sn = sn if (len(sn) == 8 and sn.isdigit()) else None
    SHIFT, WRAP = 0, False        # image question i is labelled in column q{i+SHIFT}
    def _lab(i):
        j = i + SHIFT
        if j > 120:
            if not WRAP: return None
            j -= 120
        v = row[f"q{j}"]
        return v if v in ANSWERS else None
    ans = {str(i): _lab(i) for i in range(1, 121)}
    return {"student_number": sn, "answers": ans,
            "name": row.get("name") or None,
            "module_code": row.get("module_code") or None,
            "seat_number": row.get("seat_number") or None,
            "date": row.get("date") or None}