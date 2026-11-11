# S3231 MCQ answer-sheet reader

Reads scanned or photographed **S3231 multiple-choice answer sheets**: an 8-digit student number and 120 answers (A–E), and writes them as JSON. It is a classical computer-vision pipeline with no neural network. It runs on CPU only.

Project for **UCD COMP41790 Applied AI in Vision & Imaging** by <Your Name> (<student number>).

## Results (845 held-out test forms from unseen students)

| Method | Student no. | Per-question | Exact form | Time per form |
|---|---|---|---|---|
| **Ours** | **0.909** | **0.895** | **0.173** | 0.087 s |
| Contour warp only | 0.182 | 0.477 | 0.000 | 0.193 s |
| Plain argmax (no blank option) | 0.828 | 0.455 | 0.000 | 0.087 s |

On the 90-form held-out stress set: student number 0.256, per-question 0.849, exact form 0.044.
See `report/report.pdf` for the method, ablations, confusion matrix and error analysis.

## How it works

1. **Register** each page to a canonical template (contour warp, then SIFT + RANSAC homography, with retries).
2. **Layout** of rows and columns is learned once from registered training pages.
3. **Bubble reading** uses the red channel (so the pink print is nearly invisible) with background flattening and mean ink per bubble.
4. **Decision rule** picks the darkest option only if it passes an absolute and a relative threshold, otherwise `null`. Thresholds are tuned on training students only.
5. **Student-number block** gets a small local alignment search before reading digits.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python predict.py --input path/form.jpg --output path/prediction.json
```

Output (always valid JSON):

```json
{"student_number": "07365680", "answers": {"1": "E", "2": "A", "14": null, "...": "..."},
 "name": null, "module_code": null, "seat_number": null, "date": null}
```

A corrupt or missing image produces an all-`null` prediction instead of a crash. Header fields (name, module code, seat number, date) are not read and are always `null`. Typical time is about 1.4 s per form including start-up, CPU only.

## Repository layout

| Path | Content |
|---|---|
| `predict.py` | command-line entry point |
| `src/` | registration, layout learning, bubble reading, pipeline, data loader |
| `Data/layout.json`, `Data/template_canonical.png`, `Data/params.json` | the "model weights": learned layout, canonical template, tuned thresholds |
| `evaluation.ipynb` | reproducible experiments: student-wise split, baselines, ablations, confusion matrix, error analysis |
| `Data/AI-for-Vision-Dataset*`, `Data/stress/` | the datasets (see below) |
| `report/` | technical report (NeurIPS preprint format) and its figures |

## Dataset

The data was **created by us**: the sheets were filled in by students of the module and captured as flatbed scans and phone photos (conditions: scan, photo, hard). It is split **by student** into 1,249 training forms and 845 test forms, with a code check that no student appears in both. A separate **stress set** of 90 forms (participants p03, p04, p19) is kept in `Data/stress/` and is used only for the final stress evaluation.

> The forms contain handwriting and student numbers. Do not redistribute them outside the course. <edit this note to match your course rules>

## Reproduce the evaluation

Keep the dataset folders under `Data/` with these names, then open `evaluation.ipynb` and **Run All**. Cached intermediate files (`Data/pages`, `*.pkl`) are rebuilt automatically and are not stored in the repo.

## Limitations

Failures are concentrated: the worst 10 % of forms hold 56 % of missed marks, mostly very faint pencil on tinted paper. Student-number reading degrades strongly on the stress set. The stress forms were inspected once while diagnosing the student number (see the report's evaluation section).
