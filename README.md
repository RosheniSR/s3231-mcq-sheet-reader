# S3231 MCQ answer-sheet reader (COMP41790 Vision Project)

Classical computer-vision pipeline (no neural network): page warp, SIFT + RANSAC registration to a
canonical template, a learned bubble layout, red-channel ink scoring and a tuned abstaining decision rule.
Runs on CPU only (about 1.4 s per form including start-up). No GPU is needed.

## Install
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Run
    python predict.py --input path/form.jpg --output path/prediction.json

The output is always valid JSON: student_number (8-digit string or null), answers {"1".."120": "A".."E" or null},
and name, module_code, seat_number, date (null; header fields are not read). A corrupt or missing image
gives an all-null prediction instead of a crash.

## Contents
- predict.py: command-line entry point
- src/: registration (align.py, register.py, robust.py), layout learning (grid.py), bubble reading (bubbles.py), pipeline.py, data helpers (s3231_data.py)
- Data/layout.json, Data/template_canonical.png, Data/params.json: the "model weights" (learned layout, canonical template, tuned thresholds)
- evaluation.ipynb: reproducible experiments (student-wise train/test split, baselines, ablations, confusion matrix, error analysis)
- report/: technical report (NeurIPS preprint format) and its figures

## Reproducing the evaluation
The datasets are not included. Put them under Data/ (AI-for-Vision-Dataset*, stress/) and run evaluation.ipynb top to bottom.
The stress set (p03, p04, p19) is only used for the final stress evaluation.
