"""python predict.py --input form.jpg --output prediction.json"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "Data")
FIELDS = ("name", "module_code", "seat_number", "date")


def empty():
    return {"student_number": None, "answers": {str(i): None for i in range(1, 121)},
            "name": None, "module_code": None, "seat_number": None, "date": None}


def run(path, debug_image=None):
    import cv2
    from robust import RobustRegistrar
    from bubbles import bubble_scores, decode, annotate, DEFAULT_PARAMS
    layout = json.load(open(os.path.join(DATA, "layout.json")))
    params = dict(DEFAULT_PARAMS)
    pp = os.path.join(DATA, "params.json")           # written by the evaluation notebook
    if os.path.exists(pp):
        params.update(json.load(open(pp)))
    template = cv2.imread(os.path.join(DATA, "template_canonical.png"))
    img = cv2.imread(path)
    if img is None:
        raise IOError("cannot read image")
    page, _ = RobustRegistrar(template).warp(img)
    A, D = bubble_scores(page, layout)
    sn, answers = decode(A, D, params)
    out = empty()
    out["student_number"], out["answers"] = sn, answers
    # Optional extension: fill name / module_code / seat_number / date with OCR here.
    if debug_image:
        cv2.imwrite(debug_image, annotate(page, layout, answers, sn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--debug_image", default=None)
    a = ap.parse_args()
    try:
        out = run(a.input, a.debug_image)
    except Exception as e:                           # always emit valid JSON
        print(f"predict.py: {e!r}", file=sys.stderr)
        out = empty()
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()