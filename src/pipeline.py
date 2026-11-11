"""Parallel batch runner: image -> registered page -> bubble scores (used by the evaluation notebook).
variant: "sift" (plain SIFT), "align" (contour warp), "robust" (SIFT + alignment check + fallbacks),
         "pages" (re-score pages already saved by a previous "robust" run: seconds instead of minutes)."""
import os, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
import align, register, robust
from bubbles import bubble_scores

_G = {}


def _init(layout, template_path, page_dir):
    cv2.setNumThreads(1)                       # one thread per worker, parallelism is across images
    _G["layout"], _G["page_dir"] = layout, page_dir
    t = cv2.imread(template_path)
    _G["reg"] = register.Registrar(t)
    _G["rreg"] = robust.RobustRegistrar(t)


def _page_path(fname):
    return os.path.join(_G["page_dir"], os.path.splitext(fname)[0] + ".jpg")


def _one(args):
    fname, path, variant = args
    layout = _G["layout"]
    t0 = time.perf_counter()
    try:
        if variant == "pages":
            page, info = cv2.imread(_page_path(fname)), {"method": "cached"}
        else:
            img = cv2.imread(path)
            if variant == "sift":              # plain SIFT registration (baseline)
                page, info = _G["reg"].warp(img)
            elif variant == "robust":          # SIFT + alignment check + fallbacks (ours)
                page, info = _G["rreg"].warp(img)
                if _G["page_dir"]:
                    cv2.imwrite(_page_path(fname), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:                              # baseline: contour-based warp only
                page, info = align.warp_sheet(img)
                page = cv2.resize(page, (layout["W"], layout["H"]))
        t1 = time.perf_counter()
        A, D = bubble_scores(page, layout)
        return dict(fname=fname, A=A, D=D, method=info.get("method", variant),
                    t_warp=t1 - t0, t_total=time.perf_counter() - t0)
    except Exception as e:                     # count as a failed form instead of crashing the run
        return dict(fname=fname, A=np.zeros((120, 5)), D=np.zeros((8, len(layout["digit_rows"]))),
                    method="error: " + repr(e)[:60], t_warp=np.nan, t_total=np.nan)


def process_df(df, variant, layout, template_path, n_jobs=None, limit=None, page_dir=None):
    if limit:
        df = df.head(limit)
    if page_dir:
        os.makedirs(page_dir, exist_ok=True)
    jobs = [(r.fname, r.path, variant) for r in df.itertuples()]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(layout, template_path, page_dir)) as ex:
        it = ex.map(_one, jobs, chunksize=2)
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(jobs), desc=variant)
        except Exception:
            pass
        recs = list(it)
    R = pd.DataFrame(recs)
    print(f"{variant}: {len(R)} forms in {time.perf_counter()-t0:.0f}s with {n_jobs} workers, "
          f"{R.method.str.startswith('error').sum()} errors")
    return R