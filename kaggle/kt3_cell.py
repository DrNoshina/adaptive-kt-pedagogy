# EdNet-KT3: the same model-selection rule as ASSISTments 2017 + significance vs HGAKT/AKT. ONE cell.
# Needs: Add Input -> kt-code (+ the output of the previous run of THIS notebook, if there was one), GPU T4 x2, Internet ON.
# Run with Save Version -> Save & Run All. It stops by itself before Kaggle's 12 h limit; run it again and it continues.
import os, sys, json, shutil, subprocess, tarfile, threading, time, zipfile
IN = os.environ.get("KT_INPUT", "/kaggle/input"); OUT = os.environ.get("KT_OUTPUT", "/kaggle/working"); K = os.environ.get("KT_ROOT", "/kaggle/temp/kt")
T0 = time.time(); LAST = [time.time()]; EVERY = float(os.environ.get("KT_SAVE_MINUTES", "15")) * 60
SOFT, HARD = float(os.environ.get("KT_SOFT_HOURS", "10.0")) * 3600, float(os.environ.get("KT_HARD_HOURS", "11.0")) * 3600
MIN_FREE_GB = float(os.environ.get("KT_MIN_FREE_GB", "12"))
CFG, SPLIT = "configs/ednet_kt3_trackB.yaml", "results/ednet_kt3_trackB/split_assignments.csv"
V3, SF, RES_REL = "results/ednet_kt3_trackB_v3", "results/ednet_kt3_trackB_samefeat", "results/kt3_final"
BL = os.environ.get("KT3_BL", "results/ednet_kt3_kaggle_baselines"); HG = os.environ.get("KT3_HG", "results/hgakt_kt3_trackB")
LAB = os.environ.get("KT_LAB", "0") == "1"   # lab machine: data already prepared, no Step D, no 3 GB state tar
RES, LOGS = f"{K}/{RES_REL}", f"{K}/logs/kt3final"
def ram():
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo")); kb = lambda key: float(info[key].split()[0])
        return (kb("MemTotal") - kb("MemAvailable")) / 1e6, kb("MemAvailable") / 1e6
    except Exception: return float("nan"), float("inf")
def say(text): LAST[0] = time.time(); print(f"[{(time.time() - T0) / 3600:5.2f} h | RAM {ram()[0]:4.1f} GB] {text}", flush=True)
def walk(root, depth=7):
    base = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        if cur.count("/") - base >= depth: dirs[:] = []
        yield cur, dirs, files

say("cell started: checking the GPU ...")
try: GPUS = int(os.environ.get("KT_GPUS") or len([l for l in subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines() if l.strip()]))
except Exception: GPUS = 0
if GPUS == 0: raise SystemExit("NO GPU in this session. Settings -> Accelerator -> GPU T4 x2. If GPU cannot be selected, the weekly Kaggle GPU quota is used up: wait for the weekly reset, then Save & Run All again (nothing is lost).")

say(f"GPUs found: {GPUS}")
# 1) code. kt-code is used from Add Input if attached; otherwise the public dataset is downloaded (Internet ON).
KT_DATASET = os.environ.get("KT_DATASET", "umarfarooq06/kt-code"); ROOTS = [IN]
def code_in(root):
    return any(("pyproject.toml" in files and os.path.basename(cur) == "kt_code") or "kt_kaggle_code.zip" in files for cur, dirs, files in walk(root))
if not os.path.exists(f"{K}/pyproject.toml") and not code_in(IN):
    say(f"kt-code is not attached -> attaching the public dataset {KT_DATASET} (needs Internet ON; a few minutes)")
    try: import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=False)
        try: import kagglehub
        except ImportError: raise SystemExit("kt-code is not attached and kagglehub is missing (Internet OFF?).\nFix: Settings -> Internet ON, or right panel -> Add Input -> search 'kt-code' (owner umarfarooq06) -> Add. Then Save & Run All again.")
    box = {}
    def fetch():
        try: box["path"] = kagglehub.dataset_download(KT_DATASET)
        except Exception as exc: box["error"] = exc
    worker = threading.Thread(target=fetch, daemon=True); worker.start(); waited = 0
    while worker.is_alive() and waited < float(os.environ.get("KT_ATTACH_MINUTES", "30")) * 60:
        worker.join(60); waited += 60
        if worker.is_alive(): say(f"  still attaching kt-code ... {waited // 60:.0f} min")
    if "path" in box: ROOTS.append(box["path"]); say(f"kt-code attached: {box['path']}")
    else:
        why = f"{type(box['error']).__name__}: {box['error']}" if "error" in box else "no answer after 30 min"
        raise SystemExit(f"kt-code could not be attached ({why}).\nFix: right panel -> Add Input -> search 'kt-code' (owner umarfarooq06) -> Add, then Save & Run All again.")
if not os.path.exists(f"{K}/pyproject.toml"):
    src = zp = None
    for cur, dirs, files in (item for root in ROOTS for item in walk(root)):
        if os.path.basename(cur) == "kt_code" and "pyproject.toml" in files: src = cur
        if "kt_kaggle_code.zip" in files: zp = os.path.join(cur, "kt_kaggle_code.zip")
    os.makedirs(os.path.dirname(K), exist_ok=True)
    if src: shutil.copytree(src, K, dirs_exist_ok=True)
    elif zp:
        zipfile.ZipFile(zp).extractall(K + "_zip"); shutil.copytree(K + "_zip/kt_code", K, dirs_exist_ok=True)
    else: raise SystemExit("Code not found: Add Input -> kt-code dataset")
os.makedirs(LOGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# 2) every saved KT3 state that is attached (old EdNet runs + earlier runs of this cell), oldest first
STATE_ZIPS = ("kt_resume.zip", "kt_results_small.zip", "kt3_final.zip"); STATE_TARS = ("kt_state.tar", "kt3_full.tar")
def stamp_of(kind, path):
    try:
        if kind == "tar":
            with tarfile.open(path) as a: return float(a.extractfile("logs/state_stamp.txt").read())
        if kind == "zip":
            with zipfile.ZipFile(path) as z: return float(z.read("logs/state_stamp.txt"))
        return float(open(f"{path}/logs/state_stamp.txt").read())
    except Exception: return 0.0
say("looking for saved results in the inputs (kt-code, earlier outputs) ...")
if not os.path.exists(f"{LOGS}/restore_done.txt"):
    found = []
    for root in [OUT] + ROOTS:
        for cur, dirs, files in walk(root):
            found += [("tar", os.path.join(cur, n)) for n in STATE_TARS if n in files] + [("zip", os.path.join(cur, n)) for n in STATE_ZIPS if n in files]
            if os.path.isdir(f"{cur}/results") and os.path.realpath(cur) != os.path.realpath(K) and \
               any(d.startswith(("ednet_kt3_trackB", "hgakt_kt3", "ednet_kt3_kaggle", "kt3_final")) for d in os.listdir(f"{cur}/results")): found.append(("folder", cur))
    unique = {}
    for kind, path in found: unique.setdefault(os.path.realpath(path), (kind, path))
    for kind, path in sorted(unique.values(), key=lambda item: stamp_of(*item)):
        say(f"restoring {kind}: {path}")
        try:
            if kind == "tar":
                with tarfile.open(path) as a: a.extractall(K)
            elif kind == "zip":
                with zipfile.ZipFile(path) as z: z.extractall(K)
            else:
                for name in ("data", "results", "logs"):
                    if os.path.isdir(f"{path}/{name}"): shutil.copytree(f"{path}/{name}", f"{K}/{name}", dirs_exist_ok=True, symlinks=True)
        except Exception as exc: say(f"  could not restore it ({type(exc).__name__}: {exc}); continuing")
    open(f"{LOGS}/restore_done.txt", "w").write(str(len(found)))
def resume_points():
    """Finished runs and unfinished runs with a checkpoint (= where training continues)."""
    finished, resume = [], []
    for folder in (V3, SF, BL, HG):
        for cur, dirs, files in os.walk(f"{K}/{folder}"):
            if os.path.basename(cur) in ("cache", "predictions"): dirs[:] = []; continue
            for name in files:
                if name.endswith("_metrics.json"):
                    try:
                        if "test" in json.load(open(os.path.join(cur, name))): finished.append(name[:-len("_metrics.json")])
                    except Exception: pass
                if name.endswith(".resume.pt"):
                    stem = name[:-len(".resume.pt")]; base = os.path.dirname(cur) if os.path.basename(cur) == "checkpoints" else cur
                    if os.path.exists(f"{base}/{stem}_metrics.json") and "test" in open(f"{base}/{stem}_metrics.json").read(): continue
                    epochs = "?"
                    for hist in (f"{base}/training_history/{stem}.csv", f"{base}/{stem}_history.csv"):
                        if os.path.exists(hist): epochs = max(0, sum(1 for line in open(hist) if line.strip()) - 1); break
                    resume.append(f"{stem} (continues after epoch {epochs})")
    return sorted(set(finished)), resume
done_runs, resume_runs = resume_points()
say(f"finished runs available: {len(done_runs)} -> {', '.join(done_runs) or 'none'}")
say("resume points: " + ("; ".join(resume_runs) if resume_runs else "none"))
for folder, link in ((V3, "v3"), (SF, "samefeat"), (BL, "baselines"), (HG, "hgakt")):
    os.makedirs(f"{K}/{folder}", exist_ok=True); os.makedirs(RES, exist_ok=True)
    if not os.path.lexists(f"{RES}/{link}"): os.symlink(os.path.relpath(f"{K}/{folder}", RES), f"{RES}/{link}")

# 3) helper scripts written into the code copy
STAGES_PY = r'''"""ASSISTments 2017 (cross-dataset): stages 'data' and 'table'. Run from the project root."""
import json, os, shutil, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

K = os.environ.get("KT_ROOT", os.getcwd()); IN = os.environ.get("KT_INPUT", "/kaggle/input"); OUT = os.environ.get("KT_OUTPUT", "/kaggle/working")
T0 = float(os.environ.get("KT_T0", time.time()))
NAME = "anonymized_full_release_competition_dataset"
RAW = f"{K}/data/raw/assist2017"
PROC = f"{K}/" + os.environ.get("KT_PROC", "data/processed/assist2017"); RES = f"{K}/" + os.environ.get("KT_RES", "results/assist2017")
CFG = f"{K}/" + os.environ.get("KT_CFG", "configs/assistments2017.yaml"); SPLIT = f"{K}/" + os.environ.get("KT_SPLIT", "results/assist2017/split_assignments.csv")
QUESTIONS = f"{RAW}/contents/contents/questions.csv"
PREFIX, DS_NAME = os.environ.get("KT_PREFIX", "assist2017"), os.environ.get("KT_DS_NAME", "ASSISTments 2017")
HF_REPO, HF_FILE = "Lucy9999/assist2017", f"Released Full Dataset/{NAME}.csv"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/Released%20Full%20Dataset/{NAME}.csv"
USTC_URL = f"http://base.ustc.edu.cn/data/ASSISTment/{NAME}.zip"          # EduData (USTC) mirror, used only if Hugging Face fails
PUBLISHED = {"rows": 942816, "learners": 1709, "problems": 3162, "skills": 102}   # numbers reported in the KT literature for this file
BATCH, EPOCHS, PATIENCE = int(os.environ.get("KT_A17_BATCH", "32")), int(os.environ.get("KT_A17_EPOCHS", "200")), int(os.environ.get("KT_A17_PATIENCE", "10"))
V3_SEEDS = (42, 52, 62)
ABLATIONS = [("profile,experience,timing,recency,engagement,source,elo,meta", "Step 0: answers only"),
             ("experience,timing,recency,engagement,source,elo,meta", "Step 1: + learner profile"),
             ("experience,timing,engagement,source,elo,meta", "Step 2: + recency"),
             ("experience,timing,engagement", "Step 3: + difficulty-adjusted mastery, skill tags"),
             ("", "Full V3: + experience, timing, hint behaviour"),
             ("mastery", "Without mastery score"), ("response_time", "Without response time"), ("recent_accuracy", "Without recent accuracy")]
MODELS = [("Proposed V3", "v3/v3_seed42"), ("HGAKT (base paper, our adaptation)", "hgakt/hgakt_seed42"), ("AKT", "baselines/akt_seed42"), ("SAKT", "baselines/sakt_seed42"),
          ("DKT", "baselines/dkt_lstm_seed42"), ("Transformer-KT", "baselines/transformer_kt_seed42"), ("SAINT+", "baselines/saint_plus_seed42"),
          ("Same inputs as V3 + XGBoost", "samefeat/xgb_f_seed42"), ("Same inputs as V3 + LSTM backbone (DKT-style)", "samefeat/dkt_f_seed42"),
          ("XGBoost", "baselines/xgboost_seed42"), ("Random Forest", "baselines/random_forest_seed42"), ("Logistic Regression", "baselines/logistic_regression_seed42")]
LEAVE_ONE_OUT = [("profile", "Without learner profile"), ("recency", "Without recency (same item / same skill before)"), ("experience,timing", "Without experience and timing"),
                 ("engagement", "Without hint (explanation) context"), ("elo,meta", "Without difficulty-adjusted mastery and skill tags"),
                 ("mastery", "Without mastery score (topic mastery + difficulty-adjusted)"), ("response_time", "Without response time"), ("recent_accuracy", "Without recent accuracy")]
METRICS = ["auc", "accuracy", "precision", "recall", "f1", "rmse"]


def say(text): print(f"[{(time.time() - T0) / 60:6.1f} min] {text}", flush=True)
def tag_of(drops):
    names = sorted(item for item in drops.split(",") if item)
    return "v3" if not names else "v3_no_" + "_".join(names)
def walk(root, depth=7):
    base = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        if cur.count("/") - base >= depth: dirs[:] = []
        yield cur, files


# ------------------------------------------------------------------ stage: data
def from_zip(zip_path, target):
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if os.path.basename(n) == NAME + ".csv"]
        if not names: return False
        with archive.open(names[0]) as src, open(target + ".part", "wb") as dst: shutil.copyfileobj(src, dst, 1 << 22)
    os.replace(target + ".part", target); return True

def download(url, target):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(target + ".part", "wb") as out:
        total, done, mark = int(response.headers.get("Content-Length") or 0), 0, 0
        while True:
            chunk = response.read(1 << 22)
            if not chunk: break
            out.write(chunk); done += len(chunk)
            if done - mark >= 100e6: mark = done; say(f"  downloaded {done / 1e6:.0f} of {total / 1e6:.0f} MB")
    if total and done != total: raise IOError(f"download incomplete: {done} of {total} bytes")
    os.replace(target + ".part", target)

def find_csv():
    """The official file, from (1) an attached Kaggle input, (2) the public Hugging Face copy, (3) the USTC mirror."""
    target = f"{RAW}/{NAME}.csv"; os.makedirs(RAW, exist_ok=True)
    if os.path.exists(target): return target, "downloaded earlier in this session"
    for root in (IN, RAW):
        for cur, files in walk(root):
            if NAME + ".csv" in files: return os.path.join(cur, NAME + ".csv"), f"attached input ({cur})"
            for name in files:
                if name.endswith(".zip") and (NAME in name or "assist" in name.lower()) and "kt_" not in name:
                    try:
                        if from_zip(os.path.join(cur, name), target): return target, f"attached zip ({os.path.join(cur, name)})"
                    except Exception: pass
    def hub():
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset"), target)
    def ustc():
        download(USTC_URL, f"{RAW}/{NAME}.zip")
        if not from_zip(f"{RAW}/{NAME}.zip", target): raise IOError("the csv is not inside the zip")
    errors = []
    for label, action in ((f"Hugging Face dataset {HF_REPO}", lambda: download(HF_URL, target)), (f"Hugging Face dataset {HF_REPO} (hub library)", hub), ("USTC EduData mirror", ustc)):
        try:
            say(f"downloading ASSISTments 2017 from: {label}"); action(); return target, label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:150]}"); say(f"  did not work ({errors[-1]})")
    raise SystemExit("ASSISTments 2017 could not be downloaded. Check: notebook Settings -> Internet = ON, then run again.\n  " + "\n  ".join(errors))

def pick(columns, *names, required=True):
    lower = {str(c).lower(): c for c in columns}
    for name in names:
        if name.lower() in lower: return lower[name.lower()]
    if required: raise SystemExit(f"Column '{names[0]}' is not in the file, so this is not the ASSISTments 2017 file. Columns found: {list(columns)[:30]}")
    return None

def as_id(series):
    number = pd.to_numeric(series, errors="coerce")
    if number.notna().all() and (number % 1 == 0).all(): return number.astype("int64").astype(str)
    return series.astype(str).str.strip()

def convert(path):
    """Same row rules as the usual KT preprocessing of this file (pyKT): keep every logged action with a learner,
    problem, skill, start time and a 0/1 answer; order by learner, start time, file order."""
    encoding = "utf-8"
    try: head = pd.read_csv(path, nrows=5, encoding=encoding)
    except UnicodeDecodeError: encoding = "latin-1"; head = pd.read_csv(path, nrows=5, encoding=encoding)
    cols = list(head.columns)
    c = dict(user=pick(cols, "studentId", "ITEST_id", "student_id", "user_id"), item=pick(cols, "problemId", "problem_id"), skill=pick(cols, "skill", "skill_id", "skill_name"),
             correct=pick(cols, "correct"), start=pick(cols, "startTime", "start_time"), taken=pick(cols, "timeTaken", "time_taken"),
             hint=pick(cols, "hint", required=False), attempt=pick(cols, "attemptCount", "attempt_count", required=False))
    df = pd.read_csv(path, usecols=[v for v in c.values() if v], low_memory=False, encoding=encoding)
    raw_rows = len(df); df["_row"] = np.arange(raw_rows)
    for key in ("correct", "start", "taken"): df[c[key]] = pd.to_numeric(df[c[key]], errors="coerce")
    df = df.dropna(subset=[c["user"], c["item"], c["skill"], c["start"]])
    df = df[df[c["correct"]].isin([0, 1])]
    if df.empty: raise SystemExit("No usable rows in the file (learner, problem, skill, startTime, correct).")
    df["user_id"], df["item_id"], df["skill_id"] = as_id(df[c["user"]]), as_id(df[c["item"]]), df[c["skill"]].astype(str).str.strip()
    df = df.sort_values(["user_id", c["start"], "_row"], kind="mergesort").reset_index(drop=True)
    users = df["user_id"].to_numpy()
    taken = df[c["taken"]].fillna(0.0).clip(lower=0.0).to_numpy(np.float64)                    # seconds spent on this action
    start_ms = np.rint(df[c["start"]].to_numpy(np.float64) * 1000.0).astype(np.int64)            # moment the action became available
    end_ms = np.rint((df[c["start"]].to_numpy(np.float64) + taken) * 1000.0).astype(np.int64)   # moment the action ended
    rank = df.groupby("user_id", sort=False).cumcount().to_numpy(np.int64)
    # timestamp = end of the action, made strictly increasing inside a learner so the row order above is kept exactly.
    # response time = timestamp - start, so "timestamp - response time" is always the true start time (known before the answer).
    timestamp = pd.Series(end_ms - rank).groupby(users, sort=False).cummax().to_numpy(np.int64) + rank
    response_ms = (timestamp - start_ms).astype(np.float64)
    hint = (pd.to_numeric(df[c["hint"]], errors="coerce").fillna(0) > 0).astype(np.int64) if c["hint"] else pd.Series(np.zeros(len(df), dtype=np.int64))
    hint_ms = hint * np.clip(taken, 0.0, 1200.0) * 1000.0
    # hint requests are ASSISTments' "explanation" behaviour. Only hints of EARLIER rows are counted (cumulative sum minus the row itself).
    hint_before = hint.groupby(users, sort=False).cumsum() - hint
    hint_ms_before = hint_ms.groupby(users, sort=False).cumsum() - hint_ms
    attempts = pd.to_numeric(df[c["attempt"]], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(np.float64) if c["attempt"] else np.ones(len(df))
    inter = pd.DataFrame({"user_id": df["user_id"].astype(str).to_numpy(), "timestamp": timestamp, "item_id": df["item_id"].astype(str).to_numpy(), "skill_id": df["skill_id"].astype(str).to_numpy(),
                          "correct": df[c["correct"]].to_numpy(np.int8), "response_time_ms": response_ms, "attempt_count": attempts,
                          "explanation_count_prior": hint_before.to_numpy(np.float64), "explanation_time_ms_prior": hint_ms_before.to_numpy(np.float64),
                          "lecture_count_prior": 0.0, "lecture_time_ms_prior": 0.0, "review_count_prior": 0.0, "adaptive_offer_count_prior": 0.0})
    pairs = inter.groupby(["item_id", "skill_id"]).size().reset_index(name="n").sort_values(["item_id", "n", "skill_id"], ascending=[True, False, True])
    code = {skill: index + 1 for index, skill in enumerate(sorted(inter["skill_id"].unique()))}
    tags = pairs.groupby("item_id", sort=True)["skill_id"].apply(lambda s: ";".join(str(code[v]) for v in list(s)[:7]))
    questions = pd.DataFrame({"question_id": tags.index.astype(str), "tags": tags.to_numpy(), "part": 0})
    is_hint = hint.to_numpy() == 1
    stats = {"file_rows": int(raw_rows), "rows": int(len(inter)), "learners": int(inter["user_id"].nunique()), "problems": int(inter["item_id"].nunique()), "skills": int(inter["skill_id"].nunique()),
             "correct_rate": float(inter["correct"].mean()), "hint_request_share": float(is_hint.mean()), "correct_rate_on_hint_rows": float(inter["correct"].to_numpy()[is_hint].mean()) if is_hint.any() else None,
             "timestamps_moved_to_keep_order_share": float(np.mean(timestamp != end_ms)), "problems_with_more_than_one_skill": int((pairs.groupby("item_id").size() > 1).sum()),
             "median_actions_per_learner": float(inter.groupby("user_id").size().median()), "columns_used": c}
    return inter, questions, stats

def stage_data():
    from adaptive_kt.data.features import build_causal_features
    from adaptive_kt.data.splits import split_assignment_frame, split_by_user
    from adaptive_kt.utils.tableio import read_collection, write_table
    import yaml
    os.makedirs(RES, exist_ok=True)
    stats_path = f"{RES}/data_stats.json"; done = f"{PROC}/interactions/_DONE"
    if all(os.path.exists(p) for p in (done, f"{PROC}/features/_DONE", SPLIT, CFG, QUESTIONS, stats_path)):
        stats = json.load(open(stats_path)); say(f"data already prepared: {stats['rows']:,} rows, {stats['learners']:,} learners (source: {stats['source']})"); return
    path, source = find_csv()
    say(f"file: {path} ({os.path.getsize(path) / 1e6:.0f} MB) | source: {source}")
    inter, questions, stats = convert(path); stats["source"] = source
    say(f"rows in file {stats['file_rows']:,} -> used {stats['rows']:,} | learners {stats['learners']:,} | problems {stats['problems']:,} | skills {stats['skills']:,} | correct {stats['correct_rate']:.4f}")
    same = {k: stats[k] == v for k, v in PUBLISHED.items()}; stats["matches_published_numbers"] = same
    say("check against the published numbers (942,816 rows, 1,709 learners, 3,162 problems, 102 skills): " + ("ALL MATCH" if all(same.values()) else "DIFFERENT -> " + ", ".join(f"{k}: file {stats[k]:,} vs published {PUBLISHED[k]:,}" for k, ok in same.items() if not ok)))
    hint_rate = stats["correct_rate_on_hint_rows"]
    say(f"hint requests: {stats['hint_request_share']:.3f} of rows (correct rate on those rows: {'n/a' if hint_rate is None else f'{hint_rate:.3f}'}) | rows whose time was shifted to keep the order: {stats['timestamps_moved_to_keep_order_share']:.4f}")
    for folder in ("interactions", "features"): shutil.rmtree(f"{PROC}/{folder}", ignore_errors=True)
    write_table(inter, Path(PROC) / "interactions" / "part-00000"); open(done, "w").write("ok")
    os.makedirs(os.path.dirname(QUESTIONS), exist_ok=True); questions.to_csv(QUESTIONS, index=False)

    split = split_by_user(inter, 0.7, 0.1, 0.2, seed=42)           # same function and seed as the EdNet-KT3 split
    assignment = split_assignment_frame(split)
    if os.path.exists(SPLIT):
        old = pd.read_csv(SPLIT, dtype=str)
        if dict(zip(old["user_id"], old["split"])) != dict(zip(assignment["user_id"], assignment["split"])):
            raise SystemExit(f"A different split file already exists ({SPLIT}). Nothing was overwritten.")
    else: assignment.to_csv(SPLIT, index=False)
    stats["split"] = {name: {"learners": int(part["user_id"].nunique()), "interactions": int(len(part)), "correct_rate": float(part["correct"].mean())} for name, part in split.items()}
    for name, info in stats["split"].items(): say(f"split {name:10s}: learners {info['learners']:5,} | interactions {info['interactions']:8,} | correct {info['correct_rate']:.4f}")
    del split

    say("building the feature table for Logistic Regression / Random Forest / XGBoost ...")
    features = build_causal_features(inter, recent_window=10, min_history=5, max_response_time_seconds=1200)
    write_table(features, Path(PROC) / "features" / "part-00000"); open(f"{PROC}/features/_DONE", "w").write("ok")
    stats["feature_rows"] = int(len(features))

    cfg = yaml.safe_load(open(f"{K}/configs/ednet_kt3_trackB.yaml"))   # every other setting stays exactly as on EdNet-KT3
    cfg["dataset"].update({"name": "assistments_2017", "raw_root": "data/raw/assist2017", "processed_root": "data/processed/assist2017", "feature_set": "ednet_enriched"})
    cfg["sequence"].update({"batch_size": BATCH, "akt_batch_size": BATCH, "epochs": EPOCHS, "patience": PATIENCE})
    cfg["output"] = {"root": "results/assist2017"}
    yaml.safe_dump(cfg, open(CFG, "w"), sort_keys=False)
    stats["settings"] = {"batch_size": BATCH, "max_epochs": EPOCHS, "patience": PATIENCE, "everything_else": "same as configs/ednet_kt3_trackB.yaml"}
    json.dump(stats, open(stats_path, "w"), indent=2, default=str)
    say(f"data ready | feature rows {len(features):,} | settings: batch {BATCH}, max epochs {EPOCHS}, patience {PATIENCE}, everything else as on EdNet-KT3")


# ------------------------------------------------------------------ stage: table
def read_metrics(relative):
    try: data = json.load(open(f"{RES}/{relative}_metrics.json"))
    except (OSError, ValueError): return None
    return data if "test" in data else None

def auc_fast(y, p):
    from scipy.stats import rankdata
    ranks = rankdata(p); positives = int(y.sum()); negatives = len(y) - positives
    if positives == 0 or negatives == 0: return float("nan")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))

def test_targets():
    """Learner of every test target, in the order the models write their predictions (learner, then time; first 5 answers of a learner are not targets)."""
    from adaptive_kt.utils.tableio import read_collection
    inter = read_collection(Path(PROC) / "interactions"); inter["user_id"] = inter["user_id"].astype(str)
    assignment = pd.read_csv(SPLIT, dtype=str)
    users = set(assignment.loc[assignment["split"].eq("test"), "user_id"])
    test = inter[inter["user_id"].isin(users)].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    position = test.groupby("user_id", sort=False).cumcount().to_numpy()
    keep = position >= 5
    return pd.factorize(test["user_id"])[0][keep], test["correct"].to_numpy(np.uint8)[keep]

def bootstrap(reference, others):
    """Paired bootstrap over test learners: whole learners are drawn with replacement (as weights), the same draws for every model.
    Returns AUC(reference) - AUC(other) with 95% CI and two-sided p-value."""
    learner, y_ref = test_targets()
    def load(relative):
        folder, stem = relative.split("/")
        path = f"{RES}/{folder}/predictions/{stem}_test.npz"
        if not os.path.exists(path): return None
        data = np.load(path); y = data["y_true"].astype(np.uint8)
        if len(y) != len(y_ref) or not np.array_equal(y, y_ref): say(f"  {relative}: different test targets -> left out"); return None
        return data["y_prob"].astype(np.float64)
    ref = load(reference[1]); preds = {label: p for label, rel in others if (p := load(rel)) is not None}
    if ref is None or not preds: return None
    n_learners = int(learner.max()) + 1; y = y_ref.astype(np.float64)
    def prepare(p):
        order = np.argsort(p, kind="mergesort"); sorted_p = p[order]
        group = np.r_[0, np.cumsum(sorted_p[1:] != sorted_p[:-1])]
        return order, group, int(group[-1]) + 1, y[order], learner[order]
    def weighted_auc(prep, draws):
        order, group, size, ys, ls = prep; w = draws[ls]
        pos = np.bincount(group, weights=w * ys, minlength=size); neg = np.bincount(group, weights=w * (1.0 - ys), minlength=size)
        return float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))
    ones = np.ones(n_learners); ref_prep = prepare(ref); preps = {label: prepare(p) for label, p in preds.items()}
    ref_auc = weighted_auc(ref_prep, ones); aucs = {label: weighted_auc(pr, ones) for label, pr in preps.items()}
    rounds = int(os.environ.get("KT_A17_BOOTSTRAP", "1000")); rng = np.random.default_rng(12345); diffs = {label: np.empty(rounds) for label in preds}
    for b in range(rounds):
        draws = np.bincount(rng.integers(0, n_learners, n_learners), minlength=n_learners).astype(np.float64)
        base = weighted_auc(ref_prep, draws)
        for label, pr in preps.items(): diffs[label][b] = base - weighted_auc(pr, draws)
        if (b + 1) % 250 == 0: say(f"  bootstrap {reference[0]}: {b + 1}/{rounds}")
    out = []
    for label in preds:
        d = diffs[label][np.isfinite(diffs[label])]
        out.append({"compared_with": label, f"auc_{reference[0].split()[0].lower()}": ref_auc, "other_auc": aucs[label], "auc_difference": ref_auc - aucs[label],
                    "ci95_low": float(np.quantile(d, 0.025)), "ci95_high": float(np.quantile(d, 0.975)),
                    "p_value": float(min(1.0, max(2.0 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / len(d)))), "test_learners": n_learners, "resamples": int(len(d))})
    return pd.DataFrame(out)

BASELINES = [("HGAKT (base paper, our adaptation)", "hgakt/hgakt"), ("AKT", "baselines/akt"), ("DKT", "baselines/dkt_lstm"), ("SAKT", "baselines/sakt"),
             ("Transformer-KT", "baselines/transformer_kt"), ("SAINT+", "baselines/saint_plus"), ("Same inputs as Proposed + XGBoost", "samefeat/xgb_f"),
             ("XGBoost", "baselines/xgboost"), ("Random Forest", "baselines/random_forest"), ("Logistic Regression", "baselines/logistic_regression")]
GROUP_LABEL = {"profile": "learner profile", "recency": "recency (same item / skill before)", "experience,timing": "experience + timing", "engagement": "hint (explanation) context",
               "elo,meta": "difficulty-adjusted mastery + skill tags", "source": "source / platform", "response_time": "response time", "recent_accuracy": "recent accuracy", "mastery": "mastery score (topic + difficulty-adjusted)"}
SEEDS = (42, 52, 62)

def suffix(drops): return "" if not drops else "_no_" + "_".join(sorted(x for x in drops.split(",") if x))
def merge(*parts): return ",".join(sorted({x for p in parts for x in p.split(",") if x}))

def stage_checks():
    """Leakage test on the REAL file: change a row's own answer, hint flag, time and attempts, then every later row.
    The row's model inputs (context, learner profile, difficulty-adjusted mastery) must not change."""
    import tempfile
    from adaptive_kt.models.sequence import SequenceVocabulary, _causal_behavior_matrix
    from adaptive_kt.models.sequence_v2 import build_context
    from adaptive_kt.models.sequence_v3 import fit_item_easiness, item_lookup, load_question_meta, mastery_features
    from adaptive_kt.utils.tableio import read_collection
    import yaml
    out_path = f"{RES}/leakage_check.json"
    if PREFIX != "assist2017": say("raw-file leakage test is for ASSISTments; on EdNet-KT3 the same checks ran in the V2/V3 smoke tests"); return
    if os.path.exists(out_path): say("leakage test on the real file: " + json.load(open(out_path))["status"] + " (done earlier)"); return
    cfg = yaml.safe_load(open(CFG)); assignment = pd.read_csv(SPLIT, dtype=str)
    inter = read_collection(Path(PROC) / "interactions"); inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["user_id"].isin(set(assignment.loc[assignment["split"].eq("train"), "user_id"]))].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    item_vocab = SequenceVocabulary.fit(train["item_id"]); items = train["item_id"].map(item_vocab.mapping).fillna(1).to_numpy(np.int32)
    first = np.r_[True, train["user_id"].to_numpy()[1:] != train["user_id"].to_numpy()[:-1]]
    full, _, _ = fit_item_easiness(items, train["correct"].to_numpy(np.uint8), first, item_vocab.size)
    part_of, tags_of, _ = load_question_meta(cfg["dataset"]["raw_root"]); item_part, item_tags = item_lookup(item_vocab, part_of, tags_of)
    path, _ = find_csv(); stats = json.load(open(f"{RES}/data_stats.json")); col = stats["columns_used"]
    raw = pd.read_csv(path, usecols=[v for v in col.values() if v], low_memory=False); raw["_file"] = np.arange(len(raw))
    raw["_uid"] = as_id(raw[col["user"]])
    def inputs(frame):
        with tempfile.TemporaryDirectory() as tmp:
            frame.drop(columns=["_file", "_uid"]).to_csv(f"{tmp}/x.csv", index=False); one, _, _ = convert(f"{tmp}/x.csv")
        one = one.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
        codes = one["item_id"].map(item_vocab.mapping).fillna(1).to_numpy(np.int32); start = np.r_[True, np.zeros(len(one) - 1, bool)]
        mastery = mastery_features(codes, one["correct"].to_numpy(np.uint8), start, full[codes], item_part, item_tags).astype(np.float32)
        return np.hstack([build_context(one).astype(np.float32), _causal_behavior_matrix(one)[0][:, :12], mastery])
    rng = np.random.default_rng(3); worst_own = worst_future = 0.0; reached = checked = 0
    held = sorted(set(assignment.loc[assignment["split"].ne("train"), "user_id"]))
    for uid in rng.choice(held, min(30, len(held)), replace=False):
        mine = raw[raw["_uid"] == uid].copy()
        ok = mine.dropna(subset=[col["user"], col["item"], col["skill"], col["start"]]); ok = ok[pd.to_numeric(ok[col["correct"]], errors="coerce").isin([0, 1])]
        order = ok.assign(_s=pd.to_numeric(ok[col["start"]], errors="coerce")).sort_values(["_s", "_file"], kind="mergesort")["_file"].to_numpy()
        if len(order) < 3: continue
        base = inputs(mine)
        for pos in rng.choice(len(order), 3, replace=False):
            row = mine.index[mine["_file"] == order[pos]][0]; own = mine.copy()
            own.loc[row, col["correct"]] = 1 - own.loc[row, col["correct"]]; own.loc[row, col["taken"]] = float(own.loc[row, col["taken"]]) + 37.0
            if col.get("hint"): own.loc[row, col["hint"]] = 1 - own.loc[row, col["hint"]]
            if col.get("attempt"): own.loc[row, col["attempt"]] = own.loc[row, col["attempt"]] + 3
            x = inputs(own); worst_own = max(worst_own, float(np.abs(x[pos] - base[pos]).max())); checked += 1
            if pos + 1 < len(order): reached += int(np.abs(x[pos + 1:] - base[pos + 1:]).max() > 0)
            later = mine.index[mine["_file"].isin(order[pos + 1:])]
            if len(later):
                fut = mine.copy(); fut.loc[later, col["correct"]] = 1 - fut.loc[later, col["correct"]]; fut.loc[later, col["taken"]] = fut.loc[later, col["taken"]].astype(float) * 0.5
                if col.get("hint"): fut.loc[later, col["hint"]] = 1 - fut.loc[later, col["hint"]]
                worst_future = max(worst_future, float(np.abs(inputs(fut)[pos] - base[pos]).max()))
    status = "PASS" if worst_own == 0 and worst_future == 0 and checked else "FAIL"
    result = {"status": status, "rows_checked": checked, "learners": int(min(30, len(held))), "max_input_change_from_own_answer": worst_own,
              "max_input_change_from_later_rows": worst_future, "change_reached_later_rows": reached}
    json.dump(result, open(out_path, "w"), indent=2)
    say(f"leakage test on the real file: {status} | {checked} rows | input change from own answer/hint/time/attempts = {worst_own:.1e}, from later rows = {worst_future:.1e}")

def history_of(stem, seed):
    folder, prefix = stem.split("/")
    for path in (f"{RES}/{folder}/training_history/{prefix}_seed{seed}.csv", f"{RES}/{folder}/{prefix}_seed{seed}_history.csv"):
        if os.path.exists(path):
            try: return pd.read_csv(path)
            except Exception: return None
    return None

def stage_table():
    import yaml
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60); made = []
    def show(title, frame, name):
        frame.to_csv(f"{RES}/{name}", index=False); shutil.copy(f"{RES}/{name}", f"{OUT}/{name}"); made.append(name)
        print(f"\n===== {title} =====\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    sel = json.load(open(f"{RES}/selection.json")); FINAL = sel["final_drops"]
    final_stem = f"{sel['final_folder']}/{sel['final_prefix']}{suffix(FINAL)}"
    # 1) how the final model was chosen (validation only)
    rows = [{"step": "backbone", "option": "attention (V3)", "validation_auc": sel["validation_auc_attention"], "result": "chosen" if sel["backbone"] == "attention" else "-"},
            {"step": "backbone", "option": "LSTM", "validation_auc": sel["validation_auc_lstm"], "result": "chosen" if sel["backbone"] == "lstm" else "-"},
            {"step": "inputs", "option": "all inputs", "validation_auc": sel["validation_auc_full"], "result": "-" if FINAL else "chosen"}]
    for g, v in sel["validation_auc_without_group"].items():
        helped = v < sel["validation_auc_full"]
        rows.append({"step": "inputs", "option": f"without {GROUP_LABEL[g]}", "validation_auc": v,
                     "result": "helped -> kept" if helped else ("did not help -> removed" if g in (sel["groups_removed"] or []) else "did not help alone, but removing together was worse -> kept")})
    if sel.get("validation_auc_pruned") is not None:
        rows.append({"step": "inputs", "option": "all inputs minus the ones that did not help", "validation_auc": sel["validation_auc_pruned"], "result": "chosen" if FINAL else "worse -> not used"})
    show("How the final model was chosen (VALIDATION AUC only, seed 42; test not used)", pd.DataFrame(rows), f"{PREFIX}_selection.csv")
    # 2) main table, mean of 3 seeds
    def summary(label, stem):
        ms = [m for s in SEEDS if (m := read_metrics(f"{stem}_seed{s}"))]
        if not ms: return None
        row = {"model": label, "seeds": len(ms), "val_auc": np.mean([m["validation"]["auc"] for m in ms]), "test_auc": np.mean([m["test"]["auc"] for m in ms]),
               "test_auc_std": np.std([m["test"]["auc"] for m in ms], ddof=1) if len(ms) > 1 else float("nan")}
        row.update({k: np.mean([m["test"][k] for m in ms]) for k in METRICS if k != "auc"}); return row
    main = [r for r in [summary("Proposed (final)", final_stem)] + [summary(l, s) for l, s in BASELINES] if r]
    main = pd.DataFrame(main).sort_values("test_auc", ascending=False).reset_index(drop=True)
    show(f"{DS_NAME}, TEST, mean of seeds 42/52/62 (same split, same targets for every model)", main, f"{PREFIX}_main_table.csv")
    # 3) where the final model beats each model
    sig = bootstrap(("Proposed", f"{final_stem}_seed42"), [(l, f"{s}_seed42") for l, s in BASELINES])
    fin = main[main["model"].eq("Proposed (final)")].iloc[0]; beats = []
    for _, other in main[~main["model"].eq("Proposed (final)")].iterrows():
        better = [k for k in METRICS if (fin[k if k != "auc" else "test_auc"] < other[k if k != "auc" else "test_auc"] if k == "rmse" else fin[k if k != "auc" else "test_auc"] > other[k if k != "auc" else "test_auc"])]
        s = sig[sig["compared_with"].eq(other["model"])] if sig is not None else pd.DataFrame()
        beats.append({"compared_with": other["model"], "auc_gain": fin["test_auc"] - other["test_auc"],
                      "significant_seed42": ("yes" if s.iloc[0]["ci95_low"] > 0 else ("NO, worse" if s.iloc[0]["ci95_high"] < 0 else "no")) if len(s) else "n/a",
                      "p_value": s.iloc[0]["p_value"] if len(s) else float("nan"), "better_on": ", ".join(better) or "-", "not_better_on": ", ".join(k for k in METRICS if k not in better) or "-"})
    show("Where the final model wins (AUC gain = mean of 3 seeds; significance = paired bootstrap over test learners, seed 42)", pd.DataFrame(beats), f"{PREFIX}_beats.csv")
    # 4) ablation of the FINAL model
    kept = sel["kept_groups"] + [g for g in ("mastery",) if g not in sel["groups_removed"]]
    variants = [(GROUP_LABEL[g], f"{sel['final_folder']}/{sel['final_prefix']}{suffix(merge(FINAL, g))}_seed42") for g in kept]
    full = read_metrics(f"{final_stem}_seed42"); abl = bootstrap(("Final", f"{final_stem}_seed42"), variants)
    rows = []
    for label, rel in variants:
        m = read_metrics(rel)
        if not (m and full): continue
        s = abl[abl["compared_with"].eq(label)] if abl is not None else pd.DataFrame()
        low, high = (s.iloc[0]["ci95_low"], s.iloc[0]["ci95_high"]) if len(s) else (float("nan"), float("nan"))
        rows.append({"removed": label, "val_auc": m["validation"]["auc"], "test_auc": m["test"]["auc"], "auc_drop_when_removed": full["test"]["auc"] - m["test"]["auc"],
                     "ci95_low": low, "ci95_high": high, "p_value": s.iloc[0]["p_value"] if len(s) else float("nan"),
                     "verdict": "helps (significant)" if low > 0 else ("hurts (significant)" if high < 0 else "no significant effect")})
    if rows: show(f"Ablation of the final model (full test AUC {full['test']['auc']:.4f}; drop > 0 means the part helps)", pd.DataFrame(rows), f"{PREFIX}_ablation_final.csv")
    # 5) training check
    max_epochs = int(yaml.safe_load(open(CFG))["sequence"]["epochs"]); rows = []
    for label, stem in [("Proposed (final)", final_stem)] + BASELINES:
        for seed in SEEDS:
            m = read_metrics(f"{stem}_seed{seed}")
            if not m: continue
            h = history_of(stem, seed); ran = int(len(h)) if h is not None else None; tree = stem.split("/")[1] in ("xgb_f", "xgboost", "random_forest", "logistic_regression")
            rows.append({"model": label, "seed": seed, "best_epoch": "-" if tree or m.get("best_epoch") is None else int(m["best_epoch"]),
                         "epochs_run": ran if ran is not None else ("- (tree/linear model)" if tree else "history missing"),
                         "stopped_by_early_stopping": ("yes" if ran < max_epochs else "NO, hit the epoch limit") if ran is not None and not tree else "-",
                         "val_auc": m["validation"]["auc"], "test_auc": m["test"]["auc"], "val_test_gap": abs(m["validation"]["auc"] - m["test"]["auc"])})
    show(f"Training check (max {max_epochs} epochs, early stopping on validation AUC, best epoch restored)", pd.DataFrame(rows), f"{PREFIX}_training_check.csv")
    leak = json.load(open(f"{RES}/leakage_check.json")) if os.path.exists(f"{RES}/leakage_check.json") else {}
    log_dir = f"{K}/" + os.environ.get("KT_LOGS", "logs/assist17")
    logs = " ".join(open(os.path.join(log_dir, f), errors="ignore").read() for f in (os.listdir(log_dir) if os.path.isdir(log_dir) else []) if f.endswith(".log"))
    print(f"\nLeakage: real-file input test = {leak.get('status', 'not run')} ({leak.get('rows_checked', 0)} rows) | model-level test (own answer cannot move its own prediction): "
          f"{logs.count('[leakage-model] PASS')} runs PASS, {logs.count('[leakage-model] FAIL')} FAIL", flush=True)
    print("Saved: " + ", ".join(made), flush=True)


if __name__ == "__main__":
    {"data": stage_data, "table": stage_table, "checks": stage_checks}[sys.argv[1]]()
'''
open(f"{K}/scripts/kt_a17.py", "w").write(STAGES_PY)
SAMEFEAT_PY = r'''#!/usr/bin/env python3
"""Same-information baselines for Proposed V3 (answers the question "is the gain only the features?").

  xgb_f : XGBoost on exactly the per-target inputs V3 gets (learner profile, pre-response context,
          difficulty-adjusted mastery incl. item easiness, source/platform, part). No sequence model.
  dkt_f : V3 with its attention backbone (history Transformer + cross-attention) replaced by one LSTM
          (DKT-style). Same inputs, same prediction head, same training settings.
Same split, same targets, same threshold rule (validation F1) as every other model. Test is scored once.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml
from torch import nn

import run_trackB_proposed_v3 as v3r          # importing only defines functions (and routes V3 batches through the trainer)
import adaptive_kt.training.kt_trainer as kt_trainer
from adaptive_kt.evaluation.metrics import binary_metrics, select_f1_threshold
from adaptive_kt.models.behavior_aware_kt_v2 import PROFILE_KEEP
from adaptive_kt.models.behavior_aware_kt_v3 import BehaviorAwareKTV3
from adaptive_kt.models.sequence import SequenceVocabulary
from adaptive_kt.models.sequence_v2 import fit_category_vocab
from adaptive_kt.models.sequence_v3 import KTSequenceDatasetV3, fit_item_easiness, item_lookup, load_question_meta
from adaptive_kt.utils.reproducibility import seed_everything, torch_device


class LSTMHistory(nn.Module):
    """Drop-in for V3's history Transformer: causal by construction (output t sees history 0..t)."""
    def __init__(self, dim: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        return self.lstm(x)[0]


class LastState(nn.Module):
    """Drop-in for V3's cross-attention: target t+1 reads the LSTM state after interaction t."""
    def forward(self, query, key, value, **_):
        return value, None


def one_hot(codes, size):
    out = np.zeros((len(codes), max(int(size), 1)), dtype=np.float32)
    out[np.arange(len(codes)), np.clip(codes.astype(np.int64), 0, max(int(size), 1) - 1)] = 1.0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", required=True, help="the V3 cache folder of this dataset (reused, never rewritten differently)")
    p.add_argument("--model", choices=["xgb_f", "dkt_f"], required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop", default="", help="input groups removed (dkt_f only), same names as the V3 runner")
    args = p.parse_args()
    drops = v3r.parse_drops(args.drop)
    if drops and args.model != "dkt_f": raise SystemExit("--drop is only for dkt_f")
    name = f"{args.model}{'_no_' + '_'.join(drops) if drops else ''}_seed{args.seed}"
    cfg = yaml.safe_load(open(args.config)); seq = cfg["sequence"]
    seed_everything(args.seed); device = torch_device(); started = time.time()
    out = Path(args.out); (out / "predictions").mkdir(parents=True, exist_ok=True); cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    split = v3r.load_split(Path(cfg["dataset"]["processed_root"]), Path(args.split), 0)
    skill_vocab, item_vocab = SequenceVocabulary.fit(split["train"]["skill_id"]), SequenceVocabulary.fit(split["train"]["item_id"])
    source_vocab, platform_vocab = fit_category_vocab(split["train"], "source"), fit_category_vocab(split["train"], "platform")
    part_of, tags_of, num_tags = load_question_meta(cfg["dataset"]["raw_root"]); item_part, item_tags = item_lookup(item_vocab, part_of, tags_of)
    max_len, min_history = int(seq["max_len"]), int(cfg["dataset"].get("min_history", 1))

    def dataset(n):
        return KTSequenceDatasetV3(split[n], skill_vocab, item_vocab, source_vocab, platform_vocab, max_len=max_len, min_history=min_history,
                                   split_name=n, cache_path=cache / f"{n}.npz")
    data = {n: dataset(n) for n in ("train", "validation", "test")}
    split.clear()
    tr = data["train"]
    full, oof, fold = fit_item_easiness(tr.items, tr.correct, tr.first_rows(), item_vocab.size)
    v3r.attach_mastery(tr, oof[fold, tr.items], item_part, item_tags, cache / "train_mastery_v1.npy")
    for n in ("validation", "test"):
        v3r.attach_mastery(data[n], full[data[n].items], item_part, item_tags, cache / f"{n}_mastery_v1.npy")

    if args.model == "xgb_f":
        import xgboost as xgb
        bcfg = cfg["baseline"]; xcfg = bcfg["xgboost"]; rng = np.random.default_rng(args.seed)
        def block(d, limit=None):
            rows = np.flatnonzero(d.eligible)                       # ascending = the order every other model writes its predictions
            if limit and len(rows) > limit: rows = np.sort(rng.choice(rows, size=int(limit), replace=False))
            x = np.hstack([d.behavior[rows][:, PROFILE_KEEP].astype(np.float32), d.ctx[rows].astype(np.float32), d.mastery[rows].astype(np.float32),
                           one_hot(d.source[rows], source_vocab.size), one_hot(d.platform[rows], platform_vocab.size), one_hot(item_part[d.items[rows]], 8)])
            return x, d.correct[rows].astype(np.int32)
        x_tr, y_tr = block(tr, bcfg.get("max_train_rows")); x_es, y_es = block(data["validation"], bcfg.get("max_validation_rows"))
        print(f"[{name}] inputs per target={x_tr.shape[1]} | train rows={len(y_tr):,}", flush=True)
        params = dict(n_estimators=int(xcfg["n_estimators"]), max_depth=int(xcfg["max_depth"]), learning_rate=float(xcfg["learning_rate"]),
                      subsample=float(xcfg["subsample"]), colsample_bytree=float(xcfg["colsample_bytree"]), reg_lambda=float(xcfg["reg_lambda"]),
                      min_child_weight=float(xcfg["min_child_weight"]), early_stopping_rounds=int(xcfg["early_stopping_rounds"]),
                      objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=args.seed, n_jobs=-1)
        if device.type == "cuda": params["device"] = "cuda"
        model = xgb.XGBClassifier(**params); model.fit(x_tr, y_tr, eval_set=[(x_es, y_es)], verbose=100)
        x_va, y_va = block(data["validation"]); x_te, y_te = block(data["test"])
        p_va, p_te = model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]
        best_epoch = int(getattr(model, "best_iteration", -1))
    else:
        dim = int(seq["embedding_dim"]); ctx_idx, behavior_idx = v3r.drop_indices(drops)
        model = BehaviorAwareKTV3(num_skills=skill_vocab.size, num_items=item_vocab.size, num_sources=source_vocab.size, num_platforms=platform_vocab.size,
                                  num_tags=num_tags, item_part=torch.from_numpy(item_part), item_tags=torch.from_numpy(item_tags), max_len=max_len + 1,
                                  embedding_dim=dim, num_heads=int(seq["transformer_heads"]), num_layers=int(seq["transformer_layers"]), dropout=float(seq["dropout"]),
                                  ablate_ctx_indices=tuple(ctx_idx), ablate_behavior_indices=tuple(behavior_idx), use_source="source" not in drops,
                                  use_profile="profile" not in drops, use_meta="meta" not in drops, use_elapsed="response_time" not in drops)
        model.history_encoder = LSTMHistory(dim); model.cross_attention = LastState()
        print(f"[{name}] params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)
        bs, workers = int(seq["batch_size"]), int(seq.get("num_workers", 0))
        train_loader = v3r.make_loader(tr, bs, True, args.seed, workers); val_loader = v3r.make_loader(data["validation"], bs, False, args.seed, workers)
        model.to(device)
        v3r.smoke_checks(model, next(iter(train_loader)), device)   # stops the run if a prediction could see its own answer
        trained = kt_trainer.train_kt_model(model=model, train_loader=train_loader, validation_loader=val_loader, device=device,
                                            epochs=int(seq["epochs"]), learning_rate=float(seq["learning_rate"]), weight_decay=float(seq["weight_decay"]),
                                            patience=int(seq["patience"]), checkpoint_path=out / "checkpoints" / f"{name}.resume.pt",
                                            history_path=out / "training_history" / f"{name}.csv", model_name=name, seed=args.seed)
        model, best_epoch = trained.model, int(trained.best_epoch)
        y_va, p_va = kt_trainer.predict_kt(model, val_loader, device)
        y_te, p_te = kt_trainer.predict_kt(model, v3r.make_loader(data["test"], bs, False, args.seed, workers), device)

    threshold = select_f1_threshold(y_va, p_va)
    result = {"model": args.model, "dropped_groups": drops, "seed": args.seed, "best_epoch": best_epoch, "decision_threshold": float(threshold),
              "validation": binary_metrics(y_va, p_va, threshold), "test": binary_metrics(y_te, p_te, threshold)}
    for part, (y, prob) in (("validation", (y_va, p_va)), ("test", (y_te, p_te))):
        np.savez(out / "predictions" / f"{name}_{part}.npz", y_true=np.asarray(y).astype(np.uint8), y_prob=np.asarray(prob).astype(np.float32))
    (out / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] done in {(time.time() - started) / 60:.1f} min | validation AUC {result['validation']['auc']:.4f} | test AUC {result['test']['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
'''
open(f"{K}/scripts/kt_samefeat.py", "w").write(SAMEFEAT_PY)
open(f"{K}/scripts/kt_a17_run.py", "w").write('''import runpy, sys, types
try: import shap
except Exception: sys.modules["shap"] = types.ModuleType("shap")
try:
    import matplotlib; matplotlib.use("Agg")
except Exception: pass
sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")
''')
# Base-paper model on KT3: the SAME script and settings as before. Only added: test predictions are saved (for the
# significance test) and checkpoints are written atomically (safe to stop at the time limit and resume).
HGAKT_OK = True; hgakt = open(f"{K}/scripts/run_hgakt_kt3_trackB.py").read()
if HG != "results/hgakt_kt3_trackB":
    if hgakt.count('"results/hgakt_kt3_trackB"') != 1: HGAKT_OK = False
    hgakt = hgakt.replace('"results/hgakt_kt3_trackB"', f'"{HG}"')
    # the trained base-paper model is COPIED (never moved): the script finds it complete, skips training and only scores the test set
    for name in ("hgakt_seed42.resume.pt", "hgakt_seed42_history.csv"):
        src, dst = f"{K}/results/hgakt_kt3_trackB/{name}", f"{K}/{HG}/{name}"
        if os.path.exists(src) and not os.path.exists(dst): os.makedirs(f"{K}/{HG}", exist_ok=True); shutil.copy2(src, dst); say(f"copied trained HGAKT {name} -> {HG} (no retraining)")
for old, new, count in (("torch.save(", "_atomic_save(", 2),
                        ("from adaptive_kt.utils.reproducibility import torch_device\n", "from adaptive_kt.utils.reproducibility import torch_device\n\n\ndef _atomic_save(obj, path):\n    tmp = Path(str(path) + \".tmp\")\n    torch.save(obj, tmp)\n    tmp.replace(path)\n", 1),
                        ('    result = {\n        "model": "HGAKT_KT3_trackB_adaptation",', '    (output / "predictions").mkdir(parents=True, exist_ok=True)\n    np.savez(output / "predictions" / f"hgakt_seed{args.seed}_validation.npz", y_true=np.asarray(val_true).astype(np.uint8), y_prob=np.asarray(val_prob).astype(np.float32))\n    np.savez(output / "predictions" / f"hgakt_seed{args.seed}_test.npz", y_true=np.asarray(test_true).astype(np.uint8), y_prob=np.asarray(test_prob).astype(np.float32))\n    result = {\n        "model": "HGAKT_KT3_trackB_adaptation",', 1)):
    if hgakt.count(old) != count: HGAKT_OK = False
    hgakt = hgakt.replace(old, new)
open(f"{K}/scripts/kt3_hgakt.py", "w").write(hgakt)

ENV = dict(os.environ, KT_ROOT=K, KT_OUTPUT=OUT, KT_INPUT=IN, KT_T0=str(T0), PYTHONUNBUFFERED="1", PYTHONPATH=os.pathsep.join(p for p in (f"{K}/src", os.environ.get("PYTHONPATH", "")) if p),
           KT_PROC="data/processed/ednet_kt3", KT_RES=RES_REL, KT_CFG=CFG, KT_SPLIT=SPLIT, KT_PREFIX="kt3", KT_DS_NAME="EdNet-KT3", KT_LOGS="logs/kt3final")
def save_zip():
    """Results, logs, predictions and the checkpoints of UNFINISHED runs (finished runs need only their results)."""
    try:
        open(f"{K}/logs/state_stamp.txt", "w").write(str(time.time()))
        with zipfile.ZipFile(f"{OUT}/kt3_final.zip.part", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for top in ("results", "logs"):
                for cur, dirs, files in os.walk(f"{K}/{top}"):
                    if os.path.basename(cur) in ("cache", "restart_checkpoints"): dirs[:] = []; continue
                    for name in files:
                        path = os.path.join(cur, name)
                        if ".tmp" in name or not name.endswith((".json", ".csv", ".log", ".txt", ".npz", ".resume.pt")): continue
                        if name.endswith(".resume.pt"):
                            stem = name[:-len(".resume.pt")]
                            if any(os.path.exists(p) and "test" in open(p).read() for p in (f"{cur}/{stem}_metrics.json", f"{os.path.dirname(cur)}/{stem}_metrics.json")): continue
                        z.write(path, os.path.relpath(path, K))
        os.replace(f"{OUT}/kt3_final.zip.part", f"{OUT}/kt3_final.zip")
        return f"kt3_final.zip ({os.path.getsize(OUT + '/kt3_final.zip') / 1e6:.0f} MB)"
    except Exception as exc: return f"zip not saved ({type(exc).__name__}: {exc})"
def save_full():
    """Prepared data + feature table + sequence caches, so the next run skips ~1 h of preparation."""
    if LAB: return "full state not needed on the lab machine"
    try:
        if not os.path.exists(f"{K}/{V3}/cache/test_mastery_v1.npy"): return "full state not saved (caches not built yet)"
        skip = lambda info: None if ".tmp" in os.path.basename(info.name) else info
        with tarfile.open(f"{OUT}/kt3_full.tar.part", "w") as a:
            for name in ("data/processed/ednet_kt3", f"{V3}/cache", "logs/state_stamp.txt"):
                if os.path.exists(f"{K}/{name}"): a.add(f"{K}/{name}", arcname=name, filter=skip)
        os.replace(f"{OUT}/kt3_full.tar.part", f"{OUT}/kt3_full.tar")
        return f"kt3_full.tar ({os.path.getsize(OUT + '/kt3_full.tar') / 1e9:.1f} GB)"
    except Exception as exc: return f"full state not saved ({type(exc).__name__}: {exc})"
def run_step(name, command):
    say(f"===== {name} =====")
    process = subprocess.Popen(command, cwd=K, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    for line in process.stdout:
        LAST[0] = time.time(); print(line.rstrip("\n"), flush=True)
    return process.wait() == 0

# 4) data: EdNet-KT3 prepared exactly as before (same parser, same frozen split) + the feature table for LR/RF/XGBoost
DATA_PY = '''import os, sys, time
sys.path[:0] = ["scripts", "src"]
import kaggle_run as kr
from pathlib import Path
from adaptive_kt.data.features import build_causal_features
from adaptive_kt.utils.tableio import read_table, write_table
if not (kr.INTERACTIONS / "_DONE").exists(): kr.prepare(kr.find_raw_kt3())
kr.make_split()
features = kr.INTERACTIONS.parent / "features"
if not (features / "_DONE").exists():
    features.mkdir(exist_ok=True); shards = sorted(kr.INTERACTIONS.glob("part-*.parquet")) + sorted(kr.INTERACTIONS.glob("part-*.pkl")); rows = 0
    for i, shard in enumerate(shards):
        target = features / shard.stem
        if not (target.with_suffix(".parquet").exists() or target.with_suffix(".pkl").exists()):
            write_table(build_causal_features(read_table(shard), recent_window=10, min_history=5, max_response_time_seconds=1200), target)
        if (i + 1) % 50 == 0 or i + 1 == len(shards): print(f"[features] {i + 1}/{len(shards)} shards", flush=True)
    rows = sum(len(read_table(p)) for p in sorted(features.glob("part-*.parquet")) + sorted(features.glob("part-*.pkl")))
    (features / "_DONE").write_text(str(rows)); print(f"[features] {rows:,} feature rows (original machine: 16,488,101)", flush=True)
'''
if LAB:
    need = [f"{K}/data/processed/ednet_kt3/interactions", f"{K}/{SPLIT}"]
    missing = [p for p in need if not os.path.exists(p)]
    if missing: raise SystemExit("Lab machine: these are missing, nothing was started: " + ", ".join(missing))
    cache = os.path.exists(f"{K}/{V3}/cache/test_mastery_v1.npy")
    say("lab machine: prepared data and frozen split found (no data step); V3 caches " + ("found" if cache else "NOT found -> first run rebuilds them (~30 min)"))
elif not run_step("data: EdNet-KT3 (same parser and frozen 70/10/20 split as before)", [sys.executable, "-u", "-c", DATA_PY]):
    raise SystemExit("\n##### The data step did not finish. Read the lines above. Fix it and run again.")

# 5) runs
def has_test(path):
    try: return "test" in json.load(open(path))
    except Exception: return False
def val_auc(task):
    try: return float(json.load(open(task["files"][0]))["validation"]["auc"])
    except Exception: return float("nan")
def merge(*parts): return ",".join(sorted({x for p in parts for x in p.split(",") if x}))
def suffix(drops): return "" if not drops else "_no_" + "_".join(sorted(x for x in drops.split(",") if x))
def v3_task(seed, drops=""):
    return dict(name=f"v3{suffix(drops)}_seed{seed}", family="v3", command=["scripts/run_trackB_proposed_v3.py", "--config", CFG, "--split", SPLIT, "--out", V3, "--v2-cache", f"{V3}/no_v2_cache",
                "--seed", str(seed), "--test"] + (["--drop", drops] if drops else []), files=[f"{K}/{V3}/v3{suffix(drops)}_seed{seed}_metrics.json"])
def samefeat_task(model, seed=42, drops=""):
    return dict(name=f"{model}{suffix(drops)}_seed{seed}", family="v3", command=["scripts/kt_samefeat.py", "--config", CFG, "--split", SPLIT, "--out", SF, "--cache", f"{V3}/cache",
                "--model", model, "--seed", str(seed)] + (["--drop", drops] if drops else []), files=[f"{K}/{SF}/{model}{suffix(drops)}_seed{seed}_metrics.json"])
def baseline_task(name, models, seed=42):
    return dict(name=f"{name}_seed{seed}", family="baseline", command=["scripts/run_trackB_baselines.py", "--config", CFG, "--split", SPLIT, "--out", BL, "--seed", str(seed), "--models", ",".join(models)],
                files=[f"{K}/{BL}/{m}_seed{seed}_metrics.json" for m in models])
def hgakt_task(seed): return dict(name=f"hgakt_seed{seed}", family="hgakt", command=["scripts/kt3_hgakt.py", "--config", CFG, "--seed", str(seed)], files=[f"{K}/{HG}/hgakt_seed{seed}_metrics.json"])
def key_baselines(seed): return [baseline_task("akt", ["akt"], seed)] + ([hgakt_task(seed)] if HGAKT_OK else [])
def other_baselines(seed): return [samefeat_task("xgb_f", seed), baseline_task("classical", ["logistic_regression", "random_forest", "xgboost"], seed),
                                   baseline_task("other_sequence", ["dkt_lstm", "transformer_kt", "sakt", "saint_plus"], seed)]
if not HGAKT_OK: say("base-paper script looks different from the expected one -> HGAKT is skipped (tell me)")
done = lambda task: all(has_test(path) for path in task["files"])
def log_text(task):
    try: return open(f"{LOGS}/{task['name']}.log", errors="ignore").read().replace("\r", "\n")
    except OSError: return ""
def status(task):
    lines = [line.strip() for line in log_text(task).splitlines() if line.strip()]
    epochs = [line for line in lines if "epoch" in line and "val_auc" in line]
    return (epochs[-1] if epochs else (lines[-1] if lines else "starting"))[-150:]
def result_line(task):
    return " | ".join(f"{os.path.basename(p).replace('_metrics.json', '')}: test AUC {json.load(open(p))['test']['auc']:.4f} (val {json.load(open(p))['validation']['auc']:.4f})" for p in task["files"] if has_test(p))
V3_CACHE = f"{K}/{V3}/cache/test_mastery_v1.npy"; RUNNING = {}; ALL = []; last = [time.time()]
def stop_all():
    for name, (process, gpu, task) in list(RUNNING.items()):
        if done(task):
            try: process.wait(timeout=300)
            except Exception: pass
            continue
        say(f"stopping {name} (time limit; next run continues from its last finished epoch)"); process.terminate()
    for name, (process, gpu, task) in list(RUNNING.items()):
        try: process.wait(timeout=120)
        except Exception: process.kill()
        RUNNING.pop(name, None)
def schedule(title, phase, background=()):
    """Run until every task in `phase` is finished. `background` tasks fill free GPUs and may keep running afterwards.
    Returns False when the time limit is reached or a phase task failed in this session."""
    for task in list(phase) + list(background):
        if task["name"] not in [t["name"] for t in ALL]: ALL.append(task)
    say(f"===== {title}: {sum(done(t) for t in phase)}/{len(phase)} finished =====")
    while True:
        for name, (process, gpu, task) in list(RUNNING.items()):
            if process.poll() is None: continue
            RUNNING.pop(name)
            if done(task): say(f"FINISHED {name} -> {result_line(task)}")
            else:
                FAILED_NOW.add(name); say(f"FAILED {name} (exit code {process.returncode}); the other runs continue. Last lines of its log:\n" + "\n".join(log_text(task).splitlines()[-25:]))
            say("saved: " + save_zip()); last[0] = time.time()
        if time.time() - T0 > HARD: stop_all(); return False
        left = [t for t in phase if not done(t)]
        if not left: return True
        if all(t["name"] in FAILED_NOW for t in left): return False
        for task in [t for t in list(phase) + list(background) if not done(t) and t["name"] not in RUNNING and t["name"] not in FAILED_NOW]:
            if len(RUNNING) >= GPUS or time.time() - T0 > SOFT: break
            if RUNNING and ram()[1] < MIN_FREE_GB: break                                      # not enough free memory for one more KT3 process
            if task["family"] == "v3" and not os.path.exists(V3_CACHE) and any(t["family"] == "v3" for _, _, t in RUNNING.values()): continue
            gpu = next(g for g in range(GPUS) if g not in [g2 for _, g2, _ in RUNNING.values()])
            say(f"started {task['name']} on GPU {gpu}")
            RUNNING[task["name"]] = (subprocess.Popen([sys.executable, "-u", "scripts/kt_a17_run.py"] + task["command"], cwd=K, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                                                      stdout=open(f"{LOGS}/{task['name']}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), gpu, task)
        if not RUNNING: return False                                                              # nothing could start (time limit reached)
        if time.time() - last[0] > EVERY:
            last[0] = time.time()
            for name, (_, _, task) in RUNNING.items(): say(f"  {name}: {status(task)}")
            say("  saved: " + save_zip())
        time.sleep(float(os.environ.get("KT_POLL", "30")))
FAILED_NOW = set()
def finish(message):
    stop_all(); say("saved: " + save_zip()); say("saved: " + save_full())
    try:
        from IPython.display import FileLink, display
        os.chdir(OUT)
        for name in ("kt3_final.zip", "kt3_main_table.csv", "kt3_beats.csv", "kt3_ablation_final.csv", "kt3_training_check.csv", "kt3_selection.csv"):
            if os.path.exists(name): display(FileLink(name))
    except Exception: pass
    missing = [t["name"] for t in ALL if not done(t)]
    print(message + (f"\nNot finished yet: {missing}" if missing else ""), flush=True)

AGAIN = ("NOT FINISHED (Kaggle time limit or a failed run). Everything is saved. Next: open the notebook, Add Input -> the Output of this run "
         "(or upload kt3_final.zip + kt3_full.tar to kt-code), then Save Version -> Save & Run All. It continues where it stopped.")
GROUPS = ["profile", "recency", "experience,timing", "engagement", "source", "elo,meta", "response_time", "recent_accuracy"]
# Step A: backbone (attention = V3, LSTM) on VALIDATION, seed 42. AKT and HGAKT (needed for the significance test) use free GPUs meanwhile.
ok = schedule("Step A: backbone", [v3_task(42), samefeat_task("dkt_f")], key_baselines(42))
if not ok: finish(AGAIN); raise SystemExit(0)
att, lstm = val_auc(v3_task(42)), val_auc(samefeat_task("dkt_f"))
if not (att == att and lstm == lstm): finish(AGAIN); raise SystemExit(0)
BACKBONE = "lstm" if lstm > att else "attention"
final = (lambda seed, drops="": samefeat_task("dkt_f", seed, drops)) if BACKBONE == "lstm" else (lambda seed, drops="": v3_task(seed, drops))
say(f"backbone chosen on validation: {BACKBONE} (attention {att:.4f}, LSTM {lstm:.4f})")
# Step B: an input group is removed when removing it does not lower validation AUC
ok = schedule("Step B: each input group removed once", [final(42, g) for g in GROUPS], key_baselines(42))
if not ok: finish(AGAIN); raise SystemExit(0)
full_val = val_auc(final(42)); group_val = {g: val_auc(final(42, g)) for g in GROUPS}
removed = [g for g in GROUPS if group_val[g] == group_val[g] and group_val[g] >= full_val]; FINAL = ""
if removed:
    ok = schedule("Step B: model without the groups that did not help", [final(42, merge(*removed))], key_baselines(42))
    if not ok: finish(AGAIN); raise SystemExit(0)
    if val_auc(final(42, merge(*removed))) >= full_val: FINAL = merge(*removed)
kept = [g for g in GROUPS if g not in removed or not FINAL]
selection = {"backbone": BACKBONE, "validation_auc_attention": att, "validation_auc_lstm": lstm, "validation_auc_full": full_val, "validation_auc_without_group": group_val,
             "groups_removed": removed if FINAL else [], "validation_auc_pruned": val_auc(final(42, merge(*removed))) if removed else None, "final_drops": FINAL,
             "final_folder": "samefeat" if BACKBONE == "lstm" else "v3", "final_prefix": "dkt_f" if BACKBONE == "lstm" else "v3", "kept_groups": kept,
             "rule": "backbone and input groups chosen by validation AUC only (seed 42); test used once afterwards"}
json.dump(selection, open(f"{RES}/selection.json", "w"), indent=2)
say(f"final model: {BACKBONE} backbone, inputs removed by validation: {selection['groups_removed'] or 'none'}")
# Step C: final model 3 seeds, ablation of the final model, baselines (seed 42 first, then 52 and 62)
step_c = [final(s, FINAL) for s in (42, 52, 62)] + key_baselines(42) + [final(42, merge(FINAL, g)) for g in kept + (["mastery"] if "mastery" not in kept else [])] + other_baselines(42)
ok = schedule("Step C: final model seeds + ablation + baselines", step_c)
if not ok: finish(AGAIN); raise SystemExit(0)
TABLE = [sys.executable, "-u", "scripts/kt_a17.py", "table"]
if not os.path.exists(f"{LOGS}/tables_seed42_done.txt"):                     # tables as soon as every baseline has seed 42
    save_zip()
    if run_step("tables + significance test (baselines: seed 42 so far)", TABLE): open(f"{LOGS}/tables_seed42_done.txt", "w").write("ok")
if LAB: finish("DONE (lab): Steps A-C finished, tables printed above. Step D (baselines seeds 52/62) is optional and was not run."); raise SystemExit(0)
ok = schedule("Step D: baselines seeds 52 and 62", key_baselines(52) + key_baselines(62) + other_baselines(52) + other_baselines(62))
if not ok: finish(AGAIN); raise SystemExit(0)
save_zip()
if run_step("tables + significance test (all seeds)", TABLE): finish("DONE: all EdNet-KT3 runs finished")
else: finish("All runs finished, but the table step stopped with an error (see above). Send me its last lines.")
