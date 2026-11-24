"""ASSISTments 2017 (cross-dataset): stages 'data' and 'table'. Run from the project root."""
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
