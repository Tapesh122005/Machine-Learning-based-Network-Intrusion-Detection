#!/usr/bin/env python3
"""
Machine Learning based Network Intrusion Detection
==================================================
Exp 7 - Intrusion Detection System

Trains and compares several classifiers (Random Forest, SVM, Decision Tree,
KNN, Logistic Regression, Naive Bayes, Gradient Boosting) on the NSL-KDD
dataset and reports Accuracy, Precision, Recall (and F1) for each.

Usage
-----
    python nids.py                      # binary: normal vs attack (NSL-KDD in ./data)
    python nids.py --multiclass         # 5 classes: Normal, DoS, Probe, R2L, U2R
    python nids.py --synthetic          # force synthetic demo data (pipeline test only)
    python nids.py --data-dir path/     # folder containing KDDTrain+.txt / KDDTest+.txt

If the NSL-KDD files are not found, the script automatically falls back to a
synthetic dataset with the same schema so the pipeline can still be tested.
Results on synthetic data are NOT valid for your report.
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# NSL-KDD schema
# --------------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def find_file(data_dir, candidates):
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    return None


def load_nsl_kdd(data_dir):
    train_p = find_file(data_dir, ["KDDTrain+.txt", "KDDTrain+.csv", "KDDTrain.txt"])
    test_p = find_file(data_dir, ["KDDTest+.txt", "KDDTest+.csv", "KDDTest.txt"])
    if not (train_p and test_p):
        return None
    read = lambda p: pd.read_csv(p, header=None, names=COLUMNS)
    return read(train_p), read(test_p), "NSL-KDD (KDDTrain+ / KDDTest+)"


def make_synthetic(n_train=20000, n_test=6000, seed=RANDOM_STATE):
    """Synthetic data with NSL-KDD schema. For testing the pipeline only."""
    rng = np.random.default_rng(seed)
    services = ["http", "ftp_data", "smtp", "domain_u", "private", "ecr_i",
                "telnet", "other", "ftp", "eco_i"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH"]
    protos = ["tcp", "udp", "icmp"]

    def gen(n, shift):
        probs = [0.52, 0.30, 0.10, 0.06, 0.02]
        cats = rng.choice(CLASS_ORDER, size=n, p=probs)
        rows = []
        for c in cats:
            r = {}
            if c == "Normal":
                r.update(duration=rng.exponential(30), protocol_type=rng.choice(protos, p=[.8, .15, .05]),
                         service=rng.choice(services[:5]), flag=rng.choice(flags, p=[.9, .02, .03, .03, .02]),
                         src_bytes=rng.lognormal(6, 1.5), dst_bytes=rng.lognormal(7, 1.8),
                         logged_in=int(rng.random() < .8), count=rng.integers(1, 40),
                         serror_rate=rng.beta(1, 30), rerror_rate=rng.beta(1, 30),
                         same_srv_rate=rng.beta(8, 2), diff_srv_rate=rng.beta(1, 12))
            elif c == "DoS":
                r.update(duration=0.0, protocol_type=rng.choice(protos, p=[.5, .1, .4]),
                         service=rng.choice(["private", "ecr_i", "http", "other"]),
                         flag=rng.choice(flags, p=[.15, .7, .08, .04, .03]),
                         src_bytes=rng.choice([0, 1032, 520, rng.lognormal(5, 1)]), dst_bytes=0,
                         logged_in=0, count=rng.integers(150, 511),
                         serror_rate=rng.beta(12, 1.5), rerror_rate=rng.beta(1, 25),
                         same_srv_rate=rng.beta(9, 1.5), diff_srv_rate=rng.beta(1, 25))
            elif c == "Probe":
                r.update(duration=rng.exponential(3), protocol_type=rng.choice(protos, p=[.6, .2, .2]),
                         service=rng.choice(services), flag=rng.choice(flags, p=[.3, .15, .4, .1, .05]),
                         src_bytes=rng.lognormal(2, 1.5), dst_bytes=rng.lognormal(1, 1.5),
                         logged_in=0, count=rng.integers(1, 120),
                         serror_rate=rng.beta(2, 6), rerror_rate=rng.beta(4, 4),
                         same_srv_rate=rng.beta(2, 5), diff_srv_rate=rng.beta(5, 3))
            elif c == "R2L":
                r.update(duration=rng.exponential(120), protocol_type="tcp",
                         service=rng.choice(["ftp_data", "ftp", "smtp", "telnet", "http"]),
                         flag="SF", src_bytes=rng.lognormal(7, 1.5), dst_bytes=rng.lognormal(6, 1.8),
                         logged_in=int(rng.random() < .35), count=rng.integers(1, 15),
                         serror_rate=rng.beta(1, 30), rerror_rate=rng.beta(1, 20),
                         same_srv_rate=rng.beta(7, 2), diff_srv_rate=rng.beta(1, 14))
            else:  # U2R
                r.update(duration=rng.exponential(200), protocol_type="tcp",
                         service=rng.choice(["telnet", "ftp_data", "other"]),
                         flag="SF", src_bytes=rng.lognormal(7.5, 1.2), dst_bytes=rng.lognormal(7, 1.5),
                         logged_in=1, count=rng.integers(1, 8),
                         serror_rate=rng.beta(1, 40), rerror_rate=rng.beta(1, 40),
                         same_srv_rate=rng.beta(8, 2), diff_srv_rate=rng.beta(1, 14))
            # shared / weakly informative features
            r["wrong_fragment"] = int(c == "DoS" and rng.random() < .05)
            r["hot"] = int(rng.poisson(2.5 if c in ("R2L", "U2R") else .1))
            r["num_failed_logins"] = int(rng.poisson(1.2 if c == "R2L" else .02))
            r["root_shell"] = int(c == "U2R" and rng.random() < .5)
            r["num_root"] = int(rng.poisson(3 if c == "U2R" else .05))
            r["num_compromised"] = int(rng.poisson(1.5 if c == "U2R" else .05))
            r["srv_count"] = int(max(1, r["count"] * rng.uniform(.4, 1.2)))
            r["dst_host_count"] = int(rng.integers(1, 256))
            r["dst_host_srv_count"] = int(rng.integers(1, 256))
            r["dst_host_same_srv_rate"] = float(np.clip(r["same_srv_rate"] + rng.normal(0, .15 + shift), 0, 1))
            r["dst_host_diff_srv_rate"] = float(np.clip(r["diff_srv_rate"] + rng.normal(0, .1 + shift), 0, 1))
            r["dst_host_serror_rate"] = float(np.clip(r["serror_rate"] + rng.normal(0, .1 + shift), 0, 1))
            r["dst_host_rerror_rate"] = float(np.clip(r["rerror_rate"] + rng.normal(0, .1 + shift), 0, 1))
            r["label_cat"] = c
            rows.append(r)
        df = pd.DataFrame(rows)
        for col in COLUMNS:
            if col not in df.columns and col not in ("label", "difficulty"):
                df[col] = 0
        # label in NSL-KDD style raw names
        raw = {"Normal": "normal", "DoS": "neptune", "Probe": "satan",
               "R2L": "guess_passwd", "U2R": "buffer_overflow"}
        df["label"] = df["label_cat"].map(raw)
        df["difficulty"] = 15
        return df[COLUMNS]

    return gen(n_train, 0.0), gen(n_test, 0.08), "SYNTHETIC demo data (NOT NSL-KDD)"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def prepare(df_train, df_test, multiclass):
    def targets(df):
        cat = df["label"].str.strip().str.lower().map(ATTACK_CATEGORY)
        cat = cat.fillna("R2L")  # unseen rare attack names -> treat as attack
        return cat.values if multiclass else (cat != "Normal").astype(int).values

    y_tr, y_te = targets(df_train), targets(df_test)
    X_tr = df_train.drop(columns=["label", "difficulty"])
    X_te = df_test.drop(columns=["label", "difficulty"])
    numeric = [c for c in X_tr.columns if c not in CATEGORICAL]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return X_tr, X_te, y_tr, y_te, pre


def get_models(pre_factory, n_jobs=-1):
    mk = lambda clf: Pipeline([("pre", pre_factory()), ("clf", clf)])
    return {
        "Random Forest": mk(RandomForestClassifier(n_estimators=200, n_jobs=n_jobs,
                                                   random_state=RANDOM_STATE)),
        "SVM (RBF)": mk(SVC(kernel="rbf", C=10, gamma="scale", probability=False,
                            random_state=RANDOM_STATE)),
        "Decision Tree": mk(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "KNN (k=5)": mk(KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)),
        "Logistic Regression": mk(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Naive Bayes": mk(GaussianNB()),
        "Gradient Boosting": mk(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    }


# --------------------------------------------------------------------------
# Evaluation & plots
# --------------------------------------------------------------------------
def scores_for_roc(model, X):
    clf = model.named_steps["clf"]
    Xt = model.named_steps["pre"].transform(X)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xt)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(Xt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--multiclass", action="store_true", help="5-class instead of binary")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic demo data")
    ap.add_argument("--svm-samples", type=int, default=25000,
                    help="max training rows for the RBF SVM (it scales ~O(n^2)); 0 = use all")
    ap.add_argument("--models", default="", help="comma-separated subset of model names")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loaded = None if args.synthetic else load_nsl_kdd(args.data_dir)
    if loaded is None:
        if not args.synthetic:
            print(f"[!] NSL-KDD files not found in '{args.data_dir}/' -> using SYNTHETIC demo data.")
            print("    Put KDDTrain+.txt and KDDTest+.txt there for real results.\n")
        loaded = make_synthetic()
    df_tr, df_te, source = loaded
    mode = "multiclass (5 classes)" if args.multiclass else "binary (normal vs attack)"
    print(f"Dataset : {source}")
    print(f"Task    : {mode}")
    print(f"Train   : {len(df_tr):,} rows | Test: {len(df_te):,} rows\n")

    X_tr, X_te, y_tr, y_te, pre = prepare(df_tr, df_te, args.multiclass)
    classes = sorted(np.unique(y_tr), key=lambda c: CLASS_ORDER.index(c) if args.multiclass else c)
    avg = "weighted" if args.multiclass else "binary"

    from sklearn.base import clone
    models = get_models(lambda: clone(pre))
    if args.models:
        keep = [m.strip().lower() for m in args.models.split(",")]
        models = {k: v for k, v in models.items() if k.lower() in keep}

    rows, fitted, reports = [], {}, []
    for name, model in models.items():
        Xf, yf = X_tr, y_tr
        note = ""
        if name.startswith("SVM") and args.svm_samples and len(X_tr) > args.svm_samples:
            idx = np.random.default_rng(RANDOM_STATE).choice(len(X_tr), args.svm_samples, replace=False)
            Xf, yf = X_tr.iloc[idx], y_tr[idx]
            note = f" (trained on {args.svm_samples:,}-row stratified-random subsample)"
        t0 = time.time()
        model.fit(Xf, yf)
        t_fit = time.time() - t0
        t0 = time.time()
        pred = model.predict(X_te)
        t_pred = time.time() - t0

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
            "F1-score": f1_score(y_te, pred, average=avg, zero_division=0),
            "Train time (s)": round(t_fit, 2),
            "Predict time (s)": round(t_pred, 2),
        }
        if not args.multiclass:
            s = scores_for_roc(model, X_te)
            row["ROC-AUC"] = roc_auc_score(y_te, s) if s is not None else np.nan
        rows.append(row)
        fitted[name] = (model, pred)
        reports.append(f"===== {name}{note} =====\n" +
                       classification_report(y_te, pred, digits=4, zero_division=0) + "\n")
        print(f"{name:20s} acc={row['Accuracy']:.4f} prec={row['Precision']:.4f} "
              f"rec={row['Recall']:.4f} f1={row['F1-score']:.4f} ({t_fit:.1f}s){note}")

    res = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    tag = "multiclass" if args.multiclass else "binary"
    res.to_csv(os.path.join(args.out_dir, f"metrics_{tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"classification_reports_{tag}.txt"), "w") as f:
        f.write(f"Dataset: {source}\nTask: {mode}\n\n" + "\n".join(reports))

    print("\n" + res.round(4).to_string(index=False))

    sns.set_theme(style="whitegrid")
    # 1. metric comparison bars
    m = res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
                 var_name="Metric", value_name="Score")
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=m, x="Model", y="Score", hue="Metric")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0))
    lo = max(0.0, m["Score"].min() - 0.05)
    plt.ylim(lo, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"Classifier comparison - {mode}\n{source}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"comparison_{tag}.png"), dpi=150)
    plt.close()

    # 2. confusion matrices (best 3 + RF/SVM)
    show = list(res["Model"][:3])
    for must in ("Random Forest", "SVM (RBF)"):
        if must in fitted and must not in show:
            show.append(must)
    ncol = min(3, len(show))
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow), squeeze=False)
    labels = ["Normal", "Attack"] if not args.multiclass else classes
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), show):
        ax.axis("on")
        cm = confusion_matrix(y_te, fitted[name][1], labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"confusion_matrices_{tag}.png"), dpi=150)
    plt.close()

    # 3. ROC curves (binary only)
    if not args.multiclass:
        plt.figure(figsize=(6.5, 5.5))
        for name, (model, _) in fitted.items():
            s = scores_for_roc(model, X_te)
            if s is None:
                continue
            fpr, tpr, _ = roc_curve(y_te, s)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, s):.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate (attack recall)")
        plt.title("ROC curves")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "roc_curves.png"), dpi=150)
        plt.close()

    # 4. Random Forest feature importance
    if "Random Forest" in fitted:
        rf = fitted["Random Forest"][0]
        names = rf.named_steps["pre"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(15)
        plt.figure(figsize=(7.5, 5.5))
        sns.barplot(x=imp.values, y=imp.index, color="#3b6ea5")
        plt.title("Random Forest - top 15 features")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"rf_feature_importance_{tag}.png"), dpi=150)
        plt.close()

    print(f"\nSaved outputs to '{args.out_dir}/'")
    if "SYNTHETIC" in source:
        print("NOTE: these numbers come from synthetic data - do not put them in your report.")


if __name__ == "__main__":
    main()
