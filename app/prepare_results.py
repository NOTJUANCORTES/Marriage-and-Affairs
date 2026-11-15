"""Refit the published Marriage and Affairs models and write display data.

Settings match final_portfolio/final/final_portfolio.py:
  predictors: rate_marriage, age, yrs_married, children, religious, educ
  target: had_affair = 1 if affairs > 0 else 0
  train_test_split(test_size=0.2, random_state=42, stratify=y)
  LogisticRegression(max_iter=10000)
  DecisionTreeClassifier(max_depth=3, random_state=42)  # Gini is the default

Occupations are summarized for the charts and are not used in either model.
The script does not one-hot encode predictors.

Run from anywhere:
  python app/prepare_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "final_portfolio" / "final" / "_data" / "affairs.csv"
OUT_PATH = (
    SCRIPT_DIR.parent.parent
    / "juan-cortes-matrix-portfolio"
    / "dist"
    / "marriage-affairs"
    / "results.json"
)

PREDICTORS = [
    "rate_marriage",
    "age",
    "yrs_married",
    "children",
    "religious",
    "educ",
]

EDUC_LABELS = {
    9: "9 · grade school",
    12: "12 · high school",
    14: "14 · some college",
    16: "16 · college graduate",
    17: "17 · some graduate school",
    20: "20 · advanced degree",
}

OCC_LABELS = {
    1: "1 · Student",
    2: "2 · Agriculture",
    3: "3 · White Collar",
    4: "4 · Teacher/Nurse",
    5: "5 · Managerial",
    6: "6 · Professional",
}

RATING_LABELS = {
    1: "1 · very poor",
    2: "2 · poor",
    3: "3 · fair",
    4: "4 · good",
    5: "5 · very good",
}

RELIGIOUS_LABELS = {
    1: "1 · not religious",
    2: "2",
    3: "3",
    4: "4 · strongly religious",
}


def fmt_num(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text


def human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def comma(n: int) -> str:
    return f"{int(n):,}"


def pct(rate: float, digits: int = 1) -> str:
    return f"{100 * float(rate):.{digits}f}%"


def round3(value: float) -> str:
    return f"{float(value):.3f}"


def py(value):
    if isinstance(value, (np.floating, float)):
        return round(float(value), 10)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def level_label(feature: str, value: float) -> str:
    key = int(value) if abs(value - round(value)) < 1e-8 else None
    if feature == "rate_marriage":
        return RATING_LABELS[int(value)]
    if feature == "religious":
        return RELIGIOUS_LABELS[int(value)]
    if feature == "educ":
        return EDUC_LABELS[int(value)]
    if feature in ("occupation", "occupation_husb"):
        return OCC_LABELS[int(value)]
    if feature == "children" and abs(value - 5.5) < 1e-8:
        return "5.5"
    if feature == "age":
        return fmt_num(value)
    if feature == "yrs_married":
        return fmt_num(value)
    if key is not None:
        return str(key)
    return fmt_num(value)


def grouped(df: pd.DataFrame, feature: str) -> list[dict]:
    rows = []
    for value, part in df.groupby(feature, sort=True):
        n = int(len(part))
        n_affair = int(part["had_affair"].sum())
        rate = n_affair / n
        mean_affairs = float(part["affairs"].mean())
        rows.append(
            {
                "value": py(value),
                "label": level_label(feature, float(value)),
                "n": n,
                "n_label": comma(n),
                "n_affair": n_affair,
                "affair_rate": py(rate),
                "affair_rate_label": pct(rate),
                "mean_affairs": py(mean_affairs),
                "mean_affairs_label": f"{mean_affairs:.2f}",
            }
        )
    return rows


def reading_for(feature: str, rows: list[dict]) -> str:
    by_value = {r["value"]: r for r in rows}
    if feature == "rate_marriage":
        low, high = by_value[1], by_value[5]
        mid = by_value[2]
        return (
            f"The share who reported an affair falls as the marriage rating rises: "
            f"{low['affair_rate_label']} at rating 1 ({low['n_affair']} of {low['n_label']}) "
            f"and {high['affair_rate_label']} at rating 5 ({high['n_affair']} of {high['n_label']}). "
            f"This is the strongest pattern in the file. The average affairs measure is not "
            f"ranked the same way: it is highest at rating 2 ({mid['mean_affairs_label']}), "
            f"higher than at rating 1 ({low['mean_affairs_label']})."
        )
    if feature == "yrs_married":
        short, long = by_value[0.5], by_value[16.5]
        end = by_value[23.0]
        peak = max(rows, key=lambda r: r["mean_affairs"])
        return (
            f"Almost nobody in the shortest marriages reported an affair "
            f"({short['affair_rate_label']} at 0.5 years, {short['n_affair']} of {short['n_label']}). "
            f"The share is much higher later, including {long['affair_rate_label']} at 16.5 years. "
            f"The average affairs measure does not keep rising: it peaks at "
            f"{fmt_num(peak['value'])} years ({peak['mean_affairs_label']}) and is "
            f"{end['mean_affairs_label']} at 23 years. The yes/no rate and the original measure "
            f"are not the same story."
        )
    if feature == "age":
        young, older = by_value[17.5], by_value[37.0]
        return (
            f"The share reporting an affair is {young['affair_rate_label']} at age code 17.5 "
            f"and {older['affair_rate_label']} at 37. The average affairs measure moves the "
            f"other way: {young['mean_affairs_label']} at 17.5 and {older['mean_affairs_label']} at 37. "
            f"Age and years married travel together, which matters when the logistic model "
            f"estimates them at the same time."
        )
    if feature == "children":
        none, two = by_value[0.0], by_value[2.0]
        return (
            f"People with no children reported an affair less often "
            f"({none['affair_rate_label']}, {none['n_affair']} of {none['n_label']}) than people "
            f"with two ({two['affair_rate_label']}). The share levels off after two children. "
            f"In the logistic model, the children coefficient is nearly zero once the other "
            f"predictors are included."
        )
    if feature == "religious":
        low, high = by_value[1], by_value[4]
        return (
            f"Religious commitment is stored as 1, 2, 3, and 4. "
            f"{low['affair_rate_label']} of people at 1 (not religious) reported an affair, "
            f"compared with {high['affair_rate_label']} at 4 (strongly religious). "
            f"The average affairs measure falls across the same scale, from "
            f"{low['mean_affairs_label']} to {high['mean_affairs_label']}."
        )
    if feature == "educ":
        hs, college = by_value[12], by_value[16]
        grade = by_value[9]
        return (
            f"High school ({hs['affair_rate_label']}) and some college sit higher than a college "
            f"degree ({college['affair_rate_label']}) or the graduate codes. The grade-school code "
            f"is only {grade['n_label']} people, so that bar is thin. Education is a mild pattern, "
            f"and the logistic coefficient on the education code is essentially flat."
        )
    if feature == "occupation":
        mgr, student = by_value[5], by_value[1]
        return (
            f"Exploratory only — occupation is not in either model. "
            f"Managerial respondents have the highest affair share "
            f"({mgr['affair_rate_label']}, {mgr['n_affair']} of {mgr['n_label']}). "
            f"The student code is only {student['n_label']} people. "
            f"These codes are job categories, not a numeric scale."
        )
    if feature == "occupation_husb":
        rates = [r["affair_rate"] for r in rows]
        return (
            f"Exploratory only — husband’s occupation is not in either model. "
            f"Affair shares sit between {pct(min(rates))} and {pct(max(rates))}, a narrower spread "
            f"than marriage rating. The same 1–6 category labels are used."
        )
    return ""


VARIABLES = [
    {
        "id": "rate_marriage",
        "label": "Marriage rating",
        "in_model": True,
        "scale": "1 = very poor, 5 = very good. Five recorded values.",
    },
    {
        "id": "yrs_married",
        "label": "Years married",
        "in_model": True,
        "scale": "Recorded codes: 0.5, 2.5, 6, 9, 13, 16.5, and 23.",
    },
    {
        "id": "age",
        "label": "Age",
        "in_model": True,
        "scale": "Recorded codes: 17.5, 22, 27, 32, 37, and 42.",
    },
    {
        "id": "children",
        "label": "Children",
        "in_model": True,
        "scale": "Recorded codes: 0, 1, 2, 3, 4, and 5.5.",
    },
    {
        "id": "religious",
        "label": "Religious commitment",
        "in_model": True,
        "scale": "1 = not religious, 4 = strongly religious. The file has no 5.",
    },
    {
        "id": "educ",
        "label": "Education",
        "in_model": True,
        "scale": "9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree.",
    },
    {
        "id": "occupation",
        "label": "Respondent occupation",
        "in_model": False,
        "scale": "Category codes used only on this chart: 1 Student, 2 Agriculture, 3 White Collar, 4 Teacher/Nurse, 5 Managerial, 6 Professional.",
    },
    {
        "id": "occupation_husb",
        "label": "Husband occupation",
        "in_model": False,
        "scale": "Same category codes as respondent occupation. Chart only.",
    },
]


def class_block(y_true, y_pred) -> dict:
    labels = [0, 1]
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp = (int(matrix[0, 0]), int(matrix[0, 1]))
    fn, tp = (int(matrix[1, 0]), int(matrix[1, 1]))
    accuracy = float(accuracy_score(y_true, y_pred))

    def pack(key: str) -> dict:
        row = report[key]
        return {
            "precision": py(row["precision"]),
            "precision_label": round3(row["precision"]),
            "recall": py(row["recall"]),
            "recall_label": round3(row["recall"]),
            "f1": py(row["f1-score"]),
            "f1_label": round3(row["f1-score"]),
            "support": int(row["support"]),
            "support_label": comma(row["support"]),
        }

    no_affair = pack("0")
    affair = pack("1")
    return {
        "accuracy": py(accuracy),
        "accuracy_label": pct(accuracy, 2),
        "correct": tn + tp,
        "n": int(len(y_true)),
        "correct_label": f"{comma(tn + tp)} of {comma(len(y_true))}",
        "no_affair": no_affair,
        "affair": affair,
        "confusion": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "predicted_no_affair": tn + fn,
        "predicted_affair": fp + tp,
    }


def export_tree(tree: DecisionTreeClassifier, frame: pd.DataFrame, y: pd.Series) -> dict:
    feature_labels = {
        "rate_marriage": "Marriage rating",
        "age": "Age",
        "yrs_married": "Years married",
        "children": "Children",
        "religious": "Religious commitment",
        "educ": "Education",
    }
    estimator = tree.tree_
    reached = tree.decision_path(frame[PREDICTORS])

    def levels_on(node_id: int, feature: str, predicate) -> list[str]:
        row_ids = reached[:, node_id].toarray().ravel().astype(bool)
        present = sorted(float(v) for v in frame.loc[row_ids, feature].unique())
        kept = [v for v in present if predicate(v)]
        return [fmt_num(v) for v in kept]

    def walk(node_id: int) -> dict:
        n_train = int(estimator.n_node_samples[node_id])
        row_ids = reached[:, node_id].toarray().ravel().astype(bool)
        y_node = y.loc[row_ids]
        n_affair = int(y_node.sum())
        n_no = int((y_node == 0).sum())
        gini = float(estimator.impurity[node_id])
        common = {
            "n_train": n_train,
            "n_train_label": comma(n_train),
            "n_affair": n_affair,
            "n_no_affair": n_no,
            "affair_share_label": pct(n_affair / n_train) if n_train else "—",
            "gini": py(gini),
            "gini_label": f"{gini:.3f}",
        }
        if estimator.children_left[node_id] == -1:
            predicted = "Affair" if n_affair >= n_no else "No affair"
            # Majority matches the classifier; ties would go to the first class.
            pred_code = int(tree.classes_[np.argmax(estimator.value[node_id])])
            predicted = "Affair" if pred_code == 1 else "No affair"
            return {
                **common,
                "leaf": True,
                "predicted": predicted,
            }

        feature = PREDICTORS[int(estimator.feature[node_id])]
        threshold = float(estimator.threshold[node_id])
        left_levels = levels_on(node_id, feature, lambda v: v <= threshold + 1e-8)
        right_levels = levels_on(node_id, feature, lambda v: v > threshold + 1e-8)
        return {
            **common,
            "leaf": False,
            "feature": feature,
            "feature_label": feature_labels[feature],
            "threshold": py(threshold),
            "threshold_label": fmt_num(threshold),
            "left_label": human_join(left_levels),
            "right_label": human_join(right_levels),
            "left": walk(int(estimator.children_left[node_id])),
            "right": walk(int(estimator.children_right[node_id])),
        }

    def paths(node: dict, steps: list[dict]) -> list[dict]:
        if node["leaf"]:
            return [
                {
                    "steps": steps,
                    "predicted": node["predicted"],
                    "n_train": node["n_train"],
                    "n_train_label": node["n_train_label"],
                    "n_affair": node["n_affair"],
                    "n_no_affair": node["n_no_affair"],
                    "affair_share_label": node["affair_share_label"],
                }
            ]
        left_steps = steps + [
            {
                "feature_label": node["feature_label"],
                "detail": node["left_label"],
            }
        ]
        right_steps = steps + [
            {
                "feature_label": node["feature_label"],
                "detail": node["right_label"],
            }
        ]
        return paths(node["left"], left_steps) + paths(node["right"], right_steps)

    root = walk(0)
    leaf_paths = paths(root, [])
    # The published depth-3 tree's third split does not change the predicted class.
    # Guard that description so a future refit cannot silently lie.
    third_changes_label = False
    for index in range(0, len(leaf_paths), 2):
        if leaf_paths[index]["predicted"] != leaf_paths[index + 1]["predicted"]:
            third_changes_label = True
    importances = [
        {
            "feature": name,
            "label": feature_labels[name],
            "importance": py(value),
            "importance_label": pct(value),
        }
        for name, value in zip(PREDICTORS, tree.feature_importances_)
    ]
    def collapsed(steps: list[dict]) -> str:
        last = {}
        order = []
        for step in steps:
            key = step["feature_label"]
            if key not in last:
                order.append(key)
            last[key] = step["detail"]
        return "; ".join(f"{key.lower()} {last[key]}" for key in order)

    affair_leaves = [p for p in leaf_paths if p["predicted"] == "Affair"]
    closest = min(
        leaf_paths,
        key=lambda p: abs((p["n_affair"] / p["n_train"]) - 0.5),
    )
    affair_bits = [
        (
            f"{collapsed(leaf['steps'])} "
            f"({leaf['n_train_label']} training rows, {leaf['affair_share_label']} affair)"
        )
        for leaf in affair_leaves
    ]
    rule = (
        "Predict an affair only when the marriage rating is 1, 2, or 3 and years married "
        "is 6, 9, 13, 16.5, or 23. Every other path is labeled no affair. "
        "The third split is still there. In this fit it changes how mixed the leaf is, "
        "and it does not change the predicted label. Age, children, religious commitment, "
        "and education are unused."
    )
    no_affair_n = closest["n_train"] - closest["n_affair"]
    caveat = (
        f"The two affair leaves are {affair_bits[0]}, and {affair_bits[1]}. "
        f"One no-affair leaf is nearly even — {collapsed(closest['steps'])} — with "
        f"{closest['n_affair']} affair and {no_affair_n} no affair out of "
        f"{closest['n_train_label']} training rows ({closest['affair_share_label']}). "
        f"The tree still calls that leaf no affair because that class is slightly larger."
    )
    return {
        "root": root,
        "paths": leaf_paths,
        "third_split_changes_prediction": third_changes_label,
        "importances": importances,
        "rule": rule,
        "caveat": caveat,
    }


def affairs_bins(df: pd.DataFrame) -> list[dict]:
    positive = df.loc[df["affairs"] > 0, "affairs"]
    edges = [0, 1, 2, 4, 8, np.inf]
    labels = ["Above 0 and under 1", "1 to under 2", "2 to under 4", "4 to under 8", "8 or more"]
    counts = pd.cut(positive, bins=edges, labels=labels, right=False).value_counts().sort_index()
    rows = [
        {
            "label": "Exactly 0",
            "n": int((df["affairs"] == 0).sum()),
        }
    ]
    for label, count in counts.items():
        rows.append({"label": str(label), "n": int(count)})
    total = sum(row["n"] for row in rows)
    peak = max(row["n"] for row in rows)
    for row in rows:
        row["n_label"] = comma(row["n"])
        row["share_label"] = pct(row["n"] / total)
        row["width"] = round(100 * row["n"] / peak, 2)
    return rows


def coefficient_rows(model: LogisticRegression) -> list[dict]:
    notes = {
        "rate_marriage": "Each one-point higher marriage rating multiplies the odds of an affair by {or_label}, holding the other five predictors fixed. Higher ratings go with lower odds.",
        "age": "Each one-unit increase in the age code multiplies the odds by {or_label}. That is a small decrease, and it is the partial association after years married is in the model. The raw age chart moves the other way.",
        "yrs_married": "Each one-unit increase in the years-married code multiplies the odds by {or_label}. Longer recorded marriages go with higher odds.",
        "children": "Each one-unit increase in the children code multiplies the odds by {or_label}. Once the other predictors are included, children add almost nothing.",
        "religious": "Each one-point higher religious commitment, on the 1–4 scale, multiplies the odds by {or_label}.",
        "educ": "Each one-point increase in the education code multiplies the odds by {or_label}. That is a flat line for practical purposes. The gaps between education codes are not all one year.",
    }
    labels = {
        "rate_marriage": "Marriage rating",
        "age": "Age",
        "yrs_married": "Years married",
        "children": "Children",
        "religious": "Religious commitment",
        "educ": "Education",
    }
    rows = []
    for name, coef in zip(PREDICTORS, model.coef_[0]):
        odds = float(np.exp(coef))
        or_label = f"{odds:.2f}"
        rows.append(
            {
                "feature": name,
                "label": labels[name],
                "coefficient": py(coef),
                "coefficient_label": f"{coef:+.3f}",
                "odds_ratio": py(odds),
                "odds_ratio_label": or_label,
                "plain": notes[name].format(or_label=or_label),
            }
        )
    intercept = float(model.intercept_[0])
    rows.append(
        {
            "feature": "intercept",
            "label": "Intercept",
            "coefficient": py(intercept),
            "coefficient_label": f"{intercept:+.3f}",
            "odds_ratio": py(np.exp(intercept)),
            "odds_ratio_label": f"{float(np.exp(intercept)):.1f}",
            "plain": "Log-odds when every predictor is 0. That person is not in this file — ratings, age, and the other codes all start above 0 — so the intercept is not a profile to interpret.",
        }
    )
    return rows


def build() -> dict:
    df = pd.read_csv(CSV_PATH)
    if df.shape[0] != 6366:
        raise SystemExit(f"Expected 6,366 rows, found {df.shape[0]}")
    religious_values = sorted(float(v) for v in df["religious"].unique())
    if religious_values != [1.0, 2.0, 3.0, 4.0]:
        raise SystemExit(f"Unexpected religious codes: {religious_values}")

    df["had_affair"] = (df["affairs"] > 0).astype(int)
    n = int(len(df))
    n_no = int((df["had_affair"] == 0).sum())
    n_yes = int((df["had_affair"] == 1).sum())
    positive = df.loc[df["affairs"] > 0, "affairs"]

    X = df[PREDICTORS]
    y = df["had_affair"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_frame = X_train.copy()
    train_frame["had_affair"] = y_train

    logistic = LogisticRegression(max_iter=10000)
    logistic.fit(X_train, y_train)
    logistic_test = class_block(y_test, logistic.predict(X_test))
    logistic_train_acc = float(logistic.score(X_train, y_train))

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    tree_test = class_block(y_test, tree.predict(X_test))
    tree_train_acc = float(tree.score(X_train, y_train))
    tree_export = export_tree(tree, train_frame, y_train)

    baseline = float((y_test == 0).mean())
    variables = []
    for spec in VARIABLES:
        rows = grouped(df, spec["id"])
        variables.append(
            {
                **spec,
                "rows": rows,
                "reading": reading_for(spec["id"], rows),
            }
        )

    params = logistic.get_params()
    payload = {
        "meta": {
            "sklearn": __import__("sklearn").__version__,
            "csv": "final_portfolio/final/_data/affairs.csv",
            "n": n,
            "n_label": comma(n),
            "n_no_affair": n_no,
            "n_no_affair_label": comma(n_no),
            "n_affair": n_yes,
            "n_affair_label": comma(n_yes),
            "no_affair_share_label": pct(n_no / n),
            "affair_share_label": pct(n_yes / n),
            "affairs_min": py(df["affairs"].min()),
            "affairs_max": py(df["affairs"].max()),
            "affairs_max_label": f"{float(df['affairs'].max()):.1f}",
            "affairs_mean_label": f"{float(df['affairs'].mean()):.2f}",
            "positive_median_label": f"{float(positive.median()):.2f}",
            "positive_mean_label": f"{float(positive.mean()):.2f}",
            "positive_n": int(len(positive)),
            "fractional_positive_n": int(((positive % 1) != 0).sum()),
            "religious_min": 1,
            "religious_max": 4,
            "one_hot_encoded": False,
            "logistic_constructor": "LogisticRegression(max_iter=10000)",
            "logistic_solver": params["solver"],
            "logistic_C": params["C"],
            "logistic_l1_ratio": params["l1_ratio"],
            "tree_constructor": "DecisionTreeClassifier(max_depth=3, random_state=42)",
            "tree_criterion": "gini",
            "split": "train_test_split(test_size=0.2, random_state=42, stratify=had_affair)",
            "predictors": PREDICTORS,
        },
        "affairs_bins": affairs_bins(df),
        "variables": variables,
        "split": {
            "n_train": int(len(X_train)),
            "n_train_label": comma(len(X_train)),
            "n_test": int(len(X_test)),
            "n_test_label": comma(len(X_test)),
            "test_no_affair": int((y_test == 0).sum()),
            "test_affair": int((y_test == 1).sum()),
            "test_no_affair_label": comma(int((y_test == 0).sum())),
            "test_affair_label": comma(int((y_test == 1).sum())),
            "baseline_accuracy": py(baseline),
            "baseline_accuracy_label": pct(baseline, 2),
            "baseline_correct_label": f"{comma(int((y_test == 0).sum()))} of {comma(len(y_test))}",
        },
        "logistic": {
            **logistic_test,
            "train_accuracy": py(logistic_train_acc),
            "train_accuracy_label": pct(logistic_train_acc, 2),
            "coefficients": coefficient_rows(logistic),
        },
        "tree": {
            **tree_test,
            "train_accuracy": py(tree_train_acc),
            "train_accuracy_label": pct(tree_train_acc, 2),
            "max_depth": 3,
            **tree_export,
        },
    }
    if payload["tree"]["third_split_changes_prediction"]:
        raise SystemExit("Tree wording assumes the third split does not change the predicted class.")
    return payload


def main() -> None:
    payload = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logistic = payload["logistic"]
    tree = payload["tree"]
    print(f"Wrote {OUT_PATH}")
    print(
        "logistic",
        logistic["accuracy_label"],
        "recall0",
        logistic["no_affair"]["recall_label"],
        "recall1",
        logistic["affair"]["recall_label"],
        "precision1",
        logistic["affair"]["precision_label"],
    )
    print(
        "tree",
        tree["accuracy_label"],
        "recall0",
        tree["no_affair"]["recall_label"],
        "recall1",
        tree["affair"]["recall_label"],
    )


if __name__ == "__main__":
    main()
