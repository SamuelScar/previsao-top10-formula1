#!/usr/bin/env python3
"""Analisa os resultados do treinamento e produz gráficos e explicações.

O script não treina modelos. Ele lê os quatro arquivos gerados pela etapa de
treinamento, confere se eles são consistentes entre si e cria um relatório em
Markdown acompanhado de gráficos em PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, matthews_corrcoef


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "resultados" / "treinamento"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "analise"

CHECKPOINTS = (10, 25, 50, 75)
FEATURE_SETS = ("race", "race_and_grid")
ALGORITHMS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svm",
)
BASELINE = "current_top_10"
METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "pr_auc")

ALGORITHM_LABELS = {
    "logistic_regression": "Regressão Logística",
    "decision_tree": "Árvore de Decisão",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    BASELINE: "Top 10 no checkpoint",
}
FEATURE_LABELS = {
    "race": "Dados da corrida",
    "race_and_grid": "Corrida + grid",
    "baseline": "Referência simples",
}
COLORS = {
    "logistic_regression": "#3366CC",
    "decision_tree": "#E67E22",
    "random_forest": "#2E8B57",
    "gradient_boosting": "#C0392B",
    "svm": "#7D3C98",
    BASELINE: "#333333",
}
MARKERS = {
    "logistic_regression": "o",
    "decision_tree": "s",
    "random_forest": "^",
    "gradient_boosting": "D",
    "svm": "P",
    BASELINE: "X",
}

GRAPH_FILES = (
    "01_f1_validacao.png",
    "02_f1_teste_2020.png",
    "03_maior_f1_vs_referencia.png",
    "04_efeito_grid_no_f1.png",
    "05_matrizes_confusao.png",
    "06_f1_por_corrida.png",
    "07_mcc_teste_2020.png",
    "08_pr_auc_teste_2020.png",
)
OBSOLETE_GRAPH_FILES = ("07_metricas_complementares_2020.png",)
REPORT_FILE = "relatorio_analise.md"


class AnalysisError(Exception):
    """Indica resultados ausentes ou incompatíveis com a análise."""


@dataclass(frozen=True)
class ResultFiles:
    validation: pd.DataFrame
    final: pd.DataFrame
    confusion: pd.DataFrame
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa os resultados dos classificadores e gera gráficos e "
            "um relatório em Markdown."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Pasta que contém os resultados do treinamento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta na qual a análise será gravada.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir os gráficos e o relatório já existentes.",
    )
    return parser.parse_args()


def read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise AnalysisError(f"Arquivo não encontrado: {path}")

    frame = pd.read_csv(
        path,
        dtype={"protocol_version": str, "input_sha256": str},
    )
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise AnalysisError(f"Colunas ausentes em {path.name}: {missing}")
    if frame.empty:
        raise AnalysisError(f"O arquivo {path.name} está vazio")
    return frame


def load_results(results_dir: Path) -> ResultFiles:
    common = {
        "checkpoint_percent",
        "feature_set",
        "algorithm",
        "protocol_version",
        "input_sha256",
    }
    validation = read_csv(
        results_dir / "melhores_resultados_validacao.csv",
        common
        | {
            "parameters",
            "accuracy_2018",
            "precision_2018",
            "recall_2018",
            "f1_2018",
            "accuracy_2019",
            "precision_2019",
            "recall_2019",
            "f1_2019",
            "mean_accuracy",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mcc_2018",
            "mcc_2019",
            "mean_mcc",
            "pr_auc_2018",
            "pr_auc_2019",
            "mean_pr_auc",
            "f1_gap",
        },
    )
    final = read_csv(
        results_dir / "avaliacao_final_2020.csv",
        common
        | {
            "parameters",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "pr_auc",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "examples",
        },
    )
    confusion = read_csv(
        results_dir / "matrizes_confusao.csv",
        common | {"actual_class", "predicted_class", "count"},
    )
    predictions = read_csv(
        results_dir / "detalhes" / "predicoes_finais_2020.csv",
        common
        | {
            "race_id",
            "year",
            "driver_id",
            "actual_top_10",
            "predicted_top_10",
            "top10_score",
            "correct",
        },
    )
    return ResultFiles(validation, final, confusion, predictions)


def scenario_keys() -> set[tuple[int, str, str]]:
    model_keys = {
        (checkpoint, feature_set, algorithm)
        for checkpoint in CHECKPOINTS
        for feature_set in FEATURE_SETS
        for algorithm in ALGORITHMS
    }
    baseline_keys = {
        (checkpoint, "baseline", BASELINE) for checkpoint in CHECKPOINTS
    }
    return model_keys | baseline_keys


def frame_keys(frame: pd.DataFrame) -> set[tuple[int, str, str]]:
    return {
        (int(row.checkpoint_percent), str(row.feature_set), str(row.algorithm))
        for row in frame.itertuples(index=False)
    }


def canonical_parameters(value: Any) -> str:
    return json.dumps(json.loads(str(value)), sort_keys=True, separators=(",", ":"))


def metrics_from_predictions(group: pd.DataFrame) -> dict[str, float | int]:
    actual = group["actual_top_10"].to_numpy(dtype=int)
    predicted = group["predicted_top_10"].to_numpy(dtype=int)
    top10_score = group["top10_score"].to_numpy(dtype=float)
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())
    total = len(actual)
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(actual, predicted),
        "pr_auc": average_precision_score(actual, top10_score),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "examples": total,
    }


def validate_results(data: ResultFiles) -> None:
    frames = {
        "validação": data.validation,
        "avaliação final": data.final,
        "matrizes": data.confusion,
        "previsões": data.predictions,
    }
    signatures: set[tuple[str, str]] = set()
    for name, frame in frames.items():
        protocols = set(frame["protocol_version"].astype(str))
        hashes = set(frame["input_sha256"].astype(str))
        if len(protocols) != 1 or len(hashes) != 1:
            raise AnalysisError(f"{name} contém mais de um protocolo ou base")
        signatures.add((next(iter(protocols)), next(iter(hashes))))
    if len(signatures) != 1:
        raise AnalysisError("Os arquivos não usam o mesmo protocolo e a mesma base")

    expected_keys = scenario_keys()
    key_columns = ["checkpoint_percent", "feature_set", "algorithm"]
    for name, frame in {
        "melhores_resultados_validacao.csv": data.validation,
        "avaliacao_final_2020.csv": data.final,
    }.items():
        if frame.duplicated(key_columns).any():
            raise AnalysisError(f"Há cenários duplicados em {name}")
        if frame_keys(frame) != expected_keys:
            raise AnalysisError(f"Os cenários estão incompletos em {name}")

    validation_models = data.validation[data.validation["algorithm"].ne(BASELINE)]
    final_models = data.final[data.final["algorithm"].ne(BASELINE)]
    joined = validation_models.merge(
        final_models,
        on=key_columns,
        suffixes=("_validation", "_final"),
        validate="one_to_one",
    )
    for row in joined.itertuples(index=False):
        if canonical_parameters(row.parameters_validation) != canonical_parameters(
            row.parameters_final
        ):
            raise AnalysisError(
                "A configuração usada em 2020 difere da escolhida na validação"
            )

    matrix_key = key_columns + ["actual_class", "predicted_class"]
    if data.confusion.duplicated(matrix_key).any():
        raise AnalysisError("Há células duplicadas nas matrizes de confusão")
    matrix_groups = {
        key: group
        for key, group in data.confusion.groupby(key_columns, sort=False)
    }
    prediction_groups = {
        key: group
        for key, group in data.predictions.groupby(key_columns, sort=False)
    }
    if set(matrix_groups) != expected_keys or set(prediction_groups) != expected_keys:
        raise AnalysisError("Matrizes ou previsões não cobrem todos os cenários")

    if set(data.predictions["year"].astype(int)) != {2020}:
        raise AnalysisError("As previsões finais devem conter somente 2020")
    if not set(data.predictions["actual_top_10"].astype(int)).issubset({0, 1}):
        raise AnalysisError("actual_top_10 possui valores diferentes de 0 e 1")
    if not set(data.predictions["predicted_top_10"].astype(int)).issubset({0, 1}):
        raise AnalysisError("predicted_top_10 possui valores diferentes de 0 e 1")
    if not np.isfinite(data.predictions["top10_score"].to_numpy(dtype=float)).all():
        raise AnalysisError("top10_score possui valor ausente ou infinito")
    recorded_correct = (
        data.predictions["correct"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    expected_correct = data.predictions["actual_top_10"].eq(
        data.predictions["predicted_top_10"]
    )
    if recorded_correct.isna().any() or not recorded_correct.equals(expected_correct):
        raise AnalysisError("A coluna correct não corresponde às previsões")
    prediction_identifier = key_columns + ["race_id", "driver_id"]
    if data.predictions.duplicated(prediction_identifier).any():
        raise AnalysisError("Há previsões duplicadas para o mesmo piloto")

    count_columns = {
        (0, 0): "true_negative",
        (0, 1): "false_positive",
        (1, 0): "false_negative",
        (1, 1): "true_positive",
    }
    final_by_key = {
        (int(row.checkpoint_percent), str(row.feature_set), str(row.algorithm)): row
        for row in data.final.itertuples(index=False)
    }
    for key in expected_keys:
        prediction_metrics = metrics_from_predictions(prediction_groups[key])
        final_row = final_by_key[key]
        for metric in METRICS:
            if not np.isclose(
                prediction_metrics[metric], float(getattr(final_row, metric)), atol=1e-12
            ):
                raise AnalysisError(f"A métrica {metric} não confere em {key}")
        if prediction_metrics["examples"] != int(final_row.examples):
            raise AnalysisError(f"A quantidade de exemplos não confere em {key}")

        matrix = matrix_groups[key]
        cells = {
            (int(row.actual_class), int(row.predicted_class)): int(row.count)
            for row in matrix.itertuples(index=False)
        }
        if set(cells) != set(count_columns):
            raise AnalysisError(f"Matriz de confusão incompleta em {key}")
        for cell, column in count_columns.items():
            if cells[cell] != int(getattr(final_row, column)):
                raise AnalysisError(f"Matriz de confusão divergente em {key}")


def label_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["algorithm_label"] = result["algorithm"].map(ALGORITHM_LABELS)
    result["feature_label"] = result["feature_set"].map(FEATURE_LABELS)
    return result


def highest_validation_results(validation: pd.DataFrame) -> pd.DataFrame:
    models = validation[validation["algorithm"].ne(BASELINE)].copy()
    models["feature_priority"] = models["feature_set"].map(
        {"race": 0, "race_and_grid": 1}
    )
    best = (
        models.sort_values(
            [
                "checkpoint_percent",
                "mean_f1",
                "f1_gap",
                "mean_accuracy",
                "feature_priority",
                "algorithm",
            ],
            ascending=[True, False, True, False, True, True],
        )
        .groupby("checkpoint_percent", sort=True, as_index=False)
        .head(1)
        .drop(columns="feature_priority")
    )
    baseline = validation[validation["algorithm"].eq(BASELINE)][
        ["checkpoint_percent", "mean_f1"]
    ].rename(columns={"mean_f1": "baseline_mean_f1"})
    best = best.merge(baseline, on="checkpoint_percent", validate="one_to_one")
    best["f1_delta_vs_baseline"] = best["mean_f1"] - best["baseline_mean_f1"]
    return label_results(best.sort_values("checkpoint_percent"))


def highest_final_results(
    final: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    models = final[final["algorithm"].ne(BASELINE)].copy()
    best = (
        models.sort_values(
            ["checkpoint_percent", "f1", "accuracy", "recall", "algorithm"],
            ascending=[True, False, False, False, True],
        )
        .groupby("checkpoint_percent", sort=True, as_index=False)
        .head(1)
    )
    baseline = final[final["algorithm"].eq(BASELINE)][
        ["checkpoint_percent", "f1"]
    ].rename(columns={"f1": "baseline_f1"})
    validation_metrics = validation[
        validation["algorithm"].ne(BASELINE)
    ][
        [
            "checkpoint_percent",
            "feature_set",
            "algorithm",
            "f1_2018",
            "f1_2019",
            "mean_f1",
            "mean_mcc",
            "mean_pr_auc",
            "f1_gap",
        ]
    ]
    best = best.merge(baseline, on="checkpoint_percent", validate="one_to_one")
    best = best.merge(
        validation_metrics,
        on=["checkpoint_percent", "feature_set", "algorithm"],
        validate="one_to_one",
    )
    best["f1_delta_vs_baseline"] = best["f1"] - best["baseline_f1"]
    best["display_f1_delta_vs_baseline"] = (
        best["f1"].round(3) - best["baseline_f1"].round(3)
    )
    best["f1_delta_vs_validation"] = best["f1"] - best["mean_f1"]
    return label_results(best.sort_values("checkpoint_percent"))


def strongest_validation_by_algorithm(validation: pd.DataFrame) -> pd.DataFrame:
    """Seleciona o maior F1 médio de validação de cada algoritmo."""

    models = validation[validation["algorithm"].isin(ALGORITHMS)].copy()
    models["feature_priority"] = models["feature_set"].map(
        {"race": 0, "race_and_grid": 1}
    )
    strongest = (
        models.sort_values(
            [
                "algorithm",
                "mean_f1",
                "f1_gap",
                "mean_accuracy",
                "feature_priority",
                "checkpoint_percent",
            ],
            ascending=[True, False, True, False, True, True],
        )
        .groupby("algorithm", sort=False, as_index=False)
        .head(1)
        .drop(columns="feature_priority")
    )
    algorithm_order = {algorithm: index for index, algorithm in enumerate(ALGORITHMS)}
    strongest["algorithm_order"] = strongest["algorithm"].map(algorithm_order)
    return label_results(
        strongest.sort_values("algorithm_order").drop(columns="algorithm_order")
    )


def strongest_final_by_algorithm(final: pd.DataFrame) -> pd.DataFrame:
    """Seleciona o maior F1 observado em 2020 para cada algoritmo."""

    models = final[final["algorithm"].isin(ALGORITHMS)].copy()
    models["feature_priority"] = models["feature_set"].map(
        {"race": 0, "race_and_grid": 1}
    )
    strongest = (
        models.sort_values(
            [
                "algorithm",
                "f1",
                "accuracy",
                "feature_priority",
                "checkpoint_percent",
            ],
            ascending=[True, False, False, True, True],
        )
        .groupby("algorithm", sort=False, as_index=False)
        .head(1)
        .drop(columns="feature_priority")
    )
    baseline = final[final["algorithm"].eq(BASELINE)][
        ["checkpoint_percent", "f1"]
    ].rename(columns={"f1": "baseline_f1"})
    strongest = strongest.merge(
        baseline,
        on="checkpoint_percent",
        validate="many_to_one",
    )
    strongest["f1_delta_vs_baseline"] = strongest["f1"] - strongest["baseline_f1"]
    strongest["display_f1_delta_vs_baseline"] = (
        strongest["f1"].round(3) - strongest["baseline_f1"].round(3)
    )

    algorithm_order = {algorithm: index for index, algorithm in enumerate(ALGORITHMS)}
    strongest["algorithm_order"] = strongest["algorithm"].map(algorithm_order)
    return label_results(
        strongest.sort_values("algorithm_order").drop(columns="algorithm_order")
    )


def grid_effect(final: pd.DataFrame) -> pd.DataFrame:
    models = final[final["algorithm"].isin(ALGORITHMS)]
    race = models[models["feature_set"].eq("race")]
    grid = models[models["feature_set"].eq("race_and_grid")]
    compared = race.merge(
        grid,
        on=["checkpoint_percent", "algorithm"],
        suffixes=("_race", "_grid"),
        validate="one_to_one",
    )
    for metric in METRICS:
        compared[f"{metric}_delta_grid"] = (
            compared[f"{metric}_grid"] - compared[f"{metric}_race"]
        )
    compared["algorithm_label"] = compared["algorithm"].map(ALGORITHM_LABELS)
    return compared.sort_values(["algorithm", "checkpoint_percent"])


def validation_grid_effect(validation: pd.DataFrame) -> pd.DataFrame:
    """Compara o F1 médio de corrida com corrida mais grid na validação."""

    models = validation[validation["algorithm"].isin(ALGORITHMS)]
    race = models[models["feature_set"].eq("race")]
    grid = models[models["feature_set"].eq("race_and_grid")]
    compared = race.merge(
        grid,
        on=["checkpoint_percent", "algorithm"],
        suffixes=("_race", "_grid"),
        validate="one_to_one",
    )
    compared["mean_f1_advantage_race"] = (
        compared["mean_f1_race"] - compared["mean_f1_grid"]
    )
    compared["algorithm_label"] = compared["algorithm"].map(ALGORITHM_LABELS)
    return compared.sort_values(["checkpoint_percent", "algorithm"])


def race_level_results(
    predictions: pd.DataFrame, best_final: pd.DataFrame
) -> pd.DataFrame:
    best_keys = {
        (int(row.checkpoint_percent), str(row.feature_set), str(row.algorithm))
        for row in best_final.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    group_columns = [
        "race_id",
        "checkpoint_percent",
        "feature_set",
        "algorithm",
    ]
    for key, group in predictions.groupby(group_columns, sort=False):
        race_id, checkpoint, feature_set, algorithm = key
        scenario = (int(checkpoint), str(feature_set), str(algorithm))
        if algorithm != BASELINE and scenario not in best_keys:
            continue
        metrics = metrics_from_predictions(group)
        records.append(
            {
                "race_id": int(race_id),
                "checkpoint_percent": int(checkpoint),
                "feature_set": str(feature_set),
                "algorithm": str(algorithm),
                "series": "Referência" if algorithm == BASELINE else "Maior F1 observado",
                **metrics,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["checkpoint_percent", "series", "race_id"]
    )


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, format="png", bbox_inches="tight")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def plot_f1_panels(
    frame: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    baseline = frame[frame["algorithm"].eq(BASELINE)].sort_values(
        "checkpoint_percent"
    )
    for axis, feature_set in zip(axes, FEATURE_SETS, strict=True):
        subset = frame[frame["feature_set"].eq(feature_set)]
        for algorithm in ALGORITHMS:
            values = subset[subset["algorithm"].eq(algorithm)].sort_values(
                "checkpoint_percent"
            )
            axis.plot(
                values["checkpoint_percent"],
                values[metric_column],
                marker=MARKERS[algorithm],
                linewidth=2,
                color=COLORS[algorithm],
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.plot(
            baseline["checkpoint_percent"],
            baseline[metric_column],
            marker=MARKERS[BASELINE],
            linewidth=2,
            linestyle="--",
            color=COLORS[BASELINE],
            label=ALGORITHM_LABELS[BASELINE],
        )
        axis.set_title(FEATURE_LABELS[feature_set])
        axis.set_xlabel("Momento da corrida")
        axis.set_xticks(CHECKPOINTS, [f"{value}%" for value in CHECKPOINTS])
        axis.set_ylim(0.75, 0.925)
        axis.set_yticks(np.arange(0.75, 0.926, 0.025))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.98, fontsize=14)
    figure.text(
        0.5,
        0.925,
        "Eixo Y ampliado: 0,750–0,925",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(top=0.68, bottom=0.12, left=0.07, right=0.98, wspace=0.12)
    save_figure(figure, output_path)


def plot_best_vs_baseline(best: pd.DataFrame, output_path: Path) -> None:
    positions = np.arange(len(best))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    model_bars = axis.bar(
        positions - width / 2,
        best["f1"],
        width,
        color="#3366CC",
        label="Maior F1 observado em 2020",
    )
    baseline_bars = axis.bar(
        positions + width / 2,
        best["baseline_f1"],
        width,
        color="#777777",
        label="Top 10 no checkpoint",
    )
    axis.bar_label(model_bars, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(baseline_bars, fmt="%.3f", padding=3, fontsize=9)
    for position, row in zip(positions, best.itertuples(index=False), strict=True):
        axis.text(
            position,
            max(row.f1, row.baseline_f1) + 0.075,
            f"Diferença {row.display_f1_delta_vs_baseline:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_title("Maior F1 observado em 2020 e referência simples")
    axis.set_ylabel("F1")
    axis.set_xlabel("Momento da corrida")
    axis.set_xticks(positions, [f"{value}%" for value in best["checkpoint_percent"]])
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_grid_effect(effect: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        effect.pivot(
            index="algorithm", columns="checkpoint_percent", values="f1_delta_grid"
        )
        .reindex(index=ALGORITHMS, columns=CHECKPOINTS)
    )
    values = pivot.to_numpy(dtype=float)
    limit = max(float(np.abs(values).max()), 0.01)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    image = axis.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:+.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_xticks(range(len(CHECKPOINTS)), [f"{value}%" for value in CHECKPOINTS])
    axis.set_yticks(
        range(len(ALGORITHMS)), [ALGORITHM_LABELS[value] for value in ALGORITHMS]
    )
    axis.set_xlabel("Momento da corrida")
    axis.set_title("Diferença de F1 ao adicionar a posição de largada")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("F1 com grid menos F1 sem grid")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_confusion_matrices(
    confusion: pd.DataFrame, best: pd.DataFrame, output_path: Path
) -> None:
    selected: list[tuple[pd.Series, np.ndarray]] = []
    for _, row in best.iterrows():
        matrix_rows = confusion[
            confusion["checkpoint_percent"].eq(row["checkpoint_percent"])
            & confusion["feature_set"].eq(row["feature_set"])
            & confusion["algorithm"].eq(row["algorithm"])
        ]
        matrix = (
            matrix_rows.pivot(
                index="actual_class", columns="predicted_class", values="count"
            )
            .reindex(index=[0, 1], columns=[0, 1])
            .to_numpy(dtype=int)
        )
        selected.append((row, matrix))

    maximum = max(int(matrix.max()) for _, matrix in selected)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    image = None
    for axis, (row, matrix) in zip(axes.flat, selected, strict=True):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for actual in range(2):
            for predicted in range(2):
                value = int(matrix[actual, predicted])
                color = "white" if value > maximum * 0.55 else "black"
                axis.text(
                    predicted,
                    actual,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                )
        axis.set_xticks([0, 1], ["Fora", "Top 10"])
        axis.set_yticks([0, 1], ["Fora", "Top 10"])
        axis.set_xlabel("Previsão")
        axis.set_ylabel("Resultado real")
        axis.set_title(
            f"{int(row['checkpoint_percent'])}%: {row['algorithm_label']}\n"
            f"{row['feature_label']}"
        )
    figure.suptitle("Matrizes de confusão dos maiores F1 observados em 2020", fontsize=14)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Pilotos")
    save_figure(figure, output_path)


def plot_complementary_metric_panels(
    frame: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Mostra uma métrica complementar em todos os cenários de 2020."""

    values = frame[metric_column].to_numpy(dtype=float)
    theoretical_minimum = -1.0 if metric_column == "mcc" else 0.0
    span = float(values.max() - values.min())
    padding = max(span * 0.15, 0.02)
    lower_limit = max(theoretical_minimum, float(values.min()) - padding)
    upper_limit = min(1.0, float(values.max()) + padding)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    baseline = frame[frame["algorithm"].eq(BASELINE)].sort_values(
        "checkpoint_percent"
    )
    for axis, feature_set in zip(axes, FEATURE_SETS, strict=True):
        subset = frame[frame["feature_set"].eq(feature_set)]
        for algorithm in ALGORITHMS:
            algorithm_values = subset[
                subset["algorithm"].eq(algorithm)
            ].sort_values("checkpoint_percent")
            axis.plot(
                algorithm_values["checkpoint_percent"],
                algorithm_values[metric_column],
                marker=MARKERS[algorithm],
                linewidth=2,
                color=COLORS[algorithm],
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.plot(
            baseline["checkpoint_percent"],
            baseline[metric_column],
            marker=MARKERS[BASELINE],
            linewidth=2,
            linestyle="--",
            color=COLORS[BASELINE],
            label=ALGORITHM_LABELS[BASELINE],
        )
        axis.set_title(FEATURE_LABELS[feature_set])
        axis.set_xlabel("Momento da corrida")
        axis.set_xticks(CHECKPOINTS, [f"{value}%" for value in CHECKPOINTS])
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)

    axes[0].set_ylabel(metric_label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.98, fontsize=14)
    axis_range = f"{lower_limit:.3f}–{upper_limit:.3f}".replace(".", ",")
    figure.text(
        0.5,
        0.925,
        f"Eixo Y ampliado: {axis_range}",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(top=0.68, bottom=0.12, left=0.07, right=0.98, wspace=0.12)
    save_figure(figure, output_path)


def plot_race_distribution(race_results: pd.DataFrame, output_path: Path) -> None:
    values: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    for index, checkpoint in enumerate(CHECKPOINTS, start=1):
        checkpoint_rows = race_results[
            race_results["checkpoint_percent"].eq(checkpoint)
        ]
        for offset, series, color in (
            (-0.2, "Referência", "#777777"),
            (0.2, "Maior F1 observado", "#3366CC"),
        ):
            values.append(
                checkpoint_rows[checkpoint_rows["series"].eq(series)]["f1"].to_numpy()
            )
            positions.append(index + offset)
            colors.append(color)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.32,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.8)
    axis.set_xticks(range(1, len(CHECKPOINTS) + 1), [f"{value}%" for value in CHECKPOINTS])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Momento da corrida")
    axis.set_ylabel("F1 calculado por corrida")
    axis.set_title("Variação do F1 entre as corridas de 2020")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axis.legend(
        handles=[
            Patch(facecolor="#3366CC", label="Maior F1 observado em 2020"),
            Patch(facecolor="#777777", label="Top 10 no checkpoint"),
        ],
        frameon=False,
        loc="lower right",
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def format_number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    decimals: set[str] | None = None,
) -> str:
    decimals = decimals or set()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row, strict=True):
            values.append(format_number(value) if column in decimals else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    validation_by_algorithm: pd.DataFrame,
    final_by_algorithm: pd.DataFrame,
    best_validation: pd.DataFrame,
    best_final: pd.DataFrame,
    validation_effect: pd.DataFrame,
    effect: pd.DataFrame,
    race_results: pd.DataFrame,
) -> str:
    validation_table = markdown_table(
        validation_by_algorithm,
        [
            "checkpoint_percent",
            "algorithm_label",
            "feature_label",
            "f1_2018",
            "f1_2019",
            "mean_f1",
            "f1_gap",
        ],
        [
            "Checkpoint",
            "Algoritmo",
            "Atributos",
            "F1 2018",
            "F1 2019",
            "F1 médio",
            "Diferença",
        ],
        {"f1_2018", "f1_2019", "mean_f1", "f1_gap"},
    )
    final_table = markdown_table(
        final_by_algorithm,
        [
            "checkpoint_percent",
            "algorithm_label",
            "feature_label",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "baseline_f1",
            "display_f1_delta_vs_baseline",
        ],
        [
            "Checkpoint",
            "Algoritmo",
            "Atributos",
            "Acurácia",
            "Precisão",
            "Recall",
            "F1",
            "F1 referência",
            "Diferença",
        ],
        {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "baseline_f1",
            "display_f1_delta_vs_baseline",
        },
    )
    validation_complementary_table = markdown_table(
        validation_by_algorithm,
        [
            "checkpoint_percent",
            "algorithm_label",
            "feature_label",
            "mean_mcc",
            "mean_pr_auc",
        ],
        ["Checkpoint", "Algoritmo", "Atributos", "MCC médio", "PR-AUC média"],
        {"mean_mcc", "mean_pr_auc"},
    )
    final_complementary_table = markdown_table(
        final_by_algorithm,
        [
            "checkpoint_percent",
            "algorithm_label",
            "feature_label",
            "mcc",
            "pr_auc",
        ],
        ["Checkpoint", "Algoritmo", "Atributos", "MCC", "PR-AUC"],
        {"mcc", "pr_auc"},
    )

    race_better_validation = validation_effect[
        validation_effect["mean_f1_advantage_race"].gt(1e-12)
    ].copy()
    race_better_validation["advantage_display"] = race_better_validation[
        "mean_f1_advantage_race"
    ].map(lambda value: format_number(value, 6))
    validation_grid_table = markdown_table(
        race_better_validation,
        [
            "checkpoint_percent",
            "algorithm_label",
            "mean_f1_race",
            "mean_f1_grid",
            "advantage_display",
        ],
        [
            "Checkpoint",
            "Algoritmo",
            "F1 corrida",
            "F1 corrida + grid",
            "Vantagem",
        ],
        {"mean_f1_race", "mean_f1_grid"},
    )
    validation_race_better = len(race_better_validation)
    validation_grid_tied = int(
        np.isclose(
            validation_effect["mean_f1_advantage_race"], 0, atol=1e-12
        ).sum()
    )
    validation_grid_better = (
        len(validation_effect) - validation_race_better - validation_grid_tied
    )
    tiny_validation_advantages = int(
        race_better_validation["mean_f1_advantage_race"].lt(0.001).sum()
    )

    best_overall = best_final.loc[best_final["f1"].idxmax()]
    largest_gain = best_final.loc[best_final["f1_delta_vs_baseline"].idxmax()]
    grid_improved = int(effect["f1_delta_grid"].gt(0).sum())
    grid_tied = int(np.isclose(effect["f1_delta_grid"], 0, atol=1e-12).sum())
    grid_worsened = len(effect) - grid_improved - grid_tied
    grid_mean = float(effect["f1_delta_grid"].mean())

    validation_keys = {
        int(row.checkpoint_percent): (row.algorithm, row.feature_set)
        for row in best_validation.itertuples(index=False)
    }
    final_keys = {
        int(row.checkpoint_percent): (row.algorithm, row.feature_set)
        for row in best_final.itertuples(index=False)
    }
    same_leaders = sum(
        validation_keys[checkpoint] == final_keys[checkpoint]
        for checkpoint in CHECKPOINTS
    )

    race_summary = (
        race_results.groupby(["checkpoint_percent", "series"], as_index=False)
        .agg(
            races=("race_id", "nunique"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            minimum_f1=("f1", "min"),
            maximum_f1=("f1", "max"),
        )
        .sort_values(["checkpoint_percent", "series"])
    )
    race_table = markdown_table(
        race_summary,
        [
            "checkpoint_percent",
            "series",
            "races",
            "mean_f1",
            "median_f1",
            "minimum_f1",
            "maximum_f1",
        ],
        ["Checkpoint", "Série", "Corridas", "Média", "Mediana", "Mínimo", "Máximo"],
        {"mean_f1", "median_f1", "minimum_f1", "maximum_f1"},
    )

    progression = best_final.sort_values("checkpoint_percent")["f1"].to_numpy()
    progression_text = (
        "O maior F1 observado cresceu em todos os checkpoints analisados."
        if np.all(np.diff(progression) >= -1e-12)
        else "O maior F1 observado não cresceu de forma contínua entre todos os checkpoints."
    )
    timestamp = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S %Z")

    return f"""# Análise dos resultados dos classificadores

Relatório gerado automaticamente em {timestamp}.

## Escopo

Esta análise usa as melhores configurações definidas nas validações de 2018 e
2019 e os resultados obtidos no teste final de 2020. O script não treina nem
reconfigura os modelos. O maior resultado de 2020 é apresentado de maneira
descritiva e não é usado para alterar as configurações escolhidas anteriormente.

Todos os arquivos possuem o mesmo protocolo e a mesma assinatura da base. As
métricas do teste final foram recalculadas a partir das previsões individuais e
coincidiram com a avaliação consolidada. As matrizes de confusão também
coincidiram com as contagens presentes na avaliação final.

## Validação temporal

A tabela apresenta os cinco algoritmos. Para cada um, mostra o checkpoint e o
conjunto de atributos em que ele obteve seu maior F1 médio nas validações. Cada
linha usa uma configuração escolhida sem consultar 2020.

{validation_table}

![F1 nas validações](graficos/01_f1_validacao.png)

O gráfico mantém separados os modelos que usam apenas informações da corrida e
os que também usam a posição de largada. A linha tracejada representa a regra
simples de considerar que o top 10 no checkpoint permanecerá no top 10 final.

## Teste final de 2020

A tabela apresenta uma linha para cada algoritmo. O resultado mostrado é o
maior F1 observado para aquele algoritmo em 2020, acompanhado do checkpoint, do
conjunto de atributos e das demais métricas da mesma avaliação.

{final_table}

![F1 no teste final](graficos/02_f1_teste_2020.png)

![Maior F1 e referência](graficos/03_maior_f1_vs_referencia.png)

{progression_text} O maior valor foi {format_number(best_overall['f1'])}, aos
{int(best_overall['checkpoint_percent'])}% da corrida, obtido por
{best_overall['algorithm_label']} com {best_overall['feature_label'].lower()}.

O maior ganho sobre a referência simples, considerando os valores exibidos com
três casas decimais, foi de
{format_number(largest_gain['display_f1_delta_vs_baseline'])} no checkpoint de
{int(largest_gain['checkpoint_percent'])}%.

Os maiores F1 observados na validação e no teste final apontaram para a mesma
combinação de algoritmo e atributos em {same_leaders} dos {len(CHECKPOINTS)}
checkpoints. Diferenças entre essas etapas são esperadas porque as temporadas
avaliadas não são as mesmas.

## Métricas complementares

As tabelas abaixo usam os mesmos cenários escolhidos pelo maior F1 de cada
algoritmo. MCC e PR-AUC não foram usados para trocar configurações.

### Validações de 2018 e 2019

{validation_complementary_table}

### Teste final de 2020

{final_complementary_table}

![MCC no teste final](graficos/07_mcc_teste_2020.png)

![PR-AUC no teste final](graficos/08_pr_auc_teste_2020.png)

Os dois gráficos apresentam todos os checkpoints, os cinco algoritmos e a
referência simples. Os painéis mantêm separados os modelos que usam somente os
dados da corrida e os que também usam a posição de largada.

O MCC avalia conjuntamente os quatro tipos de resultado da matriz de confusão.
A PR-AUC, calculada como Average Precision, avalia a ordenação produzida pelo
escore de top 10 em diferentes limites de decisão.

## Efeito da posição de largada

### Validações de 2018 e 2019

Na média das duas validações, usar somente informações da corrida superou
corrida mais grid em {validation_race_better} das {len(validation_effect)}
comparações:

{validation_grid_table}

Corrida mais grid foi melhor em {validation_grid_better} comparações e houve
empate em {validation_grid_tied}. Em {tiny_validation_advantages} dos casos
favoráveis à base sem grid, a vantagem foi inferior a 0,001. Diferenças tão
pequenas devem ser interpretadas como resultados praticamente equivalentes.

### Teste final de 2020

![Efeito do grid](graficos/04_efeito_grid_no_f1.png)

Adicionar as informações do grid aumentou o F1 em {grid_improved} das
{len(effect)} comparações, manteve o resultado em {grid_tied} e reduziu em
{grid_worsened}. A diferença média foi {format_number(grid_mean)}. Esses valores
descrevem associação preditiva. Eles não demonstram que a posição de largada
causa o resultado final.

## Erros dos modelos

![Matrizes de confusão](graficos/05_matrizes_confusao.png)

Cada matriz separa quatro situações: acertos fora do top 10, falsos positivos,
falsos negativos e acertos no top 10. Falso positivo significa prever top 10
para um piloto que terminou fora. Falso negativo significa deixar fora da
previsão um piloto que terminou no top 10.

## Variação entre corridas

{race_table}

![F1 por corrida](graficos/06_f1_por_corrida.png)

As métricas consolidadas juntam todas as previsões de 2020. O gráfico por
corrida mostra que o desempenho não foi idêntico em todos os eventos. A linha
central de cada caixa é a mediana, e o ponto branco representa a média.

## Resposta à pergunta de pesquisa

Os resultados permitem comparar quanto a previsão melhora entre 10%, 25%, 50%
e 75% da corrida. O checkpoint com maior F1 observado em 2020 foi o de
{int(best_overall['checkpoint_percent'])}%. Entretanto, a metodologia ainda não
definiu um valor mínimo de F1 que signifique “confiável”. Portanto, o relatório
identifica o momento de melhor desempenho e a evolução das métricas, mas não
declara automaticamente o primeiro checkpoint confiável.

## Significado das métricas

- **Acurácia:** proporção de todas as previsões que estavam corretas.
- **Precisão:** entre os pilotos previstos no top 10, quantos realmente terminaram nele.
- **Recall:** entre os pilotos que terminaram no top 10, quantos foram encontrados pelo modelo.
- **F1:** equilíbrio entre precisão e recall. É a principal métrica desta análise.
- **MCC:** resume os quatro componentes da matriz de confusão; varia de -1 a 1.
- **PR-AUC:** resume a curva precisão-recall construída com `top10_score`; quanto maior, melhor a ordenação dos pilotos.

## Limitações para a interpretação

- O teste final contém somente a temporada de 2020.
- Pilotos da mesma corrida não devem ser tratados como observações totalmente independentes.
- Destacar o maior resultado de 2020 é uma comparação descritiva, não uma nova seleção de modelo.
- A classificação é feita por piloto e não obriga cada modelo a prever exatamente dez pilotos no top 10.
- Diferenças pequenas de F1 devem ser interpretadas com cautela, principalmente sem intervalo de confiança.
- `top10_score` é um escore de ordenação. Ele só representa probabilidade nos classificadores que oferecem `predict_proba`; no SVM é a função de decisão.
"""


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_outputs_available(output_dir: Path, overwrite: bool) -> None:
    expected = [output_dir / REPORT_FILE]
    expected.extend(output_dir / "graficos" / name for name in GRAPH_FILES)
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(output_dir)) for path in existing)
        raise AnalysisError(
            f"A análise já possui saídas ({names}). Use --overwrite para substituí-las"
        )


def remove_obsolete_outputs(output_dir: Path) -> None:
    """Remove gráficos substituídos somente após a nova análise ser concluída."""

    for name in OBSOLETE_GRAPH_FILES:
        (output_dir / "graficos" / name).unlink(missing_ok=True)


def run_analysis(results_dir: Path, output_dir: Path, overwrite: bool) -> None:
    data = load_results(results_dir)
    validate_results(data)
    ensure_outputs_available(output_dir, overwrite)

    best_validation = highest_validation_results(data.validation)
    best_final = highest_final_results(data.final, data.validation)
    validation_by_algorithm = strongest_validation_by_algorithm(data.validation)
    final_by_algorithm = strongest_final_by_algorithm(data.final)
    validation_effect = validation_grid_effect(data.validation)
    effect = grid_effect(data.final)
    race_results = race_level_results(data.predictions, best_final)

    configure_plots()
    graphs_dir = output_dir / "graficos"
    plot_f1_panels(
        data.validation,
        "mean_f1",
        "F1 médio nas validações de 2018 e 2019",
        graphs_dir / GRAPH_FILES[0],
    )
    plot_f1_panels(
        data.final,
        "f1",
        "F1 no teste final de 2020",
        graphs_dir / GRAPH_FILES[1],
    )
    plot_best_vs_baseline(best_final, graphs_dir / GRAPH_FILES[2])
    plot_grid_effect(effect, graphs_dir / GRAPH_FILES[3])
    plot_confusion_matrices(data.confusion, best_final, graphs_dir / GRAPH_FILES[4])
    plot_race_distribution(race_results, graphs_dir / GRAPH_FILES[5])
    plot_complementary_metric_panels(
        data.final,
        "mcc",
        "MCC",
        "MCC por checkpoint no teste final de 2020",
        graphs_dir / GRAPH_FILES[6],
    )
    plot_complementary_metric_panels(
        data.final,
        "pr_auc",
        "PR-AUC",
        "PR-AUC por checkpoint no teste final de 2020",
        graphs_dir / GRAPH_FILES[7],
    )

    report = build_report(
        validation_by_algorithm,
        final_by_algorithm,
        best_validation,
        best_final,
        validation_effect,
        effect,
        race_results,
    )
    write_text_atomic(report, output_dir / REPORT_FILE)
    remove_obsolete_outputs(output_dir)


def main() -> int:
    args = parse_args()
    try:
        run_analysis(args.results_dir, args.output_dir, args.overwrite)
        print(f"Análise gravada em: {args.output_dir}")
        print(f"Relatório: {args.output_dir / REPORT_FILE}")
        print(f"Gráficos: {args.output_dir / 'graficos'}")
        return 0
    except AnalysisError as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Falha inesperada: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
