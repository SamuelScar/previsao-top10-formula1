#!/usr/bin/env python3
"""Treina e avalia classificadores para prever o top 10 da Fórmula 1.

A execução padrão valida configurações em 2018 e 2019. ``--stage-final``
reutiliza as escolhas salvas, treina com 2017-2019 e avalia o holdout de 2020.
``--stage-check`` confere ambiente, entrada e configurações sem treinar. MCC e
PR-AUC são métricas complementares; a seleção continua baseada somente no F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dados" / "gerados" / "base_checkpoints.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "resultados" / "treinamento"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "treinamento"

RANDOM_SEED = 42
PROTOCOL_VERSION = "1.1"
EXPECTED_VERSIONS = {
    "numpy": "2.3.3",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.1",
}
CHECKPOINTS = (10, 25, 50, 75)
EXPECTED_RACE_COUNT = 79
EXPECTED_RACE_CHECKPOINT_GROUPS = 316
TARGET = "top_10_final"
LOGGER = logging.getLogger("treinamento")

IDENTIFIER_COLUMNS = ("race_id", "year", "driver_id", "checkpoint_percent")
RACE_FEATURES = (
    "current_position",
    "laps_behind_checkpoint_leader",
    "gap_to_lap_leader_ms",
    "relative_pace_to_checkpoint_leader",
)
GRID_FEATURES = ("grid_position", "non_standard_grid_start")
FEATURE_SETS = {
    "race": RACE_FEATURES,
    "race_and_grid": RACE_FEATURES + GRID_FEATURES,
}

VALIDATION_SPLITS = (
    ((2017,), 2018),
    ((2017, 2018), 2019),
)

VALIDATION_FILE = "melhores_resultados_validacao.csv"
SELECTED_CONFIGS_FILE = VALIDATION_FILE
FINAL_EVALUATION_FILE = "avaliacao_final_2020.csv"
CONFUSION_MATRICES_FILE = "matrizes_confusao.csv"
FINAL_PREDICTIONS_FILE = Path("detalhes") / "predicoes_finais_2020.csv"

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "mcc", "pr_auc")


def combinations(**parameters: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Cria uma grade pequena preservando a ordem de simplicidade definida."""

    items = list(parameters.items())
    configurations: list[dict[str, Any]] = [{}]
    for name, values in items:
        configurations = [
            {**configuration, name: value}
            for configuration in configurations
            for value in values
        ]
    return tuple(configurations)


ALGORITHM_CONFIGS = {
    "logistic_regression": combinations(C=(0.1, 1.0, 10.0)),
    "decision_tree": combinations(
        max_depth=(3, 5, 8),
        min_samples_leaf=(15, 5),
    ),
    "random_forest": combinations(
        max_depth=(5, 10, None),
        min_samples_leaf=(5, 1),
    ),
    "gradient_boosting": combinations(
        n_estimators=(50, 100),
        learning_rate=(0.05, 0.1),
        max_depth=(1, 2),
    ),
    "svm": (
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 10.0},
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    ),
}

SCALED_ALGORITHMS = {"logistic_regression", "svm"}


@dataclass(frozen=True)
class SelectedConfiguration:
    """Configuração escolhida para um algoritmo em um cenário de previsão."""

    checkpoint: int
    feature_set: str
    algorithm: str
    config_index: int
    parameters: dict[str, Any]
    f1_2018: float
    f1_2019: float
    mean_f1: float
    f1_gap: float


class TrainingError(Exception):
    """Indica uma entrada ou estado incompatível com o protocolo definido."""


class RoundedMedianGridImputer(BaseEstimator, TransformerMixin):
    """Preenche grid ausente com a mediana do treino arredondada para inteiro.

    Metades são arredondadas para cima: uma mediana 10,5 produz a posição 11.
    O indicador ``non_standard_grid_start`` permanece como atributo separado.
    """

    grid_column = "grid_position"

    def fit(self, values: pd.DataFrame, y: Any = None) -> RoundedMedianGridImputer:
        del y
        frame = self._require_dataframe(values)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.fill_value_: int | None = None

        if self.grid_column in frame.columns:
            median = frame[self.grid_column].median(skipna=True)
            if pd.isna(median):
                raise TrainingError(
                    "Não foi possível calcular a mediana de grid_position no treino"
                )
            self.fill_value_ = int(math.floor(float(median) + 0.5))
        return self

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        frame = self._require_dataframe(values).copy()
        if not hasattr(self, "feature_names_in_"):
            raise TrainingError("O preenchimento de grid ainda não foi ajustado")
        if tuple(frame.columns) != tuple(self.feature_names_in_):
            raise TrainingError("As colunas recebidas diferem das usadas no treino")

        if self.grid_column in frame.columns:
            frame[self.grid_column] = frame[self.grid_column].fillna(self.fill_value_)
        return frame

    @staticmethod
    def _require_dataframe(values: Any) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame):
            raise TrainingError("O pipeline esperava atributos em um DataFrame")
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida a base, ajusta cinco classificadores e registra resultados "
            "por checkpoint."
        )
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--stage-check",
        action="store_true",
        help="Valida ambiente, base e configurações sem treinar classificadores.",
    )
    stages.add_argument(
        "--stage-final",
        action="store_true",
        help="Usa as configurações salvas e executa o teste final em 2020.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir atomicamente os resultados da etapa escolhida.",
    )
    args = parser.parse_args()
    if args.stage_check:
        args.stage = "check"
    elif args.stage_final:
        args.stage = "final"
    else:
        args.stage = "validation"
    return args


def configure_logging(log_dir: Path, stage: str) -> Path:
    """Cria um log próprio para cada execução e mantém a saída no terminal."""

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"treinamento_{stage}_{timestamp}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    for handler in (
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return log_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Carrega a base final e confere as premissas usadas no treinamento."""

    if not path.is_file():
        raise TrainingError(f"Base de treinamento não encontrada: {path}")

    frame = pd.read_csv(path)
    required_columns = set(IDENTIFIER_COLUMNS + RACE_FEATURES + GRID_FEATURES)
    required_columns.add(TARGET)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise TrainingError(f"Colunas obrigatórias ausentes: {missing_columns}")

    frame = frame[list(IDENTIFIER_COLUMNS + RACE_FEATURES + GRID_FEATURES + (TARGET,))]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    for column in frame.columns:
        observed = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise TrainingError(f"A coluna {column} contém valor infinito")

    required_values = list(IDENTIFIER_COLUMNS + RACE_FEATURES) + [
        "non_standard_grid_start",
        TARGET,
    ]
    if frame[required_values].isna().any().any():
        raise TrainingError("Há valores ausentes fora de grid_position")

    if set(frame["year"].unique()) != {2017, 2018, 2019, 2020}:
        raise TrainingError("A base deve conter exatamente as temporadas 2017-2020")
    if set(frame["checkpoint_percent"].unique()) != set(CHECKPOINTS):
        raise TrainingError("A base não contém exatamente os quatro checkpoints")
    if frame.duplicated(list(IDENTIFIER_COLUMNS)).any():
        raise TrainingError("Há linhas duplicadas por corrida, piloto e checkpoint")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise TrainingError("top_10_final deve conter somente 0 e 1")
    if not set(frame["non_standard_grid_start"].unique()).issubset({0, 1}):
        raise TrainingError("non_standard_grid_start deve conter somente 0 e 1")
    if frame["race_id"].nunique() != EXPECTED_RACE_COUNT:
        raise TrainingError(
            f"Esperadas {EXPECTED_RACE_COUNT} corridas, encontradas "
            f"{frame['race_id'].nunique()}"
        )
    if frame.groupby("race_id")["year"].nunique().ne(1).any():
        raise TrainingError("Uma mesma corrida aparece associada a mais de um ano")

    if not frame["current_position"].mod(1).eq(0).all():
        raise TrainingError("current_position deve conter somente posições inteiras")
    position_summary = frame.groupby(
        ["race_id", "checkpoint_percent"]
    )["current_position"].agg(["count", "nunique", "min", "max"])
    continuous_positions = (
        position_summary["count"].eq(position_summary["nunique"])
        & position_summary["min"].eq(1)
        & position_summary["max"].eq(position_summary["count"])
    )
    if not continuous_positions.all():
        raise TrainingError(
            "current_position não é única e contínua em todos os checkpoints"
        )

    if frame[
        ["laps_behind_checkpoint_leader", "gap_to_lap_leader_ms"]
    ].lt(0).any().any():
        raise TrainingError("Atrasos em relação ao líder não podem ser negativos")
    if frame["relative_pace_to_checkpoint_leader"].le(0).any():
        raise TrainingError("O ritmo relativo deve ser maior que zero")
    if frame["grid_position"].dropna().le(0).any():
        raise TrainingError("grid_position deve ser positiva quando informada")

    grid_missing = frame["grid_position"].isna()
    non_standard = frame["non_standard_grid_start"].eq(1)
    if not grid_missing.equals(non_standard):
        raise TrainingError(
            "Ausências em grid_position não coincidem com o indicador de grid"
        )

    positive_counts = frame.groupby(
        ["race_id", "checkpoint_percent"], sort=False
    )[TARGET].sum()
    if len(positive_counts) != EXPECTED_RACE_CHECKPOINT_GROUPS:
        raise TrainingError(
            f"Esperados {EXPECTED_RACE_CHECKPOINT_GROUPS} grupos de corrida e "
            f"checkpoint, encontrados {len(positive_counts)}"
        )
    if not positive_counts.eq(10).all():
        raise TrainingError("Cada corrida e checkpoint deve possuir dez positivos")

    classes_by_period = frame.groupby(["year", "checkpoint_percent"])[TARGET].nunique()
    if len(classes_by_period) != 16 or not classes_by_period.eq(2).all():
        raise TrainingError("Todo ano e checkpoint deve conter as duas classes")

    return frame.sort_values(
        ["year", "race_id", "checkpoint_percent", "driver_id"]
    ).reset_index(drop=True)


def build_classifier(algorithm: str, parameters: dict[str, Any]) -> Any:
    """Instancia um classificador com parâmetros fixos e reproduzíveis."""

    if algorithm == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            l1_ratio=0.0,
            max_iter=1_000,
            random_state=RANDOM_SEED,
            **parameters,
        )
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_SEED, **parameters)
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **parameters,
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_SEED, **parameters)
    if algorithm == "svm":
        return SVC(**parameters)
    raise TrainingError(f"Algoritmo desconhecido: {algorithm}")


def build_pipeline(algorithm: str, parameters: dict[str, Any]) -> Pipeline:
    """Mantém preenchimento, escala e classificador dentro do mesmo ajuste."""

    steps: list[tuple[str, Any]] = [
        ("grid_imputer", RoundedMedianGridImputer()),
    ]
    if algorithm in SCALED_ALGORITHMS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", build_classifier(algorithm, parameters)))
    return Pipeline(steps)


def validate_environment_and_configuration(
    frame: pd.DataFrame, input_hash: str, results_dir: Path
) -> None:
    """Confere o ambiente e os pipelines sem ajustar classificadores."""

    if sys.version_info < (3, 11):
        raise TrainingError("O treinamento exige Python 3.11 ou superior")

    installed_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    if installed_versions != EXPECTED_VERSIONS:
        raise TrainingError(
            "Versões incompatíveis. Esperadas "
            f"{EXPECTED_VERSIONS}; instaladas {installed_versions}"
        )

    for algorithm, configurations in ALGORITHM_CONFIGS.items():
        for parameters in configurations:
            build_pipeline(algorithm, parameters).get_params(deep=True)

    configuration_count = sum(map(len, ALGORITHM_CONFIGS.values()))
    scenario_count = len(CHECKPOINTS) * len(FEATURE_SETS) * len(ALGORITHM_CONFIGS)
    validation_fit_count = (
        configuration_count
        * len(VALIDATION_SPLITS)
        * len(CHECKPOINTS)
        * len(FEATURE_SETS)
    )
    LOGGER.info("Protocolo: %s | Base SHA-256: %s", PROTOCOL_VERSION, input_hash)
    LOGGER.info("Versões confirmadas: %s", installed_versions)
    LOGGER.info(
        "Cenários: %d | Configurações: %d | Treinos de validação: %d",
        scenario_count,
        configuration_count,
        validation_fit_count,
    )
    LOGGER.info(
        "Base validada: %d linhas, %d corridas e %d grupos",
        len(frame),
        frame["race_id"].nunique(),
        frame.groupby(["race_id", "checkpoint_percent"]).ngroups,
    )

    expected_outputs = (
        VALIDATION_FILE,
        FINAL_EVALUATION_FILE,
        CONFUSION_MATRICES_FILE,
        FINAL_PREDICTIONS_FILE,
    )
    existing = [name for name in expected_outputs if (results_dir / name).exists()]
    LOGGER.info("Resultados existentes: %s", existing or "nenhum")
    LOGGER.info("Pré-verificação concluída; nenhum classificador foi ajustado")


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "mcc": matthews_corrcoef(actual, predicted),
        "pr_auc": average_precision_score(actual, scores),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "examples": len(actual),
        "negative_examples": int(actual.eq(0).sum()),
        "positive_examples": int(actual.eq(1).sum()),
    }


def configuration_json(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def train_and_predict(
    algorithm: str,
    parameters: dict[str, Any],
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int | None]:
    pipeline = build_pipeline(algorithm, parameters)
    pipeline.fit(train[list(features)], train[TARGET])
    evaluation_features = evaluation[list(features)]
    predicted = pipeline.predict(evaluation_features)
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(evaluation_features)[:, 1]
    else:
        scores = pipeline.decision_function(evaluation_features)
    fill_value = pipeline.named_steps["grid_imputer"].fill_value_
    return predicted, np.asarray(scores, dtype=float), fill_value


def validation_records(
    frame: pd.DataFrame, input_hash: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_scenarios = len(CHECKPOINTS) * len(FEATURE_SETS) * len(ALGORITHM_CONFIGS)
    scenario_number = 0

    for checkpoint in CHECKPOINTS:
        checkpoint_data = frame[frame["checkpoint_percent"].eq(checkpoint)]
        for feature_set, features in FEATURE_SETS.items():
            for algorithm, configurations in ALGORITHM_CONFIGS.items():
                scenario_number += 1
                started_at = perf_counter()
                LOGGER.info(
                    "Validação [%d/%d] | checkpoint=%d | atributos=%s | "
                    "algoritmo=%s | configurações=%d",
                    scenario_number,
                    total_scenarios,
                    checkpoint,
                    feature_set,
                    algorithm,
                    len(configurations),
                )
                for config_index, parameters in enumerate(configurations):
                    for train_years, validation_year in VALIDATION_SPLITS:
                        train = checkpoint_data[
                            checkpoint_data["year"].isin(train_years)
                        ]
                        validation = checkpoint_data[
                            checkpoint_data["year"].eq(validation_year)
                        ]
                        predicted, scores, fill_value = train_and_predict(
                            algorithm, parameters, train, validation, features
                        )
                        records.append(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "input_sha256": input_hash,
                                "checkpoint_percent": checkpoint,
                                "feature_set": feature_set,
                                "algorithm": algorithm,
                                "config_index": config_index,
                                "parameters": configuration_json(parameters),
                                "train_years": "-".join(map(str, train_years)),
                                "evaluation_year": validation_year,
                                "grid_fill_value": fill_value,
                                **calculate_metrics(
                                    validation[TARGET], predicted, scores
                                ),
                            }
                        )
                LOGGER.info(
                    "Validação [%d/%d] concluída em %.1fs",
                    scenario_number,
                    total_scenarios,
                    perf_counter() - started_at,
                )

        for _train_years, validation_year in VALIDATION_SPLITS:
            validation = checkpoint_data[checkpoint_data["year"].eq(validation_year)]
            predicted = validation["current_position"].le(10).astype(int).to_numpy()
            scores = -validation["current_position"].to_numpy(dtype=float)
            records.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "input_sha256": input_hash,
                    "checkpoint_percent": checkpoint,
                    "feature_set": "baseline",
                    "algorithm": "current_top_10",
                    "config_index": -1,
                    "parameters": "{}",
                    "train_years": "not_applicable",
                    "evaluation_year": validation_year,
                    "grid_fill_value": None,
                    **calculate_metrics(validation[TARGET], predicted, scores),
                }
            )
    return records


def select_configurations(
    records: Iterable[dict[str, Any]], input_hash: str
) -> list[SelectedConfiguration]:
    grouped: defaultdict[tuple[int, str, str, int], list[dict[str, Any]]]
    grouped = defaultdict(list)
    for record in records:
        if record["algorithm"] == "current_top_10":
            continue
        key = (
            record["checkpoint_percent"],
            record["feature_set"],
            record["algorithm"],
            record["config_index"],
        )
        grouped[key].append(record)

    scenarios: defaultdict[tuple[int, str, str], list[SelectedConfiguration]]
    scenarios = defaultdict(list)
    for (checkpoint, feature_set, algorithm, config_index), results in grouped.items():
        if len(results) != len(VALIDATION_SPLITS):
            raise TrainingError("Configuração sem as duas validações temporais")
        by_year = {result["evaluation_year"]: result for result in results}
        if set(by_year) != {2018, 2019}:
            raise TrainingError("Configuração sem resultados distintos em 2018 e 2019")
        f1_2018 = float(by_year[2018]["f1"])
        f1_2019 = float(by_year[2019]["f1"])
        selection = SelectedConfiguration(
            checkpoint=checkpoint,
            feature_set=feature_set,
            algorithm=algorithm,
            config_index=config_index,
            parameters=json.loads(results[0]["parameters"]),
            f1_2018=f1_2018,
            f1_2019=f1_2019,
            mean_f1=fmean((f1_2018, f1_2019)),
            f1_gap=abs(f1_2018 - f1_2019),
        )
        scenarios[(checkpoint, feature_set, algorithm)].append(selection)

    selected: list[SelectedConfiguration] = []
    for candidates in scenarios.values():
        candidates.sort(
            key=lambda item: (-item.mean_f1, item.f1_gap, item.config_index)
        )
        selected.append(candidates[0])

    expected = len(CHECKPOINTS) * len(FEATURE_SETS) * len(ALGORITHM_CONFIGS)
    if len(selected) != expected:
        raise TrainingError(
            f"Esperadas {expected} configurações selecionadas, obtidas {len(selected)}"
        )

    selected.sort(key=lambda item: (item.checkpoint, item.feature_set, item.algorithm))
    LOGGER.info("Configurações selecionadas: %d", len(selected))
    LOGGER.info("Assinatura da base: %s", input_hash)
    return selected


def selected_configurations_frame(
    selected: Iterable[SelectedConfiguration],
    records: Iterable[dict[str, Any]],
    input_hash: str,
) -> pd.DataFrame:
    grouped: defaultdict[tuple[int, str, str, int], list[dict[str, Any]]]
    grouped = defaultdict(list)
    for record in records:
        key = (
            int(record["checkpoint_percent"]),
            str(record["feature_set"]),
            str(record["algorithm"]),
            int(record["config_index"]),
        )
        grouped[key].append(record)

    selected_keys = {
        (item.checkpoint, item.feature_set, item.algorithm, item.config_index)
        for item in selected
    }
    selected_keys.update(
        (checkpoint, "baseline", "current_top_10", -1)
        for checkpoint in CHECKPOINTS
    )

    summaries: list[dict[str, Any]] = []
    for key in sorted(selected_keys):
        scenario_records = grouped.get(key, [])
        by_year = {
            int(record["evaluation_year"]): record for record in scenario_records
        }
        if set(by_year) != {2018, 2019}:
            raise TrainingError(
                f"Cenário selecionado sem as duas validações temporais: {key}"
            )

        row: dict[str, Any] = {
            "checkpoint_percent": key[0],
            "feature_set": key[1],
            "algorithm": key[2],
            "parameters": by_year[2018]["parameters"],
        }
        for year in (2018, 2019):
            for metric in METRIC_NAMES:
                row[f"{metric}_{year}"] = by_year[year][metric]
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = fmean(
                (float(by_year[2018][metric]), float(by_year[2019][metric]))
            )
        row.update(
            {
                "f1_gap": abs(
                    float(by_year[2018]["f1"]) - float(by_year[2019]["f1"])
                ),
                "protocol_version": PROTOCOL_VERSION,
                "input_sha256": input_hash,
                "random_seed": RANDOM_SEED,
            }
        )
        summaries.append(row)

    return pd.DataFrame(summaries)


def load_selected_configurations(
    path: Path, input_hash: str
) -> list[SelectedConfiguration]:
    if not path.is_file():
        raise TrainingError(
            "Configurações não encontradas. Execute primeiro a validação padrão"
        )

    frame = pd.read_csv(
        path,
        dtype={"protocol_version": str, "input_sha256": str},
    )
    required = {
        "protocol_version",
        "checkpoint_percent",
        "feature_set",
        "algorithm",
        "parameters",
        "f1_2018",
        "f1_2019",
        "mean_f1",
        "mcc_2018",
        "mcc_2019",
        "mean_mcc",
        "pr_auc_2018",
        "pr_auc_2019",
        "mean_pr_auc",
        "f1_gap",
        "input_sha256",
        "random_seed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TrainingError(f"Arquivo de configurações incompleto: {missing}")
    if set(frame["input_sha256"]) != {input_hash}:
        raise TrainingError(
            "A base mudou após a validação; refaça a validação antes do teste"
        )
    if set(frame["protocol_version"]) != {PROTOCOL_VERSION}:
        raise TrainingError(
            "O protocolo mudou após a validação; refaça a validação"
        )
    if set(frame["random_seed"]) != {RANDOM_SEED}:
        raise TrainingError("A semente salva difere da definida no protocolo")

    selected: list[SelectedConfiguration] = []
    scenario_keys: set[tuple[int, str, str]] = set()
    for row in frame.to_dict("records"):
        algorithm = str(row["algorithm"])
        if algorithm == "current_top_10":
            continue
        feature_set = str(row["feature_set"])
        checkpoint = int(row["checkpoint_percent"])
        if algorithm not in ALGORITHM_CONFIGS:
            raise TrainingError(f"Algoritmo inválido nas configurações: {algorithm}")
        if feature_set not in FEATURE_SETS:
            raise TrainingError(f"Conjunto de atributos inválido: {feature_set}")
        if checkpoint not in CHECKPOINTS:
            raise TrainingError(f"Checkpoint inválido nas configurações: {checkpoint}")

        scenario_key = (checkpoint, feature_set, algorithm)
        if scenario_key in scenario_keys:
            raise TrainingError(f"Configuração duplicada para {scenario_key}")
        scenario_keys.add(scenario_key)

        parameters = json.loads(str(row["parameters"]))
        configurations = ALGORITHM_CONFIGS[algorithm]
        matching_indexes = [
            index
            for index, configuration in enumerate(configurations)
            if parameters == configuration
        ]
        if len(matching_indexes) != 1:
            raise TrainingError(
                f"Configuração não reconhecida para {algorithm}: {parameters}"
            )
        config_index = matching_indexes[0]

        selected.append(
            SelectedConfiguration(
                checkpoint=checkpoint,
                feature_set=feature_set,
                algorithm=algorithm,
                config_index=config_index,
                parameters=parameters,
                f1_2018=float(row["f1_2018"]),
                f1_2019=float(row["f1_2019"]),
                mean_f1=float(row["mean_f1"]),
                f1_gap=float(row["f1_gap"]),
            )
        )

    expected = len(CHECKPOINTS) * len(FEATURE_SETS) * len(ALGORITHM_CONFIGS)
    if len(selected) != expected:
        raise TrainingError(
            f"Esperadas {expected} configurações salvas, encontradas {len(selected)}"
        )
    return selected


def prediction_records(
    test: pd.DataFrame,
    predicted: np.ndarray,
    scores: np.ndarray,
    feature_set: str,
    algorithm: str,
    parameters: dict[str, Any],
    input_hash: str,
) -> list[dict[str, Any]]:
    """Cria a trilha de auditoria das previsões feitas para cada piloto."""

    result = test[list(IDENTIFIER_COLUMNS)].copy()
    result["feature_set"] = feature_set
    result["algorithm"] = algorithm
    result["parameters"] = configuration_json(parameters)
    result["actual_top_10"] = test[TARGET].to_numpy(dtype=int)
    result["predicted_top_10"] = predicted.astype(int)
    result["top10_score"] = scores
    result["correct"] = result["actual_top_10"].eq(result["predicted_top_10"])
    result["protocol_version"] = PROTOCOL_VERSION
    result["input_sha256"] = input_hash
    return result.to_dict("records")


def final_evaluation_records(
    frame: pd.DataFrame,
    selected: Iterable[SelectedConfiguration],
    input_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluations: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selected_items = list(selected)

    for scenario_number, item in enumerate(selected_items, start=1):
        started_at = perf_counter()
        LOGGER.info(
            "Teste final [%d/%d] | checkpoint=%d | atributos=%s | algoritmo=%s",
            scenario_number,
            len(selected_items),
            item.checkpoint,
            item.feature_set,
            item.algorithm,
        )
        checkpoint_data = frame[frame["checkpoint_percent"].eq(item.checkpoint)]
        train = checkpoint_data[checkpoint_data["year"].le(2019)]
        test = checkpoint_data[checkpoint_data["year"].eq(2020)]
        predicted, scores, fill_value = train_and_predict(
            item.algorithm,
            item.parameters,
            train,
            test,
            FEATURE_SETS[item.feature_set],
        )
        metrics = calculate_metrics(test[TARGET], predicted, scores)
        evaluations.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "input_sha256": input_hash,
                "checkpoint_percent": item.checkpoint,
                "feature_set": item.feature_set,
                "algorithm": item.algorithm,
                "parameters": configuration_json(item.parameters),
                "train_years": "2017-2019",
                "evaluation_year": 2020,
                "grid_fill_value": fill_value,
                "random_seed": RANDOM_SEED,
                **metrics,
            }
        )
        matrices.extend(
            confusion_rows(
                item.checkpoint,
                item.feature_set,
                item.algorithm,
                metrics,
                input_hash,
            )
        )
        predictions.extend(
            prediction_records(
                test,
                predicted,
                scores,
                item.feature_set,
                item.algorithm,
                item.parameters,
                input_hash,
            )
        )
        LOGGER.info(
            "Teste final [%d/%d] concluído em %.1fs",
            scenario_number,
            len(selected_items),
            perf_counter() - started_at,
        )

    for checkpoint in CHECKPOINTS:
        test = frame[
            frame["checkpoint_percent"].eq(checkpoint) & frame["year"].eq(2020)
        ]
        predicted = test["current_position"].le(10).astype(int).to_numpy()
        scores = -test["current_position"].to_numpy(dtype=float)
        metrics = calculate_metrics(test[TARGET], predicted, scores)
        evaluations.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "input_sha256": input_hash,
                "checkpoint_percent": checkpoint,
                "feature_set": "baseline",
                "algorithm": "current_top_10",
                "parameters": "{}",
                "train_years": "not_applicable",
                "evaluation_year": 2020,
                "grid_fill_value": None,
                "random_seed": None,
                **metrics,
            }
        )
        matrices.extend(
            confusion_rows(
                checkpoint,
                "baseline",
                "current_top_10",
                metrics,
                input_hash,
            )
        )
        predictions.extend(
            prediction_records(
                test,
                predicted,
                scores,
                "baseline",
                "current_top_10",
                {},
                input_hash,
            )
        )
    return evaluations, matrices, predictions


def confusion_rows(
    checkpoint: int,
    feature_set: str,
    algorithm: str,
    metrics: dict[str, Any],
    input_hash: str,
) -> list[dict[str, Any]]:
    counts = {
        (0, 0): metrics["true_negative"],
        (0, 1): metrics["false_positive"],
        (1, 0): metrics["false_negative"],
        (1, 1): metrics["true_positive"],
    }
    return [
        {
            "protocol_version": PROTOCOL_VERSION,
            "input_sha256": input_hash,
            "checkpoint_percent": checkpoint,
            "feature_set": feature_set,
            "algorithm": algorithm,
            "actual_class": actual,
            "predicted_class": predicted,
            "count": count,
        }
        for (actual, predicted), count in counts.items()
    ]


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise TrainingError(
            f"Resultados já existem ({names}). Use --overwrite para substituí-los"
        )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publica o CSV somente depois que sua gravação termina corretamente."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_validation(
    frame: pd.DataFrame,
    input_hash: str,
    results_dir: Path,
    overwrite: bool,
) -> None:
    output_path = results_dir / VALIDATION_FILE
    ensure_outputs_available((output_path,), overwrite)

    LOGGER.info("Iniciando validações temporais")
    records = validation_records(frame, input_hash)
    expected_records = (
        sum(map(len, ALGORITHM_CONFIGS.values()))
        * len(VALIDATION_SPLITS)
        * len(CHECKPOINTS)
        * len(FEATURE_SETS)
        + len(VALIDATION_SPLITS) * len(CHECKPOINTS)
    )
    if len(records) != expected_records:
        raise TrainingError(
            f"Esperados {expected_records} resultados de validação, "
            f"obtidos {len(records)}"
        )
    selected = select_configurations(records, input_hash)
    write_csv_atomic(
        selected_configurations_frame(selected, records, input_hash), output_path
    )
    LOGGER.info("Melhores resultados de validação gravados em %s", output_path)


def run_final_evaluation(
    frame: pd.DataFrame,
    input_hash: str,
    results_dir: Path,
    overwrite: bool,
) -> None:
    evaluation_path = results_dir / FINAL_EVALUATION_FILE
    matrices_path = results_dir / CONFUSION_MATRICES_FILE
    predictions_path = results_dir / FINAL_PREDICTIONS_FILE
    ensure_outputs_available(
        (evaluation_path, matrices_path, predictions_path), overwrite
    )

    selected = load_selected_configurations(
        results_dir / SELECTED_CONFIGS_FILE, input_hash
    )
    LOGGER.info("Iniciando teste final com configurações já selecionadas")
    evaluations, matrices, predictions = final_evaluation_records(
        frame, selected, input_hash
    )
    expected_evaluations = len(selected) + len(CHECKPOINTS)
    expected_predictions = len(frame[frame["year"].eq(2020)]) * (
        len(FEATURE_SETS) * len(ALGORITHM_CONFIGS) + 1
    )
    if len(evaluations) != expected_evaluations:
        raise TrainingError("Quantidade inesperada de avaliações finais")
    if len(matrices) != expected_evaluations * 4:
        raise TrainingError("Quantidade inesperada de células de confusão")
    if len(predictions) != expected_predictions:
        raise TrainingError("Quantidade inesperada de previsões finais")
    write_csv_atomic(pd.DataFrame(evaluations), evaluation_path)
    write_csv_atomic(pd.DataFrame(matrices), matrices_path)
    write_csv_atomic(pd.DataFrame(predictions), predictions_path)
    LOGGER.info("Avaliação final gravada em %s", evaluation_path)
    LOGGER.info("Matrizes de confusão gravadas em %s", matrices_path)
    LOGGER.info("Previsões finais gravadas em %s", predictions_path)


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_dir, args.stage)
    try:
        frame = load_and_validate_data(args.input)
        input_hash = file_sha256(args.input)
        LOGGER.info("Base: %s", args.input)
        LOGGER.info(
            "Linhas: %d | Corridas: %d", len(frame), frame["race_id"].nunique()
        )
        LOGGER.info(
            "Etapa: %s | Protocolo: %s | Semente: %d",
            args.stage,
            PROTOCOL_VERSION,
            RANDOM_SEED,
        )
        validate_environment_and_configuration(
            frame, input_hash, args.results_dir
        )

        if args.stage == "check":
            pass
        elif args.stage == "validation":
            run_validation(frame, input_hash, args.results_dir, args.overwrite)
        else:
            run_final_evaluation(frame, input_hash, args.results_dir, args.overwrite)
        LOGGER.info("Execução concluída | Log: %s", log_path)
        return 0
    except TrainingError as error:
        LOGGER.error("Execução interrompida: %s", error)
        return 1
    except Exception:
        LOGGER.exception("Falha inesperada no treinamento")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
