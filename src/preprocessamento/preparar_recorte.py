#!/usr/bin/env python3
"""Prepara a base usada para prever o top 10 das corridas de Fórmula 1.

O arquivo concentra o fluxo completo: lê e valida os CSVs originais, seleciona
2017–2020, registra a cobertura e cria uma linha por piloto ativo nos checkpoints
de 10%, 25%, 50% e 75%. O abandono é aproximado pela última passagem registrada.
Campos ``audit_*`` existem somente em memória para validar a construção e não
são gravados na base final.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Hashable, Iterable, Sequence, TypeVar


# Configuração do recorte e das saídas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dados" / "originais"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dados" / "gerados"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "preprocessamento"

START_YEAR = 2017
END_YEAR = 2020
EXPECTED_RACE_COUNT = 79
NULL_MARKER = r"\N"

CHECKPOINTS_FILE = "base_checkpoints.csv"
CHECKPOINT_PERCENTAGES = (10, 25, 50, 75)
CLASSIFIED_FINISH_RE = re.compile(r"\+\d+ Laps?")
LOGGER = logging.getLogger("preprocessamento")

COVERAGE_FIELDS = """
race_id year round race_name race_date results_count drivers_with_laps
lap_records_count max_observed_lap grid_zero_count lap_durations_available
actual_race_laps_available cumulative_lap_clock_available
retirement_timing_available checkpoint_ready coverage_status notes
""".split()

CHECKPOINT_FIELDS = """
race_id year driver_id checkpoint_percent current_position
laps_behind_checkpoint_leader gap_to_lap_leader_ms
relative_pace_to_checkpoint_leader grid_position non_standard_grid_start
top_10_final
""".split()

Row = dict[str, object]
Rows = list[Row]
UniqueRule = tuple[tuple[str, ...], str]
Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class TimedLap:
    """Volta com duração individual e tempo acumulado desde a primeira volta."""

    lap: int
    position: int
    duration_ms: int
    cumulative_ms: int


@dataclass(frozen=True)
class PreparedData:
    """Fontes do recorte mantidas em memória durante o pré-processamento."""

    races: Rows
    statuses: dict[int, str]
    results: Rows
    lap_times: Rows


class PreparationError(Exception):
    """Indica que uma entrada não atende às regras mínimas de preparação."""


# Leitura e validação das fontes
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida os CSVs originais, seleciona 2017-2020, registra a "
            "auditoria e gera a base dos checkpoints."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Diretório dos CSVs originais. Padrão: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório das saídas geradas. Padrão: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Diretório dos logs de auditoria. Padrão: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir atomicamente saídas geradas anteriormente.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    """Registra a execução no terminal e em um arquivo de auditoria próprio."""

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"preprocessamento_{timestamp}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(formatter)
    LOGGER.addHandler(terminal_handler)
    return log_path


def iter_csv_rows(
    path: Path, required_columns: set[str]
) -> Iterable[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise PreparationError(f"Arquivo obrigatório não encontrado: {path}")

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise PreparationError(f"CSV sem cabeçalho: {path}")
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise PreparationError(f"Colunas ausentes em {path.name}: {missing}")

        for line_number, row in enumerate(reader, start=2):
            yield line_number, row


def required_int(
    row: dict[str, str], field: str, table: str, line_number: int
) -> int:
    value = row[field]
    if value in ("", NULL_MARKER):
        raise PreparationError(
            f"{table}:{line_number}: {field} não pode estar ausente"
        )

    try:
        return int(value)
    except ValueError as exc:
        raise PreparationError(
            f"{table}:{line_number}: {field} não é inteiro: {value!r}"
        ) from exc


def validate_minimum(value: int, minimum: int, context: str) -> None:
    if value < minimum:
        raise PreparationError(
            f"{context}: esperado valor >= {minimum}, recebido {value}"
        )


def group_rows(
    rows: Iterable[Row], key: Callable[[Row], Key]
) -> defaultdict[Key, Rows]:
    grouped: defaultdict[Key, Rows] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def group_by_race(rows: Iterable[Row]) -> defaultdict[int, Rows]:
    return group_rows(rows, lambda row: int(row["race_id"]))


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def load_races(input_dir: Path) -> Rows:
    path = input_dir / "races.csv"
    required = {"raceId", "year", "round", "name", "date"}
    races: Rows = []

    for line_number, row in iter_csv_rows(path, required):
        race_id = required_int(row, "raceId", path.name, line_number)
        year = required_int(row, "year", path.name, line_number)
        if not START_YEAR <= year <= END_YEAR:
            continue

        round_number = required_int(row, "round", path.name, line_number)
        validate_minimum(race_id, 1, f"{path.name}:{line_number}:raceId")
        validate_minimum(round_number, 1, f"{path.name}:{line_number}:round")

        try:
            race_date = date.fromisoformat(row["date"])
        except ValueError as exc:
            raise PreparationError(
                f"{path.name}:{line_number}: data inválida: {row['date']!r}"
            ) from exc

        if race_date.year != year:
            raise PreparationError(
                f"{path.name}:{line_number}: ano e data pertencem a anos diferentes"
            )

        races.append(
            {
                "race_id": race_id,
                "year": year,
                "round": round_number,
                "race_name": row["name"],
                "race_date": row["date"],
            }
        )

    ensure_unique(races, ("race_id",), "races.csv no recorte")
    ensure_unique(races, ("year", "round"), "temporada e etapa")
    if not races:
        raise PreparationError(
            f"Nenhuma corrida encontrada entre {START_YEAR} e {END_YEAR}"
        )
    ensure(
        len(races) == EXPECTED_RACE_COUNT,
        f"Esperadas {EXPECTED_RACE_COUNT} corridas no recorte, encontradas "
        f"{len(races)}",
    )
    return races


def load_statuses(input_dir: Path) -> dict[int, str]:
    path = input_dir / "status.csv"
    required = {"statusId", "status"}
    statuses: Rows = []

    for line_number, row in iter_csv_rows(path, required):
        status_id = required_int(row, "statusId", path.name, line_number)
        validate_minimum(status_id, 1, f"{path.name}:{line_number}:statusId")
        if not row["status"]:
            raise PreparationError(f"{path.name}:{line_number}: status vazio")
        statuses.append({"status_id": status_id, "status": row["status"]})

    ensure_unique(statuses, ("status_id",), "status.csv")
    return {int(row["status_id"]): str(row["status"]) for row in statuses}


@dataclass(frozen=True)
class IntegerField:
    """Conversão de uma coluna inteira do CSV para o nome usado internamente."""

    source: str
    target: str
    minimum: int | None = None


@dataclass(frozen=True)
class RaceTable:
    """Regras de leitura e unicidade de uma tabela ligada às corridas."""

    filename: str
    fields: tuple[IntegerField, ...]
    unique_keys: tuple[UniqueRule, ...]


RESULTS_TABLE = RaceTable(
    "results.csv",
    (
        IntegerField("resultId", "result_id", 1),
        IntegerField("driverId", "driver_id", 1),
        IntegerField("grid", "grid", 0),
        IntegerField("positionOrder", "position_order", 1),
        IntegerField("laps", "final_laps", 0),
        IntegerField("statusId", "status_id"),
    ),
    (
        (("result_id",), "results.resultId no recorte"),
        (("race_id", "driver_id"), "results por corrida e piloto"),
    ),
)
LAP_TIMES_TABLE = RaceTable(
    "lap_times.csv",
    (
        IntegerField("driverId", "driver_id", 1),
        IntegerField("lap", "lap", 1),
        IntegerField("position", "position", 1),
        IntegerField("milliseconds", "milliseconds", 1),
    ),
    ((("race_id", "driver_id", "lap"), "lap_times por corrida, piloto e volta"),),
)


def load_race_table(input_dir: Path, race_ids: set[int], table: RaceTable) -> Rows:
    """Lê uma tabela relacionada e conserva somente as corridas do recorte."""

    path = input_dir / table.filename
    required = {"raceId", *(field.source for field in table.fields)}
    records: Rows = []

    for line_number, source_row in iter_csv_rows(path, required):
        race_id = required_int(source_row, "raceId", path.name, line_number)
        if race_id not in race_ids:
            continue

        record: Row = {"race_id": race_id}
        for field in table.fields:
            value = required_int(source_row, field.source, path.name, line_number)
            if field.minimum is not None:
                validate_minimum(
                    value, field.minimum, f"{path.name}:{line_number}:{field.source}"
                )
            record[field.target] = value
        records.append(record)

    for fields, label in table.unique_keys:
        ensure_unique(records, fields, label)
    return records


def ensure_unique(rows: Iterable[Row], fields: tuple[str, ...], label: str) -> None:
    seen: set[tuple[object, ...]] = set()
    duplicates: list[tuple[object, ...]] = []

    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    if duplicates:
        examples = ", ".join(map(str, duplicates[:5]))
        raise PreparationError(f"Chaves duplicadas em {label}: {examples}")


def validate_references(data: PreparedData) -> None:
    race_ids = {int(row["race_id"]) for row in data.races}
    result_keys = {
        (int(row["race_id"]), int(row["driver_id"])) for row in data.results
    }

    for result in data.results:
        race_id = int(result["race_id"])
        status_id = int(result["status_id"])
        checks = (
            (race_id in race_ids, f"Resultado referencia corrida ausente: {race_id}"),
            (status_id in data.statuses, f"Resultado referencia status ausente: {status_id}"),
        )
        for valid, message in checks:
            ensure(valid, message)

    for row in data.lap_times:
        key = (int(row["race_id"]), int(row["driver_id"]))
        if key not in result_keys:
            raise PreparationError(
                f"lap_times.csv referencia participação ausente: {key}"
            )


def validate_lap_coverage(results: Rows, lap_times: Rows) -> None:
    """Garante voltas contínuas de 1 até o total registrado para cada piloto."""

    laps_by_result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for lap_time in lap_times:
        key = (int(lap_time["race_id"]), int(lap_time["driver_id"]))
        laps_by_result[key].append(int(lap_time["lap"]))

    for result in results:
        key = (int(result["race_id"]), int(result["driver_id"]))
        observed_laps = sorted(laps_by_result.get(key, []))
        final_laps = int(result["final_laps"])
        expected_laps = list(range(1, final_laps + 1))
        if observed_laps != expected_laps:
            raise PreparationError(
                "Cobertura de voltas divergente para "
                f"raceId={key[0]}, driverId={key[1]}: "
                f"resultado={final_laps}, observadas={len(observed_laps)}"
            )


# Diagnóstico de cobertura
def build_coverage(data: PreparedData) -> Rows:
    """Resume, por corrida, a disponibilidade e as limitações das fontes."""

    results_by_race = group_by_race(data.results)
    lap_times_by_race = group_by_race(data.lap_times)
    coverage: Rows = []

    for race in sorted(
        data.races, key=lambda row: (int(row["year"]), int(row["round"]))
    ):
        race_id = int(race["race_id"])
        race_results = results_by_race[race_id]
        race_laps = lap_times_by_race[race_id]
        lap_data_available = int(bool(race_laps))

        coverage.append(
            {
                **race,
                "results_count": len(race_results),
                "drivers_with_laps": len(
                    {int(row["driver_id"]) for row in race_laps}
                ),
                "lap_records_count": len(race_laps),
                "max_observed_lap": max(
                    (int(row["lap"]) for row in race_laps), default=0
                ),
                "grid_zero_count": sum(int(row["grid"]) == 0 for row in race_results),
                "lap_durations_available": lap_data_available,
                "actual_race_laps_available": lap_data_available,
                "cumulative_lap_clock_available": lap_data_available,
                "retirement_timing_available": 0,
                "checkpoint_ready": lap_data_available,
                "coverage_status": "usable_with_limitations",
                "notes": "abandono aproximado pela última passagem registrada",
            }
        )

    return coverage


# Construção dos checkpoints
def build_timed_laps(
    lap_times: Rows,
) -> dict[tuple[int, int], list[TimedLap]]:
    """Soma os tempos de volta para criar o relógio acumulado de cada piloto."""

    raw_by_driver = group_rows(
        lap_times,
        lambda row: (int(row["race_id"]), int(row["driver_id"])),
    )

    timed_by_driver: dict[tuple[int, int], list[TimedLap]] = {}
    for key, rows in raw_by_driver.items():
        cumulative_ms = 0
        timed_laps: list[TimedLap] = []
        for row in sorted(rows, key=lambda item: int(item["lap"])):
            duration_ms = int(row["milliseconds"])
            cumulative_ms += duration_ms
            timed_laps.append(
                TimedLap(
                    lap=int(row["lap"]),
                    position=int(row["position"]),
                    duration_ms=duration_ms,
                    cumulative_ms=cumulative_ms,
                )
            )
        timed_by_driver[key] = timed_laps

    return timed_by_driver


def index_first_crossings(
    timed_laps: dict[tuple[int, int], list[TimedLap]],
) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Indexa a primeira passagem em cada volta para formar o relógio comum."""

    first_crossings: dict[tuple[int, int], tuple[int, int, int]] = {}

    for (race_id, driver_id), laps in timed_laps.items():
        for lap in laps:
            key = (race_id, lap.lap)
            candidate = (lap.cumulative_ms, lap.position, driver_id)
            if key not in first_crossings or candidate < first_crossings[key]:
                first_crossings[key] = candidate

    return first_crossings


def is_classified_finisher(status: str) -> bool:
    """Reconhece quem terminou ou foi classificado a voltas do vencedor."""

    return status == "Finished" or CLASSIFIED_FINISH_RE.fullmatch(status) is not None


def latest_lap_at_or_before(
    laps: list[TimedLap], checkpoint_time_ms: int
) -> TimedLap | None:
    """Retorna a última passagem conhecida do piloto antes do checkpoint."""

    return next(
        (lap for lap in reversed(laps) if lap.cumulative_ms <= checkpoint_time_ms),
        None,
    )


def median_lap_duration(laps: list[TimedLap], last_lap: int) -> float:
    """Calcula o ritmo mediano sem a volta 1, que inclui a largada."""

    durations = [lap.duration_ms for lap in laps if 2 <= lap.lap <= last_lap]
    if not durations:
        raise PreparationError(
            f"Não há voltas completas para calcular o ritmo até a volta {last_lap}"
        )
    return statistics.median(durations)


def relative_pace(
    driver_laps: list[TimedLap], leader_laps: list[TimedLap], last_lap: int
) -> float:
    """Compara piloto e líder usando exatamente o mesmo intervalo de voltas."""

    driver_median = median_lap_duration(driver_laps, last_lap)
    leader_median = median_lap_duration(leader_laps, last_lap)
    return driver_median / leader_median


def build_checkpoint_rows(
    data: PreparedData,
) -> tuple[Rows, Counter[tuple[int, str]]]:
    """Constrói uma linha por piloto ativo e checkpoint.

    Cada checkpoint ocorre quando o líder completa a volta correspondente à
    porcentagem da distância efetivamente disputada. Todos os pilotos são
    observados nesse mesmo instante da corrida.
    """

    race_by_id = {int(row["race_id"]): row for row in data.races}
    results_by_race = group_by_race(data.results)

    timed_laps = build_timed_laps(data.lap_times)
    first_crossings = index_first_crossings(timed_laps)

    output: Rows = []
    exclusions: Counter[tuple[int, str]] = Counter()

    for race_id in sorted(
        race_by_id,
        key=lambda item: (
            int(race_by_id[item]["year"]),
            int(race_by_id[item]["round"]),
        ),
    ):
        race = race_by_id[race_id]
        race_results = results_by_race[race_id]
        # O recorte usa a distância efetivamente disputada, não a planejada.
        race_total_laps = max(int(row["final_laps"]) for row in race_results)
        observed_total_laps = max(
            lap.lap
            for (current_race_id, _), laps in timed_laps.items()
            if current_race_id == race_id
            for lap in laps
        )
        if race_total_laps != observed_total_laps:
            raise PreparationError(
                f"Total de voltas divergente em raceId={race_id}: "
                f"results={race_total_laps}, lap_times={observed_total_laps}"
            )

        for checkpoint_percent in CHECKPOINT_PERCENTAGES:
            checkpoint_lap = math.ceil(
                race_total_laps * checkpoint_percent / 100
            )
            crossing_key = (race_id, checkpoint_lap)
            if crossing_key not in first_crossings:
                raise PreparationError(
                    f"Checkpoint sem passagem em raceId={race_id}, "
                    f"volta={checkpoint_lap}"
                )

            checkpoint_time_ms, leader_position, checkpoint_leader_id = (
                first_crossings[crossing_key]
            )
            if leader_position != 1:
                raise PreparationError(
                    f"Primeira passagem do checkpoint não está na posição 1: "
                    f"raceId={race_id}, volta={checkpoint_lap}"
                )

            leader_laps = timed_laps[(race_id, checkpoint_leader_id)]
            checkpoint_rows: Rows = []

            for result in sorted(
                race_results, key=lambda row: int(row["driver_id"])
            ):
                driver_id = int(result["driver_id"])
                driver_key = (race_id, driver_id)
                driver_laps = timed_laps.get(driver_key, [])
                if not driver_laps:
                    exclusions[(checkpoint_percent, "sem_volta_registrada")] += 1
                    continue

                final_status = data.statuses[int(result["status_id"])]
                classified_finisher = is_classified_finisher(final_status)
                if (
                    not classified_finisher
                    and driver_laps[-1].cumulative_ms < checkpoint_time_ms
                ):
                    # Sem o instante exato do abandono, uma passagem posterior
                    # comprova retrospectivamente a atividade no checkpoint.
                    exclusions[(checkpoint_percent, "sem_evidencia_de_atividade")] += 1
                    continue

                current_lap = latest_lap_at_or_before(
                    driver_laps, checkpoint_time_ms
                )
                if current_lap is None:
                    exclusions[(checkpoint_percent, "sem_observacao_ate_checkpoint")] += 1
                    continue

                lap_leader_time_ms = first_crossings[(race_id, current_lap.lap)][0]
                # Grid zero não representa uma posição convencional de largada.
                grid = int(result["grid"])

                checkpoint_rows.append(
                    {
                        "race_id": race_id,
                        "year": race["year"],
                        "driver_id": driver_id,
                        "checkpoint_percent": checkpoint_percent,
                        "audit_race_total_laps": race_total_laps,
                        "checkpoint_lap": checkpoint_lap,
                        "audit_checkpoint_time_ms": checkpoint_time_ms,
                        "current_position": None,
                        "audit_last_crossing_time_ms": current_lap.cumulative_ms,
                        "laps_completed": current_lap.lap,
                        "laps_behind_checkpoint_leader": (
                            checkpoint_lap - current_lap.lap
                        ),
                        "gap_to_lap_leader_ms": (
                            current_lap.cumulative_ms - lap_leader_time_ms
                        ),
                        "relative_pace_to_checkpoint_leader": round(
                            relative_pace(
                                driver_laps, leader_laps, current_lap.lap
                            ),
                            6,
                        ),
                        "grid_position": None if grid == 0 else grid,
                        "non_standard_grid_start": int(grid == 0),
                        # A posição final só é positiva se o piloto foi classificado.
                        "top_10_final": int(
                            int(result["position_order"]) <= 10
                            and classified_finisher
                        ),
                    }
                )

            # Mais voltas prevalecem; na mesma volta, fica à frente quem
            # registrou primeiro sua última passagem no relógio comum.
            ordered_checkpoint_rows = sorted(
                checkpoint_rows,
                key=lambda row: (
                    -int(row["laps_completed"]),
                    int(row["audit_last_crossing_time_ms"]),
                    int(row["driver_id"]),
                ),
            )
            for current_position, row in enumerate(
                ordered_checkpoint_rows, start=1
            ):
                row["current_position"] = current_position

            output.extend(
                sorted(
                    ordered_checkpoint_rows,
                    key=lambda row: int(row["driver_id"]),
                )
            )

    return output, exclusions


# Validação e escrita das saídas
def validate_checkpoint_rows(rows: Rows) -> None:
    """Valida chaves, posições, tempos e os dez positivos de cada checkpoint."""

    keys: set[tuple[int, int, int]] = set()
    race_checkpoints: dict[tuple[int, int], Rows] = defaultdict(list)

    for row in rows:
        key = (
            int(row["race_id"]),
            int(row["driver_id"]),
            int(row["checkpoint_percent"]),
        )
        ensure(key not in keys, f"Linha duplicada na base de checkpoints: {key}")
        keys.add(key)

        checkpoint_lap = int(row["checkpoint_lap"])
        race_total_laps = int(row["audit_race_total_laps"])
        laps_completed = int(row["laps_completed"])
        laps_behind = int(row["laps_behind_checkpoint_leader"])
        grid_missing = row["grid_position"] is None
        expected_checkpoint_lap = math.ceil(
            race_total_laps * int(row["checkpoint_percent"]) / 100
        )
        checks = (
            (checkpoint_lap == expected_checkpoint_lap, "Volta de checkpoint divergente"),
            (int(row["top_10_final"]) in (0, 1), "Alvo inválido"),
            (laps_behind == checkpoint_lap - laps_completed, "Diferença de voltas divergente"),
            (laps_behind >= 0, "Piloto à frente da volta do checkpoint"),
            (int(row["gap_to_lap_leader_ms"]) >= 0, "Diferença negativa para líder da volta"),
            (
                int(row["audit_last_crossing_time_ms"])
                <= int(row["audit_checkpoint_time_ms"]),
                "Observação posterior ao checkpoint",
            ),
            (float(row["relative_pace_to_checkpoint_leader"]) > 0, "Ritmo relativo inválido"),
            (
                int(row["non_standard_grid_start"]) == int(grid_missing),
                "Indicador de grid especial divergente",
            ),
        )
        for valid, message in checks:
            ensure(valid, f"{message}: {key}")

        race_checkpoint = (int(row["race_id"]), int(row["checkpoint_percent"]))
        race_checkpoints[race_checkpoint].append(row)

    expected_checkpoints = EXPECTED_RACE_COUNT * len(CHECKPOINT_PERCENTAGES)
    ensure(
        len(race_checkpoints) == expected_checkpoints,
        f"Esperados {expected_checkpoints} checkpoints de corrida, "
        f"encontrados {len(race_checkpoints)}",
    )

    for race_checkpoint, checkpoint_rows in race_checkpoints.items():
        current_positions = sorted(
            int(row["current_position"]) for row in checkpoint_rows
        )
        expected_positions = list(range(1, len(checkpoint_rows) + 1))
        current_top_10 = sum(
            int(row["current_position"]) <= 10 for row in checkpoint_rows
        )
        positives = sum(int(row["top_10_final"]) for row in checkpoint_rows)
        group_checks = (
            (current_positions == expected_positions, "Posições não contínuas"),
            (current_top_10 == 10, f"Top 10 atual com {current_top_10} pilotos"),
            (positives == 10, f"Top 10 final com {positives} pilotos"),
        )
        for valid, message in group_checks:
            ensure(valid, f"{message} em {race_checkpoint}")


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        files = ", ".join(str(path) for path in existing)
        raise PreparationError(
            f"Saída já existente: {files}. Use --overwrite para substituí-la."
        )


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Row]) -> None:
    """Publica o CSV somente depois que o arquivo temporário foi concluído."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: "" if row.get(field) is None else row.get(field)
                        for field in fieldnames
                    }
                )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Calcula a assinatura usada para identificar exatamente a base gerada."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_coverage(coverage: Rows) -> None:
    """Registra a cobertura de cada corrida sem criar outro arquivo de dados."""

    for row in coverage:
        details = "; ".join(f"{field}={row[field]}" for field in COVERAGE_FIELDS)
        LOGGER.info("Cobertura | %s", details)


def prepare(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Orquestra a leitura, a validação, a auditoria e a base final."""

    checkpoints_path = output_dir / CHECKPOINTS_FILE
    ensure_outputs_available((checkpoints_path,), overwrite)

    races = load_races(input_dir)
    race_ids = {int(row["race_id"]) for row in races}
    data = PreparedData(
        races=races,
        statuses=load_statuses(input_dir),
        results=load_race_table(input_dir, race_ids, RESULTS_TABLE),
        lap_times=load_race_table(input_dir, race_ids, LAP_TIMES_TABLE),
    )

    validate_references(data)
    LOGGER.info("Validação de referências: concluída")
    validate_lap_coverage(data.results, data.lap_times)
    LOGGER.info("Validação da continuidade das voltas: concluída")

    coverage = build_coverage(data)
    checkpoint_rows, exclusions = build_checkpoint_rows(data)
    validate_checkpoint_rows(checkpoint_rows)
    LOGGER.info("Validação da base de checkpoints: concluída")

    write_csv_atomic(checkpoints_path, CHECKPOINT_FIELDS, checkpoint_rows)
    log_coverage(coverage)

    LOGGER.info("Corridas selecionadas: %s", len(data.races))
    LOGGER.info("Resultados selecionados: %s", len(data.results))
    LOGGER.info("Tempos de volta selecionados: %s", len(data.lap_times))
    LOGGER.info("Base de checkpoints: %s", checkpoints_path)
    LOGGER.info("Linhas geradas: %s", len(checkpoint_rows))
    LOGGER.info("SHA-256 da base: %s", sha256_file(checkpoints_path))

    included_by_checkpoint = Counter(
        int(row["checkpoint_percent"]) for row in checkpoint_rows
    )
    for checkpoint_percent in CHECKPOINT_PERCENTAGES:
        included = included_by_checkpoint[checkpoint_percent]
        no_lap = exclusions[(checkpoint_percent, "sem_volta_registrada")]
        no_activity = exclusions[
            (checkpoint_percent, "sem_evidencia_de_atividade")
        ]
        no_observation = exclusions[
            (checkpoint_percent, "sem_observacao_ate_checkpoint")
        ]
        LOGGER.info(
            "Checkpoint %s%%: %s incluídos; %s sem volta; "
            "%s sem evidência de atividade; %s sem observação anterior",
            checkpoint_percent,
            included,
            no_lap,
            no_activity,
            no_observation,
        )

    LOGGER.info("Pré-processamento concluído. A base está pronta para o treino.")


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir.resolve())
    LOGGER.info("Início do pré-processamento")
    LOGGER.info("Diretório de entrada: %s", args.input_dir.resolve())
    LOGGER.info("Diretório de saída: %s", args.output_dir.resolve())
    LOGGER.info("Arquivo de log: %s", log_path)
    LOGGER.info(
        "Configuração: anos=%s-%s; checkpoints=%s; sobrescrever=%s",
        START_YEAR,
        END_YEAR,
        ",".join(map(str, CHECKPOINT_PERCENTAGES)),
        args.overwrite,
    )
    prepare(args.input_dir.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except PreparationError as exc:
        LOGGER.error("Erro de preparação: %s", exc)
        raise SystemExit(1) from exc
