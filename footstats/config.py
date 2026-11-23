"""Costanti condivise: campionati, statistiche, percorsi, split temporali."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# slug -> (nome leggibile, codice divisione su football-data.co.uk)
LEAGUES = {
    "serie-a": ("Serie A", "I1"),
    "premier-league": ("Premier League", "E0"),
    "la-liga": ("La Liga", "SP1"),
    "bundesliga": ("Bundesliga", "D1"),
    "ligue-1": ("Ligue 1", "F1"),
}

# statistica -> (colonna casa, colonna ospite) nel CSV originale.
# "cards" = gialli + rossi, calcolata in data.py.
STATS = {
    "goals": ("FTHG", "FTAG"),
    "shots": ("HS", "AS"),
    "sot": ("HST", "AST"),
    "fouls": ("HF", "AF"),
    "corners": ("HC", "AC"),
    "cards": (None, None),
}

STAT_LABELS_IT = {
    "goals": "Gol",
    "shots": "Tiri",
    "sot": "Tiri in porta",
    "fouls": "Falli",
    "corners": "Corner",
    "cards": "Cartellini",
}

FIRST_SEASON = 2014  # 2014-15: le stagioni precedenti servono poco e alcune hanno buchi

# Split temporale: nessuna partita di validazione/test viene mai vista in training.
TRAIN_LAST_SEASON = 2022  # training: 2014-15 ... 2022-23
VALID_SEASON = 2023  # validazione (iperparametri + dispersione): 2023-24
TEST_SEASONS = (2024, 2025)  # test finale: 2024-25 e 2025-26

DEFAULT_LEVEL = 0.80  # livello degli intervalli di previsione


def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> int:
    """Anno di inizio della stagione in corso (la stagione parte a luglio)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1
