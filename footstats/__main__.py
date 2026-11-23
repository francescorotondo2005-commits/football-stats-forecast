"""Riga di comando: python -m footstats {download,train,predict}."""

import argparse

from .config import DEFAULT_LEVEL, LEAGUES


def main() -> None:
    parser = argparse.ArgumentParser(prog="footstats", description="Previsione di statistiche di partita con range.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="scarica i dati in data/raw/")
    d.add_argument("--force", action="store_true", help="riscarica anche le stagioni già presenti")

    sub.add_parser("train", help="addestra, valuta sul test e salva modelli e report")

    p = sub.add_parser("predict", help="prevede una partita")
    p.add_argument("home", help="squadra di casa, es. Inter")
    p.add_argument("away", help="squadra ospite, es. Roma")
    p.add_argument("--league", default="serie-a", choices=list(LEAGUES))
    p.add_argument("--level", type=float, default=DEFAULT_LEVEL, help="livello del range (default 0.8)")

    args = parser.parse_args()
    if args.cmd == "download":
        from .data import download
        download(force=args.force)
    elif args.cmd == "train":
        from .train import run, save_report
        save_report(run())
        print("Report scritto in reports/")
    elif args.cmd == "predict":
        from .predict import format_prediction, predict_match
        print(format_prediction(predict_match(args.league, args.home, args.away, args.level)))


if __name__ == "__main__":
    main()
