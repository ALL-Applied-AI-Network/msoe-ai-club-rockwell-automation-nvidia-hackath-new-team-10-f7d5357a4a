"""Command-line entry point for the M8 operator UI."""

import argparse
from pathlib import Path

from .repository import CheckerOutputRepository
from .server import OperatorUiServer
from .service import SnapshotService


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only Twin-to-Line operator UI"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Checker output directory",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--stale-after",
        type=float,
        default=30.0,
        metavar="SECONDS",
    )

    return parser


def main():
    args = build_parser().parse_args()

    if not 1 <= args.port <= 65535:
        raise SystemExit(
            "--port must be between 1 and 65535"
        )

    if args.stale_after <= 0:
        raise SystemExit(
            "--stale-after must be greater than zero"
        )

    repository = CheckerOutputRepository(
        args.output_dir,
        stale_after_seconds=args.stale_after,
    )

    service = SnapshotService(repository)

    static_dir = Path(__file__).parent / "static"

    server = OperatorUiServer(
        ("127.0.0.1", args.port),
        service,
        static_dir,
    )

    print(
        f"M8 operator UI: "
        f"http://127.0.0.1:{server.server_port}",
        flush=True,
    )

    print(
        f"Read-only source: {repository.output_dir}",
        flush=True,
    )

    try:
        server.serve_forever(
            poll_interval=0.2
        )
    except KeyboardInterrupt:
        print(
            "\nStopping operator UI",
            flush=True,
        )
    finally:
        server.server_close()


if __name__ == "__main__":
    main()