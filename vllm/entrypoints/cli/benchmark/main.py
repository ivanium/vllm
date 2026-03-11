# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import sys
import typing
from pathlib import Path

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def apply_bench_config(
    config_path: str | None,
    args: argparse.Namespace,
) -> None:
    """Apply YAML benchmark config file as defaults, respecting explicit CLI args.

    Args:
        config_path: Path to YAML config file, or None (noop).
        args: The parsed namespace to update in-place.
    """
    if config_path is None:
        return

    import yaml

    yaml_path = Path(config_path)
    with yaml_path.open() as f:
        config = yaml.safe_load(f)

    if not config:
        return

    # Collect the set of flags explicitly provided on the command line.
    # We look at sys.argv and strip leading dashes, normalizing to underscored names.
    explicit_args: set[str] = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            key = token.lstrip("-").split("=")[0].replace("-", "_")
            explicit_args.add(key)

    # Apply YAML values for keys not explicitly provided on CLI.
    for key, value in config.items():
        normalized_key = key.replace("-", "_")
        if normalized_key not in explicit_args and hasattr(args, normalized_key):
            setattr(args, normalized_key, value)


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "vLLM bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        apply_bench_config(getattr(args, "bench_config", None), args)
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"vllm {self.name} <bench_type> [options]",
        )
        bench_parser.add_argument(
            "--bench-config",
            type=str,
            default=None,
            metavar="YAML_FILE",
            help="Path to a YAML file containing benchmark configuration. "
            "YAML keys map to CLI arg names (underscored). "
            "Explicit CLI args override YAML values.",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
                subcmd=f"{self.name} {cmd_cls.name}"
            )
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
