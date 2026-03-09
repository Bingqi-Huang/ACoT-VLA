import dataclasses
import os
import pathlib
import shlex
import subprocess
import sys

import tyro


@dataclasses.dataclass
class Args:
    config_name: str
    exp_name: str
    split: str = "train"
    norm_max_frames: int | None = None
    norm_output_dir: str | None = None
    train_extra_args: list[str] = dataclasses.field(default_factory=list)
    skip_norm: bool = False


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"\n[run_norm_and_train] Running:\n  {shlex.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, env=env)


def main(args: Args) -> None:
    repo_root = _repo_root()
    python_bin = sys.executable
    env = os.environ.copy()

    if not args.skip_norm:
        norm_cmd = [
            python_bin,
            str(repo_root / "scripts" / "compute_norm_stats.py"),
            "--config-name",
            args.config_name,
            "--split",
            args.split,
        ]
        if args.norm_max_frames is not None:
            norm_cmd.extend(["--max-frames", str(args.norm_max_frames)])
        if args.norm_output_dir is not None:
            norm_cmd.extend(["--output-dir", args.norm_output_dir])
        _run_command(norm_cmd, env=env)

    train_cmd = [
        python_bin,
        str(repo_root / "scripts" / "train.py"),
        "--config-name",
        args.config_name,
        "--exp_name",
        args.exp_name,
        "--overwrite=true",
        *args.train_extra_args,
    ]
    _run_command(train_cmd, env=env)


if __name__ == "__main__":
    main(tyro.cli(Args))
