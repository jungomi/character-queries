import argparse
import platform
import subprocess
import sys
from typing import List


def run_pip(
    args: List[str], command: str = "install", upgrade: bool = True, force: bool = False
):
    cmd = ["pip", command]
    if upgrade:
        cmd.append("--upgrade")
    if force:
        cmd.append("--force")
    cmd.extend(args)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def install_requirements(force: bool = False):
    args = ["-r", "requirements.txt"]
    if platform.system() == "Windows":
        args.append("-f")
        args.append("https://download.pytorch.org/whl/torch_stable.html")
    run_pip(args, force=force)


def install_checkers(force: bool = False):
    args = ["mypy", "ruff", "black", "isort"]
    run_pip(args, force=force)


def main():
    parser = argparse.ArgumentParser(description="Install dependencies")
    parser.add_argument(
        dest="target",
        help=(
            "Optionnaly specify which targtet dependencies to install. "
            "`dev` are dev tools (checker, linter, formatter), "
            "and `regular` are the ones in requirements.txt. "
            "For convenience: `all = [deps, dev]` "
            "[Default: all]"
        ),
        choices=["all", "deps", "dev"],
        default="all",
        nargs="*",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        help=("Force re-installing the depenencies"),
        action="store_true",
    )
    options = parser.parse_args()

    if (
        "all" in options.target
        or "deps" in options.target
        or "regular" in options.target
    ):
        print("⇒ Installing regular dependencies")
        install_requirements(force=options.force)
    if "all" in options.target or "dev" in options.target:
        print("⇒ Installing dev depdendencies")
        install_checkers(force=options.force)


if __name__ == "__main__":
    main()
