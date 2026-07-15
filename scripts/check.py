"""Run STEERING's canonical local and CI verification pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(label: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n==> {label}", flush=True)
    print("    " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        fail(f"{label} failed with exit code {completed.returncode}")


def venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def validate_wheel_metadata(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        wheel_files = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata_files) != 1 or len(wheel_files) != 1:
            fail(f"{wheel.name} does not contain exactly one METADATA and WHEEL file")
        metadata = archive.read(metadata_files[0]).decode("utf-8")
    if "Name: steering\n" not in metadata or "Version: 0.1.0\n" not in metadata:
        fail("wheel metadata does not declare steering 0.1.0")


def wheel_smoke_test(wheel: Path, temporary_root: Path, uv: str) -> None:
    print("\n==> Wheel metadata and clean runtime smoke test", flush=True)
    validate_wheel_metadata(wheel)

    environment = temporary_root / "wheel-smoke"
    run("Create isolated wheel environment", [uv, "venv", "--python", sys.executable, str(environment)])
    python = venv_python(environment)
    run(
        "Install wheel and runtime dependencies",
        [uv, "pip", "install", "--python", str(python), str(wheel)],
    )
    run(
        "Import installed wheel",
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m; import steering; "
                "assert m.version('steering') == '0.1.0'; "
                "assert steering.__file__ is not None; print(steering.__file__)"
            ),
        ],
    )
    run("Run installed CLI", [str(python), "-m", "steering.cli.main", "--help"])
    run(
        "Import installed daemon factory",
        [str(python), "-c", "from steering.app import create_app; assert callable(create_app)"],
    )


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        fail("uv is required; install it from https://docs.astral.sh/uv/")

    run("Lockfile consistency", [uv, "lock", "--check"])
    run("Ruff lint", [sys.executable, "-m", "ruff", "check", "."])
    run("Ruff format check", [sys.executable, "-m", "ruff", "format", "--check", "."])
    run("Mypy", [sys.executable, "-m", "mypy"])
    run(
        "Pytest with coverage",
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=steering",
            "--cov-report=term-missing",
        ],
    )

    with tempfile.TemporaryDirectory(prefix="steering-check-") as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / "dist"
        run(
            "Package build",
            [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(output)],
        )
        wheels = list(output.glob("steering-0.1.0-*.whl"))
        if len(wheels) != 1:
            fail(f"expected one steering 0.1.0 wheel, found {len(wheels)}")
        wheel_smoke_test(wheels[0], temporary_root, uv)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
