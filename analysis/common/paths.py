import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WITHIN_PRIOR_FE_DIRECTORY = "train_within_prior_resamp_binned"


def _required_data_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"Set {variable} to the corresponding data directory before running this analysis."
        )

    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{variable} points to a directory that does not exist: {path}")
    return path.resolve()


def experiment_rawdata_root(experiment: int) -> Path:
    variables = {
        1: "MOTEST_EXP1_RAWDATA",
        2: "MOTEST_EXP2_RAWDATA",
    }
    if experiment not in variables:
        raise ValueError(f"Unsupported experiment {experiment}; expected 1 or 2.")
    return _required_data_directory(variables[experiment])


def derivatives_root() -> Path:
    return _required_data_directory("MOTEST_DERIVATIVES")


def subject_directories(subject: str | None = None) -> list[Path]:
    derivatives = derivatives_root()
    if subject is not None:
        if not subject.startswith("sub-") or Path(subject).name != subject:
            raise ValueError("Subject must be a participant folder name such as 'sub-001'.")
        subject_dir = derivatives / subject
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Participant directory does not exist: {subject_dir}")
        return [subject_dir]

    return sorted(
        (
            path
            for path in derivatives.iterdir()
            if path.is_dir() and path.name.startswith("sub-")
        ),
        key=lambda path: path.name.lower(),
    )


def analysis_cache_dir() -> Path:
    value = os.environ.get("MOTEST_CACHE_DIR")
    path = Path(value).expanduser() if value else PROJECT_ROOT / "outputs" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def within_prior_cache_dir() -> Path:
    path = analysis_cache_dir() / "within_prior"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_output_path(filename: str) -> Path:
    value = os.environ.get("MOTEST_FIGURE_DIR")
    path = Path(value).expanduser() if value else PROJECT_ROOT / "figures" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve() / filename
