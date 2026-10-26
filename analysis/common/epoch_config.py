BASELINE_WINDOWS = {
    "standard": (-0.35, -0.25),
    "s5": (-0.95, -0.85),
}


def baseline_window(name: str) -> tuple[float, float]:
    try:
        return BASELINE_WINDOWS[name]
    except KeyError as error:
        choices = ", ".join(BASELINE_WINDOWS)
        raise ValueError(f"Unknown baseline {name!r}; choose from: {choices}.") from error


def epoch_label(
    t_min: float,
    t_max: float,
    baseline: str,
    bpf_low: float = 0.2,
    bpf_high: float = 40,
) -> str:
    bl_min, bl_max = baseline_window(baseline)
    return f"t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf"


def clean_epoch_label(
    t_min: float,
    t_max: float,
    baseline: str,
    bpf_low: float = 0.2,
    bpf_high: float = 40,
) -> str:
    return f"{epoch_label(t_min, t_max, baseline, bpf_low, bpf_high)}-clean"
