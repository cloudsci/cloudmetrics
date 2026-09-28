#!/usr/bin/env python
"""
Compare two pytest-benchmark JSON files (base and head) by the ratio of their
median run-times and exit non-zero if a benchmark got slower than
--max-slowdown or is missing on head. Uses only the standard library:

    python benchmarks/compare.py --base base.json --head head.json \
        --max-slowdown 1.5 --summary summary.md
"""

import argparse
import json
import sys

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_ERROR = 2

# ratios within [1/SAME_BAND, SAME_BAND] are reported as "same"
SAME_BAND = 1.2

STATUS_FASTER = ":rocket: faster"
STATUS_SAME = ":heavy_minus_sign: same"
STATUS_SLOWER = ":warning: slower"
STATUS_REGRESSION = ":x: REGRESSION"
STATUS_MISSING_HEAD = ":x: MISSING on head"
STATUS_MISSING_BASE = ":information_source: new (not on base)"


def load_benchmarks(filename):
    """
    Return ``{fullname: benchmark}``, or ``None`` if the file cannot be read.
    """
    try:
        with open(filename) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as ex:
        print(f"could not read benchmark results from {filename}: {ex}")
        return None
    return {b["fullname"]: b for b in data.get("benchmarks", [])}


def format_time(seconds):
    if seconds is None:
        return "-"
    for unit, factor in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= factor:
            return f"{seconds / factor:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def format_ratio(ratio):
    if ratio is None:
        return "-"
    if ratio < 0.5:
        return f"{ratio:.2f}x ({1.0 / ratio:.0f}x faster)"
    return f"{ratio:.2f}x"


def short_name(benchmark):
    return benchmark.get("name") or benchmark["fullname"].split("::")[-1]


def classify(ratio, max_slowdown):
    if ratio > max_slowdown:
        return STATUS_REGRESSION
    if ratio > SAME_BAND:
        return STATUS_SLOWER
    if ratio < 1.0 / SAME_BAND:
        return STATUS_FASTER
    return STATUS_SAME


def compare(base, head, max_slowdown):
    rows = []
    exit_code = EXIT_OK
    for fullname in sorted(set(base) | set(head)):
        b, h = base.get(fullname), head.get(fullname)
        base_median = b["stats"]["median"] if b else None
        head_median = h["stats"]["median"] if h else None
        ratio = None
        if b is None:
            status = STATUS_MISSING_BASE
        elif h is None:
            status = STATUS_MISSING_HEAD
            exit_code = EXIT_REGRESSION
        else:
            ratio = head_median / base_median
            status = classify(ratio, max_slowdown)
            if status == STATUS_REGRESSION:
                exit_code = EXIT_REGRESSION
        rows.append(
            dict(
                name=short_name(h or b),
                base_median=base_median,
                head_median=head_median,
                ratio=ratio,
                status=status,
            )
        )
    return rows, exit_code


def comparison_table(rows):
    lines = [
        "| Benchmark | Base median | Head median | Ratio (head/base) | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {format_time(r['base_median'])} "
            f"| {format_time(r['head_median'])} | {format_ratio(r['ratio'])} "
            f"| {r['status']} |"
        )
    return "\n".join(lines)


def verdict(rows, exit_code, max_slowdown):
    n_regressions = sum(r["status"] == STATUS_REGRESSION for r in rows)
    n_missing = sum(r["status"] == STATUS_MISSING_HEAD for r in rows)
    n_faster = sum(r["status"] == STATUS_FASTER for r in rows)
    n_compared = sum(r["ratio"] is not None for r in rows)
    if exit_code == EXIT_OK:
        return (
            f"**PASSED**: {n_compared} benchmarks compared, none slower than "
            f"{max_slowdown:g}x the base ({n_faster} faster)."
        )
    problems = []
    if n_regressions:
        problems.append(f"{n_regressions} slower than {max_slowdown:g}x the base")
    if n_missing:
        problems.append(f"{n_missing} missing/failed on head")
    return f"**FAILED**: {n_compared} benchmarks compared, " + ", ".join(problems)


def mask_description(benchmarks):
    masks = {}
    for b in benchmarks.values():
        info = b.get("extra_info", {})
        if "mask" in info:
            masks[info["mask"]] = info
    lines = []
    for name, info in sorted(masks.items(), key=lambda kv: kv[1]["mask_n_objects"]):
        shape = "x".join(str(n) for n in info["mask_shape"])
        lines.append(f"`{name}`: {shape} px, {info['mask_n_objects']} objects")
    return lines


def version_description(benchmarks):
    versions = {
        (
            b.get("extra_info", {}).get("cloudmetrics_version", "unknown"),
            b.get("extra_info", {}).get("cloudmetrics_path", "unknown"),
        )
        for b in benchmarks.values()
    }
    return ", ".join(f"`{v}` ({p})" for v, p in sorted(versions)) or "unknown"


def raw_stats_section(label, benchmarks):
    lines = [
        "<details>",
        f"<summary>Raw benchmark statistics: {label}</summary>",
        "",
        f"cloudmetrics version: {version_description(benchmarks)}",
        "",
        "| Benchmark | Min | Median | Mean | StdDev | Rounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fullname in sorted(benchmarks):
        s = benchmarks[fullname]["stats"]
        lines.append(
            f"| `{short_name(benchmarks[fullname])}` | {format_time(s['min'])} "
            f"| {format_time(s['median'])} | {format_time(s['mean'])} "
            f"| {format_time(s['stddev'])} | {s['rounds']} |"
        )
    lines += ["", "</details>"]
    return "\n".join(lines)


def build_summary(args, base, head, rows, exit_code):
    header = [
        "## Performance benchmark comparison",
        "",
        f"- base: {args.base_label}",
        f"- head: {args.head_label}",
        f"- threshold: head median must not exceed {args.max_slowdown:g}x the "
        f"base median (ratios within {1 / SAME_BAND:.2f}x-{SAME_BAND:g}x are "
        f"reported as 'same')",
    ]
    masks = mask_description({**base, **head})
    if masks:
        header.append("- synthetic masks: " + "; ".join(masks))
    header += [f"- {note}" for note in args.note]
    header.append("")
    sections = [
        "\n".join(header),
        verdict(rows, exit_code, args.max_slowdown),
        "",
        comparison_table(rows),
        "",
        raw_stats_section(f"base ({args.base_label})", base),
        "",
        raw_stats_section(f"head ({args.head_label})", head),
        "",
    ]
    return "\n".join(sections)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split(":\n")[0])
    parser.add_argument("--base", required=True, help="pytest-benchmark JSON of base")
    parser.add_argument("--head", required=True, help="pytest-benchmark JSON of head")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.5,
        help="maximum allowed head/base ratio of the median run-time (default 1.5)",
    )
    parser.add_argument("--summary", help="write Markdown report to this file")
    parser.add_argument("--base-label", default=None, help="label for base run")
    parser.add_argument("--head-label", default=None, help="label for head run")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="additional line for the report header (repeatable)",
    )
    args = parser.parse_args(argv)
    args.note = [note.strip() for note in args.note if note.strip()]
    if args.base_label is None:
        args.base_label = args.base
    if args.head_label is None:
        args.head_label = args.head
    if args.max_slowdown <= 0:
        parser.error("--max-slowdown must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    base = load_benchmarks(args.base)
    head = load_benchmarks(args.head)
    if base is None or head is None:
        message = "benchmark results missing for: " + ", ".join(
            side for side, data in (("base", base), ("head", head)) if data is None
        )
        print(message)
        if args.summary:
            header = [f"- base: {args.base_label}", f"- head: {args.head_label}"]
            header += [f"- {note}" for note in args.note]
            with open(args.summary, "w") as fh:
                fh.write(
                    "## Performance benchmark comparison\n\n"
                    + "\n".join(header)
                    + f"\n\n**ERROR**: {message}\n"
                )
        return EXIT_ERROR

    rows, exit_code = compare(base, head, args.max_slowdown)
    summary = build_summary(args, base, head, rows, exit_code)
    if args.summary:
        with open(args.summary, "w") as fh:
            fh.write(summary)

    print(f"base: {args.base_label}")
    print(f"head: {args.head_label}")
    for note in args.note:
        print(note)
    print()
    print(comparison_table(rows))
    print()
    print(verdict(rows, exit_code, args.max_slowdown))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
