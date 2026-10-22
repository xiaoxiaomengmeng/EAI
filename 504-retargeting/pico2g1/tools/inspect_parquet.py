#!/usr/bin/env python3
"""Inspect the structure of recorded parquet files.

Without arguments it summarises every parquet file in a session directory.
Given a single file it prints the full schema, per-column statistics and
decoded sample rows.

    ./inspect_parquet.py                                   # summarise a session
    ./inspect_parquet.py session_*/data/chunk-000/*.parquet
    ./inspect_parquet.py <file> --row 600 --rows 3
    ./inspect_parquet.py <file> --column wrist
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


POSE7 = ("x", "y", "z", "qx", "qy", "qz", "qw")
STATS_ROW_LIMIT = 20000


def human_bytes(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def element_type(field_type: pa.DataType) -> tuple[str, int]:
    """Return the innermost value type and the nesting depth of a list type."""
    depth = 0
    while pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        field_type = field_type.value_type
        depth += 1
    return str(field_type), depth


def type_label(field_type: pa.DataType) -> str:
    inner, depth = element_type(field_type)
    return inner if depth == 0 else f"{'list' * depth}<{inner}>"


def as_matrix(column: pa.ChunkedArray, limit: int) -> tuple[np.ndarray | None, set[int]]:
    """Convert a list column into an (n, width) float array when the width is uniform."""
    values = column.slice(0, limit).to_pylist()
    widths = {len(value) for value in values if value is not None}
    if len(widths) != 1:
        return None, widths
    width = widths.pop()
    filled = [
        [math.nan] * width if value is None else value
        for value in values
    ]
    try:
        return np.asarray(filled, dtype=np.float64), {width}
    except (TypeError, ValueError):
        return None, {width}


def scalar_detail(column: pa.ChunkedArray, field_type: pa.DataType) -> str:
    if pa.types.is_boolean(field_type):
        values = np.asarray(column.to_pylist())
        true_count = int(np.count_nonzero(values == True))  # noqa: E712
        return f"True {true_count}/{len(values)}"
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        unique = sorted({value for value in column.to_pylist() if value is not None})
        shown = ", ".join(repr(value) for value in unique[:4])
        return f"{len(unique)} unique: {shown}{' ...' if len(unique) > 4 else ''}"
    values = np.asarray(column.to_pylist(), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return "no finite values"
    if pa.types.is_integer(field_type):
        return f"min {int(finite.min())}  max {int(finite.max())}"
    return f"min {finite.min():.4g}  max {finite.max():.4g}  mean {finite.mean():.4g}"


def list_detail(matrix: np.ndarray | None, widths: set[int]) -> str:
    if matrix is None:
        return f"ragged widths {sorted(widths)}" if widths else "empty"
    finite_rows = np.isfinite(matrix).all(axis=1)
    share = f"finite {finite_rows.mean() * 100:.0f}%"
    usable = matrix[finite_rows]
    if not len(usable):
        return share
    ranges = "  ".join(
        f"{POSE7[i] if matrix.shape[1] == 7 else i}[{usable[:, i].min():+.3f},{usable[:, i].max():+.3f}]"
        for i in range(min(matrix.shape[1], 7))
    )
    if matrix.shape[1] == 7:
        norms = np.linalg.norm(usable[:, 3:], axis=1)
        ranges += f"  |q| {norms.min():.3f}..{norms.max():.3f}"
    return f"{share}  {ranges}"


def timestamp_scale(stamp: int) -> tuple[float, str]:
    """Infer the epoch unit from the magnitude rather than from the column name."""
    magnitude = abs(int(stamp))
    if magnitude > 10**17:
        return 1e9, "ns"
    if magnitude > 10**14:
        return 1e6, "us"
    if magnitude > 10**11:
        return 1e3, "ms"
    return 1.0, "s"


def timestamp_report(table: pa.Table) -> list[str]:
    lines = []
    for name in table.schema.names:
        if "timestamp" not in name and not name.endswith("_ns"):
            continue
        if not pa.types.is_integer(table.schema.field(name).type):
            continue
        stamps = table[name].to_numpy(zero_copy_only=False).astype(np.int64)
        if len(stamps) < 2:
            continue
        scale, unit = timestamp_scale(stamps[0])
        span = (stamps[-1] - stamps[0]) / scale
        steps = np.diff(stamps)
        rate = f"{(len(stamps) - 1) / span:.1f} Hz" if span > 0 else "n/a"
        order = "increasing" if np.all(steps > 0) else f"{int(np.sum(steps <= 0))} non-increasing steps"
        lines.append(
            f"  {name} ({unit}): span {span:.2f}s  mean {rate}  "
            f"step {steps.min() / scale * 1000:.1f}..{steps.max() / scale * 1000:.1f} ms  {order}"
        )
    return lines


def format_row(table: pa.Table, index: int, columns: list[str]) -> list[str]:
    lines = []
    for name in columns:
        field_type = table.schema.field(name).type
        value = table[name][index].as_py()
        if value is None:
            lines.append(f"  {name:<28} None")
        elif isinstance(value, list) and len(value) == 7 and all(
            isinstance(item, (int, float)) for item in value
        ):
            position = ", ".join(f"{item:+.4f}" for item in value[:3])
            quaternion = ", ".join(f"{item:+.4f}" for item in value[3:])
            lines.append(f"  {name:<28} pos[{position}]  quat[{quaternion}]")
        elif isinstance(value, list):
            body = ", ".join(f"{item:+.4f}" if isinstance(item, float) else str(item)
                             for item in value[:8])
            lines.append(f"  {name:<28} [{body}{', ...' if len(value) > 8 else ''}] (len {len(value)})")
        elif isinstance(value, float):
            lines.append(f"  {name:<28} {value:+.6f}")
        else:
            lines.append(f"  {name:<28} {value!r}")
    return lines


def first_valid_row(table: pa.Table) -> int:
    for flag in ("available", "is_active"):
        if flag in table.schema.names:
            values = np.asarray(table[flag].to_pylist())
            hits = np.flatnonzero(values == True)  # noqa: E712
            if len(hits):
                return int(hits[0])
    if "calibration_state" in table.schema.names:
        values = table["calibration_state"].to_numpy(zero_copy_only=False)
        hits = np.flatnonzero(values == 3)
        if len(hits):
            return int(hits[0])
    return 0


def summarise(path: Path, label: str) -> str:
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    kinds: dict[str, int] = {}
    for field in schema:
        kinds[type_label(field.type)] = kinds.get(type_label(field.type), 0) + 1
    layout = " ".join(f"{count}x{kind}" for kind, count in sorted(kinds.items()))
    return (
        f"{meta.num_rows:>6} rows  {meta.num_columns:>3} cols  "
        f"{human_bytes(path.stat().st_size):>9}   {label}\n"
        f"{'':>18}{layout}"
    )


def labels(paths: list[Path]) -> list[str]:
    """Shorten paths against their common parent so same-named files stay distinct."""
    resolved = [path.resolve() for path in paths]
    if len(resolved) == 1:
        return [resolved[0].name]
    root = Path(os.path.commonpath([str(path) for path in resolved]))
    return [str(path.relative_to(root)) for path in resolved]


def describe(path: Path, args: argparse.Namespace) -> None:
    meta = pq.read_metadata(path)
    table = pq.read_table(path)
    limit = min(len(table), STATS_ROW_LIMIT)

    print("=" * 78)
    print(path)
    print(
        f"{human_bytes(path.stat().st_size)} · {meta.num_rows} rows · "
        f"{meta.num_row_groups} row group(s) · {meta.num_columns} columns · "
        f"created by {meta.created_by}"
    )
    if table.schema.metadata:
        print("file metadata:")
        for key, value in table.schema.metadata.items():
            text = value.decode("utf-8", "replace")
            print(f"  {key.decode()}: {text[:400]}{' ...' if len(text) > 400 else ''}")
    print("-" * 78)

    names = args.column or table.schema.names
    print(f"{'column':<28} {'type':<16} {'w':>3} {'nulls':>6}  detail")
    for name in names:
        if name not in table.schema.names:
            print(f"{name:<28} -- not present in this file --")
            continue
        field_type = table.schema.field(name).type
        column = table[name]
        nulls = column.null_count
        if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
            matrix, widths = as_matrix(column, limit)
            width = str(next(iter(widths))) if len(widths) == 1 else "?"
            detail = list_detail(matrix, widths)
        else:
            width = "-"
            detail = scalar_detail(column, field_type)
        print(f"{name:<28} {type_label(field_type):<16} {width:>3} {nulls:>6}  {detail}")
    if limit < len(table):
        print(f"(statistics computed on the first {limit} of {len(table)} rows)")

    stamps = timestamp_report(table)
    if stamps:
        print("-" * 78)
        print("timestamps:")
        print("\n".join(stamps))

    start = args.row if args.row is not None else first_valid_row(table)
    present = [name for name in names if name in table.schema.names]
    if args.rows and len(table) and present:
        print("-" * 78)
        for offset in range(args.rows):
            index = start + offset
            if index >= len(table):
                break
            print(f"row {index}:")
            print("\n".join(format_row(table, index, present)))
    print()


def collect(targets: list[str]) -> list[Path]:
    if targets:
        paths = []
        for target in targets:
            path = Path(target)
            paths.extend(sorted(path.rglob("*.parquet")) if path.is_dir() else [path])
        return paths
    sessions = sorted(Path(__file__).parent.glob("session_*"))
    if not sessions:
        raise SystemExit("no session_* directory found; pass a path explicitly")
    return sorted(sessions[-1].rglob("*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="parquet files or directories to inspect")
    parser.add_argument("--detail", action="store_true",
                        help="force the full report even for several files")
    parser.add_argument("--row", type=int, default=None,
                        help="first sample row (default: first valid frame)")
    parser.add_argument("--rows", type=int, default=1, help="number of sample rows to print")
    parser.add_argument("--column", action="append",
                        help="restrict the report to this column (repeatable)")
    args = parser.parse_args()

    paths = collect(args.paths)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(str(path) for path in missing)}")

    if len(paths) > 1 and not (args.detail or args.column):
        print(f"{len(paths)} parquet files\n")
        for path, label in zip(paths, labels(paths)):
            print(summarise(path, label))
        print("\npass a single file (or --detail) for the full schema and sample rows")
        return
    for path in paths:
        describe(path, args)


if __name__ == "__main__":
    main()
