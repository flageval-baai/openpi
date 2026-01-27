#!/usr/bin/env python3
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the first N action columns from a Parquet file."
    )
    parser.add_argument("parquet_path", help="Path to the Parquet file.")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to print (default: 5).",
    )
    parser.add_argument(
        "--limit-cols",
        type=int,
        default=5,
        help="Number of action columns to print (default: 5).",
    )
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        print("pyarrow is required to read Parquet files.", file=sys.stderr)
        return 1

    try:
        parquet_file = pq.ParquetFile(args.parquet_path)
    except (OSError, ValueError) as exc:
        print(f"Failed to open Parquet file: {exc}", file=sys.stderr)
        return 1

    all_columns = parquet_file.schema.names
    action_columns = [name for name in all_columns if name.startswith("action")]
    if not action_columns:
        print("No columns starting with 'action' were found.", file=sys.stderr)
        return 1

    selected_columns = action_columns[: args.limit_cols]
    table = parquet_file.read(columns=selected_columns)
    if args.rows is not None and args.rows >= 0:
        table = table.slice(0, args.rows)

    print("Selected columns:", ", ".join(selected_columns))

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print(json.dumps(table.to_pydict(), indent=2))
        return 0

    df = table.to_pandas()
    if args.rows is not None and args.rows >= 0:
        df = df.head(args.rows)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
