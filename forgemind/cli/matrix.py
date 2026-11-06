"""Run the benchmark over several configs and print a comparison table.

  python -m forgemind.cli.matrix --quant fp8 --backends te torchao --preset small --seq 1024
  python -m forgemind.cli.matrix --quant none fp8 --backends auto
Runs each combination in-process; a failing combination is reported, not fatal."""
from __future__ import annotations
import argparse
import itertools
import sys
from .bench import main as bench_main


def main(argv=None):
    p = argparse.ArgumentParser(prog="forgemind-matrix")
    p.add_argument("--quant", nargs="+", default=["none"], choices=["none", "fp8", "int8", "int8_wo"])
    p.add_argument("--backends", nargs="+", default=["auto"], choices=["auto", "te", "torchao"])
    args, rest = p.parse_known_args(argv)
    rows = []
    for q, b in itertools.product(args.quant, args.backends):
        if q == "none" and b != args.backends[0]:
            continue  # backend is irrelevant without quantization
        print(f"\n##### quant={q} qbackend={b}", flush=True)
        try:
            r = bench_main(rest + ["--quant", q, "--qbackend", b])
            rows.append(r)
        except Exception as e:
            print(f"FAILED quant={q} qbackend={b}: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({"quant_mode": q, "quant_requested": b, "error": str(e)[:80]})
    print("\n=== comparison ===")
    print(f"{'quant':<8}{'requested':<10}{'resolved':<10}{'tok/s':>12}{'p50 ms':>10}{'mem GB':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['quant_mode']:<8}{r['quant_requested']:<10}ERROR: {r['error']}")
        else:
            print(f"{r['quant_mode']:<8}{r['quant_requested']:<10}{r['quant_resolved']:<10}"
                  f"{r['tok_s']:>12,.0f}{r['step_ms_p50']:>10.1f}{r.get('peak_mem_gb', float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
