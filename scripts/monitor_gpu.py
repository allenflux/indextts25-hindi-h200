#!/usr/bin/env python3
"""Record nvidia-smi samples to CSV during a separately launched smoke test.

This measures device-wide observed memory, not torch allocator peak memory.
It does not start a trainer, SSH to a machine, or install NVIDIA software.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

QUERY = "index,uuid,memory.total,memory.used,utilization.gpu,power.draw"
HEADER = ["utc", "elapsed_seconds", "gpu_index", "gpu_uuid", "memory_total_mib",
          "memory_used_mib", "utilization_percent", "power_watts"]


def parse_sample(stdout):
    rows = list(csv.reader(io.StringIO(stdout)))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError("Expected one NVIDIA GPU sample with six fields")
    row = [value.strip() for value in rows[0]]
    if not row[0].isdigit() or not row[1].startswith("GPU-"):
        raise ValueError("Unexpected GPU identity in nvidia-smi output")
    for value in row[2:]:
        if value != "[N/A]":
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("Invalid negative or nonfinite GPU telemetry")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=3600)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if not 0 <= args.gpu <= 255 or not 0.1 <= args.interval <= 3600 or not 1 <= args.samples <= 100000:
        p.error("Use GPU 0..255, interval .1..3600 seconds, samples 1..100000")
    started = time.monotonic()
    observed_max = 0.0
    samples = 0
    try:
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            for index in range(args.samples):
                result = subprocess.run(["nvidia-smi", "--id", str(args.gpu),
                                         "--query-gpu=" + QUERY, "--format=csv,noheader,nounits"],
                                        check=True, capture_output=True, text=True, timeout=10)
                row = parse_sample(result.stdout)
                writer.writerow([datetime.now(timezone.utc).isoformat(),
                                 f"{time.monotonic() - started:.6f}", *row])
                stream.flush()
                if row[3] != "[N/A]":
                    observed_max = max(observed_max, float(row[3]))
                samples += 1
                if index + 1 < args.samples:
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "telemetry_failed", "reason": str(exc), "samples": samples}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "telemetry_recorded", "samples": samples,
                      "observed_device_memory_max_mib": observed_max,
                      "torch_peak_memory_measured": False, "training_success_asserted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
