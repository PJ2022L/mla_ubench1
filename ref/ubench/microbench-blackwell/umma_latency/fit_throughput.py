#!/usr/bin/env python3
"""
Fit CyclesPerMMA = a + b/D to throughput data across pipeline depths.
Then estimate per-batch overhead: overhead = b + II - latency.
"""

import csv
import numpy as np

TPUT_CSV = '../umma_throughput/tput_results_full.csv'
LAT_CSV = 'lat_results_full.csv'
OUT_CSV = 'fit_results.csv'

# Read latency data
latency = {}
with open(LAT_CSV) as f:
    for row in csv.DictReader(f):
        key = (row['Format'], row['ABLayout'], row['CTAGroup'], row['M'], row['N'], row['K'])
        latency[key] = int(row['MedianCycles'])

# Group throughput data by config
configs = {}
with open(TPUT_CSV) as f:
    for row in csv.DictReader(f):
        key = (row['Format'], row['ABLayout'], row['CTAGroup'], row['M'], row['N'], row['K'])
        if key not in configs:
            configs[key] = []
        configs[key].append((int(row['PipelineDepth']), float(row['CyclesPerMMA'])))

# Fit a + b/D for each config, then compute overhead = b + a - latency
fields = ['Format', 'ABLayout', 'CTAGroup', 'M', 'N', 'K',
          'Latency', 'a (II)', 'b', 'Overhead']
rows = []

for key in sorted(configs.keys()):
    points = sorted(configs[key])
    D = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Fit y = a + b * (1/D)
    inv_D = 1.0 / D
    A = np.column_stack([np.ones_like(inv_D), inv_D])
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    lat = latency.get(key, None)

    # b = latency + overhead - II, so overhead = b + II - latency = b + a - latency
    overhead = round(b + a - lat, 1) if lat else ''

    rows.append({
        'Format': key[0], 'ABLayout': key[1], 'CTAGroup': key[2],
        'M': key[3], 'N': key[4], 'K': key[5],
        'a (II)': round(a, 2),
        'b': round(b, 1),
        'Latency': lat if lat else '',
        'Overhead': overhead,
    })

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f'Wrote {len(rows)} rows to {OUT_CSV}')

# Print examples
print(f'\n{"Config":<35} {"Latency":>8} {"a (II)":>8} {"b":>8} {"Overhead":>9}')
print('-' * 70)
for r in rows[:48]:
    label = f"{r['Format']} {r['ABLayout']} {r['CTAGroup']}SM M={r['M']} N={r['N']}"
    print(f'{label:<35} {str(r["Latency"]):>8} {r["a (II)"]:>8} {r["b"]:>8} {str(r["Overhead"]):>9}')
