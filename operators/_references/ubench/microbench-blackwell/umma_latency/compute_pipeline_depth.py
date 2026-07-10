#!/usr/bin/env python3
"""Compute saturated pipeline depth from latency and throughput data.

D_sat = ceil(latency / II), where II = CyclesPerMMA at max pipeline depth.
"""

import csv
import math

LAT_CSV = 'lat_results_full.csv'
TPUT_CSV = '../umma_throughput/tput_results_max.csv'
OUT_CSV = 'pipeline_depth.csv'

# Read latency data
latency = {}
with open(LAT_CSV) as f:
    for row in csv.DictReader(f):
        key = (row['Format'], row['ABLayout'], row['CTAGroup'], row['M'], row['N'], row['K'])
        latency[key] = int(row['MedianCycles'])

# Read measured II (CyclesPerMMA at max pipeline depth)
measured_ii = {}
with open(TPUT_CSV) as f:
    for row in csv.DictReader(f):
        key = (row['Format'], row['ABLayout'], row['CTAGroup'], row['M'], row['N'], row['K'])
        measured_ii[key] = float(row['CyclesPerMMA'])

# Join and compute
fields = ['Format', 'ABLayout', 'CTAGroup', 'M', 'N', 'K', 'Latency', 'InitiationInterval', 'SaturatedPipelineDepth']
rows = []
for key in sorted(latency.keys()):
    if key not in measured_ii:
        continue
    lat = latency[key]
    ii = measured_ii[key]
    d_sat = math.ceil(lat / ii)
    rows.append({
        'Format': key[0], 'ABLayout': key[1], 'CTAGroup': key[2],
        'M': key[3], 'N': key[4], 'K': key[5],
        'Latency': lat, 'InitiationInterval': round(ii, 2),
        'SaturatedPipelineDepth': d_sat,
    })

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f'Wrote {len(rows)} rows to {OUT_CSV}')
