# Hopper (sm_90) Metric Name Reference

Use this reference for H100 and H200 reports. Hopper is compute capability 9.0; compile architecture-accelerated code for `sm_90a`, but query NCU as `gh100`. Do not assume an H100/H200 SKU has a particular SM count, clock, or memory bandwidth.

Metric availability is a property of the **NCU version, driver, selected sections, and report**, not just the GPU architecture. A missing metric is not evidence of zero activity. Confirm it before using it:

```bash
ncu --query-metrics --chip gh100 | grep -E "(warps_issue_stalled|pipe_tensor|t_sector)"
```

```python
all_names = action.metric_names()
```

The helpers return `None` for a metric that was not collected. Interpret that as “collect or query it”, never as “the hardware did nothing”.

---

## Hopper-specific evidence

Use generic tensor-pipe utilization to assess compute throughput. Confirm Hopper mechanisms from source/SASS rather than relying on non-portable, version-specific metric names:

| Mechanism | Evidence to collect | Interpretation |
|---|---|---|
| WGMMA | Source/SASS contains `wgmma`; tensor-pipe utilization and `wait` stalls | A four-warp warpgroup must execute cooperatively. High `wait` can mean insufficient independent WGMMA work or a shallow pipeline. |
| TMA | Source/SASS contains `cp.async.bulk.tensor`; inspect `mbarrier` waits and timeline | Use only where tile shape, reuse, and pipeline depth amortize descriptor and synchronization overhead. |
| Thread-block cluster / DSMEM | Launch attributes plus source use of cluster APIs; measure end-to-end duration | It is worthwhile only when it removes material global-memory traffic or synchronization. |
| FP8 | Input data type, library/kernel configuration, tensor-pipe utilization, numerical validation | The profile cannot prove numerical acceptability; validate output separately. |

---

## Curated Hopper metric set

These are the names used by `helpers/ncu_utils.py` for H100/H200 reports. Query the installed NCU before adding a custom metric, and collect the matching section before reading it.

### Launch geometry / occupancy

```
launch__grid_size
launch__block_size
launch__grid_dim_x, launch__grid_dim_y, launch__grid_dim_z
launch__block_dim_x, launch__block_dim_y, launch__block_dim_z
launch__thread_count
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count
device__attribute_max_warps_per_multiprocessor
sm__maximum_warps_per_active_cycle_pct
```

### Throughput, timing, and warp activity

```
gpu__time_duration.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed
smsp__cycles_active.avg
smsp__issue_active.avg.per_cycle_active
sm__warps_active.avg.pct_of_peak_sustained_active
sm__warps_active.avg.per_cycle_active
smsp__warps_eligible.avg.per_cycle_active
sm__inst_executed.avg.per_cycle_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
```

### Memory and cache behavior

```
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
```

Compute `sectors / requests` for loads and stores. Four 32-byte sectors per 128-byte warp request is ideal for a fully coalesced access; do not apply that heuristic to irregular gather/scatter work.

### Instruction and spill counters

```
smsp__sass_inst_executed_op_global_ld.sum
smsp__sass_inst_executed_op_global_st.sum
smsp__sass_inst_executed_op_local_ld.sum
smsp__sass_inst_executed_op_local_st.sum
smsp__sass_inst_executed_op_shared.sum
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio
```

### Stall reasons

```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
```

### Per-PC and PM-sampling counters

Collect per-PC counters with `--set source --section SourceCounters`:

```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_lg_throttle
smsp__pcsamp_warps_issue_stalled_not_selected
smsp__pcsamp_warps_issue_stalled_selected
smsp__pcsamp_warps_issue_stalled_membar
```

For a timeline, collect `PmSampling` and `PmSampling_WarpStates`; use `pmsampling:smsp__warps_issue_stalled_<reason>.avg` only if it has instances. The sampling interval and available series vary by NCU/driver combination.

---

## Gotchas

1. `None` usually means the metric was not collected or was renamed; query the report before changing code.
2. `0` can mean no activity, but can also be a derived value whose prerequisites were not collected.
3. Prefer `pct_of_peak_sustained_elapsed` for end-to-end utilization and `_active` to distinguish an idle-SM/tail problem from an active-SM bottleneck.
4. Use `action.source_info(pc)` only with a `-lineinfo` build. Treat WGMMA/TMA source and SASS as the authoritative feature evidence.
