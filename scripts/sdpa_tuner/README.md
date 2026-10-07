# SDPA forward config tuner

Tools for the model-based tile selection in `src/gpu/intel/sdpa/select.cpp`.

The library picks the microkernel tile configuration of the fused forward
SDPA kernel in one of three ways, chosen with the dev-mode environment
variable `SDPA_CONFIG_SELECT`:

| value    | behaviour                                                        |
|----------|------------------------------------------------------------------|
| `legacy` | hand-tuned table in `configs.cpp` (default)                      |
| `table`  | full-key lookup table, legacy fallback                           |
| `model`  | lookup table, then the cost model's best candidate, then legacy  |

Other dev-mode knobs:

- `SDPA_CONFIG_DUMP_CANDIDATES=N` prints the N best candidates with their
  estimates (`-1` for all) as `fwd_candidate,...` debuginfo lines.
- `SDPA_CONFIG_TABLE_FILE=path` adds `key config` lines to the lookup table.
  A key may end in `:eu<N>` to apply only to devices with N EUs; such a
  line is preferred over the arch-wide one (`src:table_dev` in the verbose
  line). The tuner writes tagged lines with `--table-tag-device`.
- `SDPA_MODEL_COEFS=name=value,...` overrides cost-model coefficients.
- `SDPA_CONFIG=8 ints` forces a config (existing knob).

Every selection prints one `fwd_select,...` line at `ONEDNN_VERBOSE=debuginfo=4`
carrying the lookup key, the chosen config and its derived quantities, the
device description (`hw:`) and the problem description (`prb:`).

## Files

- `sdpa_model.py`: ctypes binding to the cost model in the library. The model
  exists once, in `select.cpp`; a DNNL_DEV_MODE build exports it through the
  `dnnl_impl_sdpa_fwd_*` functions in `select_api.cpp`. The scripts find
  `build/src/dnnl.dll` or `libdnnl.so` under the repo, or take `--lib` /
  `$DNNL_LIB`.
- `tune_fwd_config.py`: runs benchdnn's `--sdpa` driver over problems and
  configs and appends rows to a CSV.
- `fit_fwd_model.py`: reports regret/rank correlation of the model and fits
  coefficients. `--check-recorded` flags a CSV whose recorded estimates came
  from a different model version than the loaded library.
- `analyze_landscape.py`: top configs and one-factor marginals per problem.

## Workflow

1. Build with `DNNL_DEV_MODE=ON` and the GPU runtime.
2. Sweep. Either list problems or draw a design:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn.exe \
           --design 60 --seed 1 --dtype f16 --topk 12 --random 8 \
           --out sweep_a750.csv --table-out table_a750.txt

   The reference config is re-timed every `--ref-every` runs; stop if the
   drift warning fires repeatedly (the part is throttling). Measurements use
   benchdnn perf mode P with `--max-ms` per config (default 300 ms); the
   10 ms fast mode (`--perf-mode F`) is quicker but leaves the GPU clock
   unsettled and measured 1.5x off on the A750.
   Candidates that fail to build show `status=fail`; add any such family to
   `fwd_describe()` and `sdpa_model.describe()` so the tiler stops emitting
   them.
3. Score the model and fit:

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --check-recorded -v
       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1

   Try the fitted coefficients without rebuilding by exporting the printed
   `SDPA_MODEL_COEFS` value, then paste the C++ block into `seed_coefs()`.
4. Promote verified winners to the lookup table: the `--table-out` lines can
   be tried through `SDPA_CONFIG_TABLE_FILE` and then added to
   `fwd_table_data[]` in `select.cpp`.
5. Second SKU of an architecture (A770 after A750, B570 after B580): rerun
   the same problems in `table` mode against `legacy`. Shapes that regress
   get a device-tagged line from a sweep with `--table-tag-device`; the
   arch-wide lines stay shared. Occupancy-bound shapes with few work-groups
   are the ones that differ between SKUs.

## Metrics

Regret is `time(config the model ranks first) / time(best measured config)`
per problem. Targets before switching the default away from `legacy`:
geomean regret at or below 1.05, worst case at or below 1.15, and no shipped
table shape slower than 0.97x of today.

Fit with `--folds 4 --restarts 1` and quote the cross-validated line: it is
the out-of-sample regret over every problem, each scored by a fit that did
not see it. The in-sample `fitted/train` line is always better and is only
the number to bake.

## Model form and A750 results

The cost model (`fwd_estimate_cost` in `select.cpp`, mirrored in
`sdpa_model.py`) is, per subgroup and per key iteration: a systolic k-block
chain (each k-block issues `(unroll_m/8) x (unroll_n/simd)` dpas and takes at
least the pipe latency), K/V loads as one message per k-block of `unroll_m`
rows (latency or bandwidth share), softmax, the S round trip through SLM and
two barriers whose cost grows with the subgroups taking part. Work-groups
are scheduled in waves with a contention term;
memory time is bytes over bandwidth with the achieved bandwidth bounded by
bytes in flight. Compute and memory combine with a smooth max. The FMA path
keeps a flop-based term that no sweep has covered yet.

A750 (xe_hpg), 22 problems, 983 timed configs including full landscapes of
seven problems. The data are the CSVs the workflow above writes (`--out`
of a top-k sweep and of an `--all` landscape sweep); they are not kept in
the tree. Rerunning the two sweeps and `fit_fwd_model.py --folds 4`
reproduces the table:

| selector | regret geomean | within 5% | within 15% | worst |
|---|---|---|---|---|
| legacy table | 1.162 | 50% | 64% | 2.33 |
| model, cross-validated | 1.127 | 41% | 64% | 1.58 |
| model, fitted on all (baked seeds) | 1.087 | 59% | 77% | 1.54 |
| built-in table hit | 1.000 by construction | | | |

The model's top 8 plus 4 random candidates contained the best measured config
on every problem, which is how the 22 built-in table entries were produced.
Other architectures use first-principles seeds until swept.
