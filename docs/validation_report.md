# Delivery validation report

Validation date: September 30, 2026.

This report records software checks performed on this package. It is not an accuracy benchmark or a reproduction of the manuscript's experimental tables.

## Environment actually exercised

- Windows, Python 3.12.14.
- PyTorch 2.3.0+cpu, NumPy 1.26.4, pandas 2.2.2.
- PyYAML 6.0.1 and Matplotlib 3.9.2.
- CPU execution. CUDA execution and the hosted GitHub Actions job were not exercised locally.

## Completed checks

| Check | Outcome |
| --- | --- |
| Automated suite | 39 tests passed |
| Python package build | A `bamtr-0.1.0-py3-none-any.whl` wheel built successfully |
| Archive installation | Extracted the release into a fresh directory, installed it, and completed the prepare/train/evaluate/predict quickstart using the installed package |
| Synthetic CSV preparation | 5,760 generated records; 20 synthetic users across six days |
| Synthetic training | Two short epochs with the documented smoke configuration |
| Synthetic held-out evaluation | 380 withheld points; prediction export and metric aggregation completed |
| Independent deployment prediction | 31 query points recovered and 17 observed points preserved |
| Loss/Recall plotting | Plots generated from actual smoke-run logs and metrics |
| Repeated-seed command | Two short runs completed with fixed data/mask seeds; aggregation files produced |
| Supplied private CSV preparation | All 14 files parsed; raw and cleaned counts verified |
| Private-data execution check | Small-model CPU training/validation and limited held-out evaluation completed |
| Default-sized model execution | The 1,276,147-parameter default model completed a real-data training batch and validation batch |

The smoke configurations deliberately reduce capacity, sequence/history limits, batches, and epochs. The default-sized model check also used only one training batch and one validation batch. These checks establish that the implementation executes against the supplied schema; they do not establish convergence or paper-level performance.

## Behavioral coverage

Tests check user-disjoint partitions, strictly earlier same-user history, training-only normalization, deterministic masks, timestamp reconstruction, duplicate handling, prepared-configuration compatibility, sparse-only mobility indicators, and missing-target alignment.

Model tests check padding and NaN isolation, absent historical memory, effective single-layer dropout, label exclusion during inference, teacher forcing, optional modules, and explicit query-time conditioning. Metric tests independently verify physical distances, point-weighted aggregation, and masked coordinate/semantic losses.

End-to-end tests compare predicted CSV rows with independently constructed missing targets, reload checkpoints, and independently recalculate the reported spatial metrics. An interrupted tiny two-epoch run resumes to exactly the same model tensors and non-timing log entries as uninterrupted training in the tested CPU environment.

Deployment tests verify that retained coordinates are unchanged, query coordinate/label columns are ignored even if present, and target-day, future, and unrelated-user history rows are excluded by the CLI wrapper.

## Private-data audit totals

The supplied collection contains 487,454 rows for 1,000 users. Default preparation removes 79,328 duplicate timestamp rows and 312 short user-days comprising 654 remaining points. Without the development-only point cap, 407,472 points remain across 13,688 user-days. Raw user split counts are 600/200/200; eligible target counts are 7,601/2,545/2,543.

The code-only `BAMTR_GitHub_Complete.zip` excludes raw data, prepared private arrays, private prediction files, and trained checkpoints. Its distributed example CSV records are generated synthetically. The separately supplied `BAMTR_Local_With_Data.zip` also includes the user's 14 original daily CSV files in `data/raw` for local execution; these files are not synthetic or covered by the code license.

## Local data bundle check (October 1, 2026)

- All 14 daily CSV files were copied without modification; every copy matched its source SHA-256 hash. The CSV files total 40,157,162 bytes.
- Preparation using the default relative `data/raw` path successfully read all 487,454 records and produced the counts reported above.
- The 20%, 40%, and 60% configurations each loaded training batches with current observations, earlier historical trajectories, and stay/move labels from the local project folders.
- Existing training artifacts were preserved. This packaging check did not run another training experiment. The local archive excludes prepared arrays and training runs; run `prepare` after extracting a fresh copy.

## Not established by these checks

The default 30-epoch experiment, all 15 paper-oriented seed/rate runs, GPU memory/latency measurements, full baseline comparisons, and agreement with the paper's accuracy tables remain experiments to run under the finalized protocol. The hidden-size convention and other manuscript ambiguities are documented in [reproducibility.md](reproducibility.md).
