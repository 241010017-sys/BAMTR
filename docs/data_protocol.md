# Data protocol

## Input boundary

The default input is a directory of daily CSV files named `2023-04-*.csv`, with a configured inclusive date range of April 1-14, 2023. The required mapping is defined in `data.columns`:

```yaml
columns:
  user: ID
  longitude: lng
  latitude: lat
  timestamp: time
  time_id: tid
```

Read user identifiers as strings to preserve leading zeros and avoid numeric precision changes. Coordinates must be geographic longitude and latitude in decimal degrees. Do not supply projected meters, swapped coordinates, or normalized model inputs in these fields.

The default `timestamp_source: auto` preserves `time` when it contains seconds. If the field has minute precision, the loader reconstructs within-day time as `tid * time_id_seconds`, with `time_id_seconds: 30.0`, and requires it to fall in the same minute as `time`. A disagreement raises an error. This handles the supplied April 1 file, whose readable timestamps have minute precision, without discarding the more precise timestamps in the other thirteen files.

`timestamp_source: time` always uses the readable timestamp; `timestamp_source: tid` always uses the within-day temporal identifier and takes the date from `time`. `tid` is not a UNIX timestamp. Dates and times use timezone-naive local civil time; the daily boundary is not inferred from machine time-zone settings. The model receives within-day seconds divided by 86,400.

The supplied files contain 487,454 raw rows and 1,000 distinct users, all present on each of the fourteen dates. The median raw user-day length is 31 records; the 95th percentile is 77 and the maximum is 186. These aggregate observations describe the supplied sample, not the manuscript's full population.

The loader groups records by user and day and sorts each trajectory chronologically. For duplicate resolved timestamps within a user-day, it keeps the last source row in sorted-file/input-row order. It never averages conflicting coordinates. Duplicate and conflicting-duplicate removal counts are written to the manifest. Invalid rows raise a file/row error rather than disappearing silently.

The default minimum is four distinct timestamped points per day. `max_points: null` keeps all eligible records; an explicit maximum selects evenly spaced chronological positions including both endpoints before labels or masks are constructed. Raw row counts, cleaned trajectory counts, and eligible target counts after history requirements are different quantities.

Under the default cleaning rules, the supplied private files produce 407,472 points across 13,688 eligible user-days. Preparation removes 79,328 duplicate-timestamp rows, of which 68,113 conflict with the preceding coordinate at that timestamp, and 312 short user-days containing 654 rows. No maximum-point truncation is applied in this count.

## User partition and normalization

Users are assigned to training, validation, and test subsets at a nominal 6:2:2 ratio using `data.split_seed`, default 42. Every trajectory belonging to one user remains in that user's subset.

The supplied collection's raw partition has 600/200/200 users. One training user has no eligible day after cleaning, leaving 599/200/200 users with eligible trajectories. Requiring at least one earlier eligible day yields 7,601/2,545/2,543 target samples. These counts describe the default full-data preparation; smoke limits or changed filtering produce different counts.

Longitude and latitude are independently min-max normalized using all eligible recorded points belonging to training users only. This includes training reference targets, which are permitted training data. The resulting statistics are reused for validation, testing, and prediction. Withheld validation/test coordinates must not be used to calculate per-trajectory means, ranges, centers, or scaling parameters. Values outside the training range are not clipped, and a constant coordinate axis uses a scale of one.

The user partition is different from a chronological holdout. A user-disjoint test on the same calendar window does not by itself measure performance on later unseen dates.

## Observation masks

`data.retention_rate` is the intended retained fraction, not a fixed stride or an independent Bernoulli probability. The main configurations use 0.2, 0.4, and 0.6. Retained positions are sampled uniformly without replacement, subject to the endpoint constraint. The implementation creates reproducible masks from `data.mask_seed`, default 42, independently of the model's training seed.

For a cleaned trajectory of length `n` and rate `r`, the retained count is `min(n - 1, max(2, floor(n * r + 0.5)))` when endpoints are kept. The two endpoints are retained and the remaining positions are drawn uniformly from the interior. With `keep_endpoints: false`, the minimum count is one and positions are drawn from the entire trajectory. At least one point remains missing. Selected indices are sorted before encoding.

The default `keep_endpoints: true` preserves the first and last records as interpolation anchors. This is an implementation choice because the manuscript does not specify endpoint treatment. A requested percentage and endpoint constraints can yield a higher realized retained fraction for short trajectories. Report the actual retained and missing counts; do not claim exact per-trajectory percentages when rounding or constraints change them.

Disabling endpoint retention can introduce missing targets outside the retained span in a benchmark. Treat this as a separate boundary-recovery experiment; the deployment prediction interface accepts interpolation queries only.

Training masks are fixed by default (`resample_train_mask: false`). Validation and test masks must remain fixed for comparisons. If training-mask resampling is enabled, retain the same validation/test masks and record the changed training protocol. Comparisons between models should use the same prepared user split and the same evaluation masks, not merely the same numeric model seed.

Each mask's random generator uses a stable SHA-256-derived seed from the mask seed, user-day sample identifier, and epoch. The epoch component is zero except for explicitly resampled training masks. Minibatch order does not change a mask. Masks at different retention rates are not guaranteed to form nested subsets.

Each target sample has distinct roles for retained observations, missing reference coordinates, reference semantic labels, and requested missing timestamps. Padding masks are also separate: padding is not an additional missing trajectory point.

## Historical inputs

For a target user and day, eligible histories are that user's recorded trajectories on strictly earlier dates. Same-day records and future days cannot be used as historical memory. Earlier trajectories of validation/test users are permissible inference context; they must not become training examples for learning model weights or tuning hyperparameters.

The default requires one historical day (`min_history_days: 1`) and does not impose an additional history cap (`max_history_days: null`). On a 14-day collection, the earliest date has no preceding history within that collection and cannot supply a main-protocol target. Later dates can have at most the earlier available days, subject to filtering.

Historical records are observations available before the target day; they should not be described as dense ground truth merely because they are used without the target day's synthetic mask. Record any historical downsampling, truncation, or maximum-point setting separately.

## Stay/move features and reference labels

Stay detection uses temporal duration and geographic spatial dispersion. The configured duration/radius pairs are:

| Retention | Duration | Radius |
| --- | ---: | ---: |
| 0.2 | 60 minutes | 150 meters |
| 0.4 | 90 minutes | 100 meters |
| 0.6 | 120 minutes | 100 meters |

The current-input stay indicators are recomputed from retained points after constructing each mask. Reference labels are generated separately from the original recorded target trajectory for supervision and evaluation. They do not enter an inference-time encoder or decoder.

These reference labels are algorithmically derived from the recorded trajectory, not independently annotated behavior. The input records are themselves irregular observations; intervals without records may conceal movement. Report this limitation when interpreting semantic precision and recall.

The implemented detector starts at a chronological anchor and extends a contiguous segment until the first point more than `R` meters from that anchor. Distances use Haversine with Earth radius 6,371,008.8 meters. If the first-to-last elapsed time is at least `T`, all segment points are marked stay and scanning resumes after the segment. Otherwise, the anchor advances by one. There is no separate maximum-gap rule. The detector is run independently on current retained observations, each earlier observed history, and the complete cleaned reference trajectory. Reference labels use the configured `T` and `R`, so changing a retention-specific configuration also changes its reference semantic task.

The manuscript leaves candidate-segment construction and spatial dispersion unspecified. This anchor-radius detector is therefore an implementation choice, not evidence that the authors' original label-generation code has been recovered. Long unobserved intervals within an anchor-radius segment can conceal movement. Validate thresholds and gap handling when adapting the method to a different dataset.

## Environmental features and calendar

Weather is disabled. Environmental context contains the first retained observation's day fraction, weekday divided by six, and a one-hot working/non-working/unknown day status.

The built-in calendar applies to mainland China in 2023. Official make-up working days override ordinary weekend status. Official holiday dates and remaining weekends are non-working days. Thus, the feature denotes working versus non-working status, not just named public holidays.

Source: [State Council General Office notice on the 2023 holiday arrangements, published December 8, 2022](https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html).

An optional external CSV can be supplied using `data.holiday_calendar_path`:

```csv
date,is_holiday
2024-01-01,1
2024-01-02,0
```

Values must be 0 for working and 1 for non-working; dates must be unique. Listed external dates override the built-in result. For other years, dates absent from the external file are marked unknown rather than guessed from weekend status.

## Recovery timestamps and evaluation

In the benchmark, the withheld record timestamps define the missing query set. Only their timestamps are available for output alignment; their coordinates and stay/move labels are evaluation targets.

In deployment, the caller supplies a desired timestamp grid or other query list. The program does not discover how many latent visits occurred, infer their times, or turn sparse records into independently observed ground truth. Queries are generated in chronological order, and retained observations remain present in the merged output.

Each prediction sample contains one user and one target day; the CLI can process multiple such groups from the same observation and query CSVs. Queries must fall on their matching day, strictly between the first and last retained timestamps, and must not overlap observations. Duplicate query timestamps are rejected; unique queries are sorted chronologically. Only histories for the matching user and strictly earlier dates are selected from a history directory. Queries without matching observations or outside the observed time span are rejected. This interface performs interpolation, not extrapolation.

The query CSV requires the configured user and timestamp columns and needs `tid` for minute-only timestamps in automatic mode. It does not require coordinates or semantic labels; those fields are ignored if present. If the optional history directory is omitted, prediction uses a current-only fallback. The output summary counts groups below the configured training-history minimum, so this fallback should not be mistaken for the main history-conditioned experiment.

Compute MAE, RMSE, and distance-threshold Recall over missing points only, pooling points across the evaluated set rather than averaging per-trajectory scores. The default geographic metric is a local Euclidean approximation with longitude scale fixed at the midpoint of the training latitude range. Haversine is an explicit configuration alternative. Both use Earth radius 6,371,008.8 meters. Metric outputs include MAE/RMSE in meters and kilometers, and Recall thresholds in meters. Model losses in normalized coordinate space are not geographic MAE or RMSE.

For point distances `d_i` in meters, `MAE = mean(d_i)`, `RMSE = sqrt(mean(d_i ** 2))`, and `Recall@r = mean(d_i <= r)`. Semantic probabilities are thresholded at 0.5 by default; stay is the positive class. Outputs also include the semantic confusion counts and the evaluated missing-point count.

## Data included in a public release

The prepared manifest records source-file hashes and the settings used for cleaning, partitioning, and constructing history lists. Loading a prepared collection with different preparation settings raises an error. Re-run preparation into a separate directory when changing filtering, history limits, user partitions, or timestamp handling. Retention rates, mask settings, stay thresholds, and calendar inputs can vary at sample construction time without re-cleaning the source data.

Only synthetic examples belong in the public package. Raw private trajectories, identifiers, prepared subsets, predictions, checkpoints trained on private trajectories, and diagnostic maps remain local unless separately reviewed and authorized for release. A hashed user identifier does not make the associated movement trace public data.
