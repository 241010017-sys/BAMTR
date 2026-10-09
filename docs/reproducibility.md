# Reproducibility and implementation choices

## Status of this repository

This is a manuscript-guided implementation produced for the supplied 14-day CSV schema. It is not a recovered copy of the authors' original experimental software. The manuscript supplies the scientific design and several training settings, but it does not uniquely specify every implementation detail.

Software checks and smoke runs establish execution behavior. They do not establish reproduction of the manuscript's accuracy tables, efficiency measurements, or external baseline comparisons. Results from a private-data run must be recorded with their actual configuration, dataset counts, model size, seeds, and environment.

## Paper-specified settings

| Item | Supplied setting |
| --- | --- |
| Main date window | April 1-14, 2023 |
| User split | Training/validation/test at 6:2:2 |
| Current and historical encoder | Shared bidirectional GRU with attention pooling |
| History boundary | Same user, strictly earlier days |
| Prediction targets | Missing points only, aligned to withheld timestamps |
| Attribute projection width | 30 with Tanh |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Epochs | 30 |
| Batch size | 256 |
| Dropout | 0.5 |
| Semantic-loss weight | 10 |
| Teacher-forcing endpoints | 0.8 and 0.3 |
| Repeated-run seeds | 42, 43, 44, 45, 46 |
| Main location Recall threshold | 100 meters |
| Additional Recall thresholds | 50, 300, 500, 1000, 2000 meters |

The sample-rate-specific stay thresholds are listed in [data_protocol.md](data_protocol.md). They are reported manuscript settings, not a new validation search performed by this repository.

## Choices required to make the specification executable

| Topic | Implementation choice or requirement | Interpretation |
| --- | --- | --- |
| Hidden size | Default encoder 128 per direction, concatenated width 256, decoder 256 | The prose's hidden size 256 and the equations' `2H` notation are ambiguous. Both widths are configurable. |
| Weather | Disabled | User-requested difference from the weather-inclusive manuscript. |
| Global time context | First retained observation's time of day | A single trajectory-level hour is not uniquely defined in the manuscript. |
| Point time scale | Seconds since local midnight divided by 86,400 | A dimensionless within-day encoding, rather than an unscaled temporal identifier. |
| Timestamp precision | Preserve second-resolution time; validate and use the 30-second temporal identifier for minute-only time | Resolves the supplied first day's lower timestamp precision. |
| Duplicate timestamps | Keep the last source record; never average conflicting coordinates | Makes the target sequence unambiguous and is recorded in preparation counts. |
| Calendar feature | Working/non-working/unknown with 2023 make-up days | Date-derived context; not measured weather or an inferred activity label. |
| Main downsampling | Deterministic random masks with endpoint retention by default | The manuscript does not uniquely define its original masking algorithm. |
| Filtering | At least four recorded points per day by default; other limits configurable | A practical sample-validity rule rather than a documented paper cohort criterion. |
| History availability | At least one earlier day by default | Avoids silently evaluating an unspecified cold-start regime. |
| Stay detection | An explicit segment detector in the implementation | The paper's duration/radius description does not uniquely define a detector. |
| Reference labels | Derived from recorded target trajectories | Proxy labels, not independent behavior annotation. |
| Single-layer dropout | Explicit external dropout | Ensures dropout is active for a single-layer recurrent model. |
| Decoder queries | Missing timestamps are supplied; direct query-time conditioning defaults off | Default follows the input list in Equation (22). Turning the option on is an extension. |
| Coordinate-head intermediate width | 256 by default | The paper specifies a two-layer MLP but not its complete width. |
| Gradient clipping | Norm limit 5.0 by default | Training-stability choice not uniquely specified by the manuscript. |
| Early stopping | Disabled by default | A maximum of 30 epochs is used unless the configuration changes it. |
| Physical distance | Local Euclidean by default; Haversine available | Latitude/longitude must be converted into a meaningful distance; the exact original projection is unspecified. |

These choices must remain visible in any accompanying methods description. Changing them to improve a score creates a new configuration that requires validation before its test set is evaluated.

## Loss reduction and padding

Coordinate loss is the average squared two-coordinate error over valid missing points. Squared latitude and longitude components are summed for each point, following Equation (25). A mean over both points and coordinate components differs by a factor of two and changes the effective balance with the semantic loss.

The implemented semantic objective is **mean BCE over valid missing points**, and the total loss is `coordinate_mean + lambda_semantic * semantic_mean`. The manuscript writes Equation (26) as a BCE sum. Dividing that term by the number of valid targets is an explicit implementation choice that makes the objective less dependent on batch size and sequence length. Consequently, the numerical weight `lambda_semantic: 10` should not be assumed equivalent to a literal unnormalized-sum implementation of Equation (26).

Padding and retained points are excluded from both losses and metrics. Teacher forcing is used during training only and feeds selected previous reference coordinates to the next decoding step. Reference semantic labels never feed the decoder, including during training; semantic feedback uses the previous predicted stay probability. Validation and test scores use no missing-coordinate or missing-state targets as model inputs.

## Repeatable comparisons

1. Prepare and record a single user partition with `data.split_seed: 42`.
2. Keep the evaluation masks fixed with `data.mask_seed: 42` and reuse them across compared runs.
3. Select hyperparameters and checkpoints using validation data only.
4. Train with the separate model seeds 42-46. Do not change the user partition merely because the model seed changes.
5. Evaluate the frozen selected configuration on test users.
6. Aggregate actual results across independent runs and report the aggregation convention and number of valid samples.

If a seed fails, record and resolve that failure instead of silently dropping an unfavorable result. A standard deviation across training seeds measures training variability; it does not provide calibrated uncertainty for one missing interval.

## What to retain for each run

- The resolved configuration, including defaults and command-line overrides.
- Input provenance and prepared-data counts, user partition, observation-mask settings, and normalizer statistics.
- The selected checkpoint, optimizer and random-number-generator state required for resumption, and epoch history.
- Validation and test metrics with geographic units, Recall thresholds, and missing-point counts.
- Prediction records and the exact target timestamps used for evaluation.
- The actual model parameter count, hardware, Python/PyTorch versions, and deterministic settings.

Run artifacts produced from private input can contain sensitive information. Preserve them locally for reproducibility; they do not automatically belong in the public GitHub release.

Training writes `config.yaml`, `environment.json`, `parameters.json`, `normalizer.json`, `provenance.json`, and `history.csv` in its run directory. `latest.pt` supports continuation of the configured schedule. `best.pt` is selected by validation MAE, and `best_validation.json` records that checkpoint's validation metrics. Evaluation writes `metrics.json` and point-level `predictions.csv` in its output directory. Evaluation verifies the prepared manifest and training normalizer; use prediction rather than held-out evaluation for a new input collection.

Mid-run continuation requires `latest.pt` from the same run directory and the original model/data/evaluation settings, seed, and training protocol. Device, worker count, and CPU thread count may change. The original total epoch count and teacher-forcing schedule remain fixed; a completed run cannot be extended with the resume command. Starting a new run with changed settings is a separate experiment. Deterministic settings improve repeatability but do not guarantee bitwise agreement across different hardware or PyTorch versions.

The resume path restores optimizer and random-number-generator states and truncates epoch-log rows written beyond the last saved checkpoint. It preserves checkpoint selection by validation MAE. Use the resolved `config.yaml` in the run directory to avoid losing earlier command-line overrides.

## Accuracy, parameter count, and efficiency

Measure parameter counts from the constructed model. Do not adjust an architecture solely to obtain a number printed in the manuscript. Sharing a module across histories means its weights are counted once.

Latency and GPU-memory results depend on sequence lengths, history length, batch size, device, warm-up, synchronization, dtype, and whether historical encoding is included. A CPU smoke-run duration or model parameter count cannot reproduce the manuscript's A100 timing and memory table.

Different algorithms' reported MAE, RMSE, and threshold Recall must use the same masks and aggregation population. Validate metric consistency and units before reporting relative improvements.

## Scope limitations

The available 14-day files support experiments within their actual observation window. Full-month, cross-city, weather-specific, and seasonal claims require additional data and evaluation.

The model produces a single coordinate sequence and stay probabilities. It does not supply a calibrated distribution over alternative trajectories, a road-feasibility guarantee, or a validated physical motion model. Those require additional modeling and evaluation.

External baselines, structured missingness tests, counterfactual-history experiments, and the paper's other figures require their own complete protocols and validated implementations. A main BAMTR training run alone does not reproduce them.
