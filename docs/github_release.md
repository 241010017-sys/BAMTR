# Preparing a GitHub release

This repository can be published as source code after reviewing the release contents. The generated package does not upload anything to GitHub automatically.

## Include

- Source code, packaging metadata, dependencies, and automated checks.
- English README, configuration examples, and method/protocol documentation.
- The MIT license for the newly written code.
- Synthetic CSV examples and their generation or provenance information.

## Keep local

- The original private mobile-phone CSV files and real user identifiers.
- Prepared data, cached histories, user-split manifests containing private identifiers, and normalization/provenance artifacts derived from private inputs.
- Private-data predictions, coordinate plots, logs containing identifiers or absolute private paths, and trained checkpoints unless separately approved for release.
- Virtual environments, Python caches, temporary notebooks, credentials, and unrelated files.

An ignore rule does not remove a file that is already tracked. Inspect both the working tree and Git's staged contents before publishing.

## Local checks

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m bamtr --help
python -m bamtr prepare --config configs/smoke.yaml
python -m bamtr train --config configs/smoke.yaml
```

Run the automated tests provided with the release. If private data are available locally, also run the small private-data configuration before a long experiment. A successful smoke run is a software check, not a paper-result reproduction.

## Review the staged release

After initializing a repository and staging the intended public files, inspect:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

Confirm that the sample data are synthetic and that no real participant files or research outputs were staged. A release archive should contain the same reviewed public source tree rather than an entire working directory with local runs.

## Describe the release accurately

The README should identify this code as a manuscript-guided implementation and list unresolved implementation conventions. Do not label the package the verified original experimental code or claim that its smoke run reproduces the paper's tables.

If future experiments reproduce a table, publish their exact configuration, measurement definitions, environment, and authorized result summaries. Keep code availability and private-data availability as separate statements. The code's MIT license does not make the source trajectories available for redistribution.

Before adding citation metadata, use the final author list, title, year, venue, and DOI supplied by the authors or publisher. Do not invent a DOI or claim journal acceptance from a draft manuscript.
