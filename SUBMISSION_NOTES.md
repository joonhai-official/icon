# Submission Notes

Guide for submitting this repository alongside the three companion papers.

## arXiv submission (three papers, same day)

The three papers form one submission batch:

1. **Park (2026, philosophical)** — *Information Flow and Self-Reference Across Substrates: On Self-Reference, Substrate, and the Need for Measurement*
2. **Park (2026a, empirical)** — *Information Capacity in Neural Networks: An Empirical Study of κ Scaling*
3. **Park (2026, spec)** — *Icon: A Framework for Measuring Information Flow*

### Comment field text for the spec paper (paper 3)

After all three are uploaded and IDs are received, set the spec paper's arXiv comment field to:

```
46 pages. Framework specification for measuring information flow in
representation-bearing systems.
Reference implementation: https://github.com/joonhai-official/Icon
Companion papers: arXiv:XXXX.AAAAA (philosophical), arXiv:XXXX.BBBBB (empirical).
```

Replace `XXXX.AAAAA` and `XXXX.BBBBB` with the IDs received for papers 1 and 2.

### Optional v2 (recommended)

After the three papers are live and IDs are known, optionally upload a v2 of the spec PDF with the References section updated to use real arXiv IDs in place of `Companion submission` placeholders. Two replacements:

- `Park, J. (2026a). ... Companion submission.` → `Park, J. (2026a). ... arXiv:XXXX.BBBBB [cs.LG].`
- `Park, J. (2026b). ... Companion submission.` → `Park, J. (2026b). ... arXiv:XXXX.AAAAA [cs.LG].`

The in-text `(Park, 2026a)` and `(Park, 2026b)` citations stay as is.

## GitHub setup

### Create the repository

1. Go to https://github.com/new
2. Repository name: **`Icon`**
3. Owner: **`joonhai-official`**
4. Visibility: **Public**
5. Do NOT initialize with README, .gitignore, or LICENSE — this repository provides them.

### Push the code

```bash
# Clone the empty repo locally
git clone https://github.com/joonhai-official/Icon.git
cd Icon

# Copy this repository's contents in (extracted from the zip)
cp -r /path/to/icon_framework/* .
cp -r /path/to/icon_framework/.gitignore .

# First commit
git add .
git commit -m "Initial commit: v0.1.0 alpha release alongside framework specification"

# Tag the release
git tag -a v0.1.0 -m "v0.1.0: initial alpha release"

# Push
git push origin main
git push origin v0.1.0
```

### Recommended repository settings

After pushing:

- **About** section (sidebar on GitHub):
  - Description: `A framework for measuring information flow in representation-bearing systems`
  - Website: link to the arXiv spec paper once available
  - Topics: `information-theory`, `mutual-information`, `neural-networks`, `measurement-framework`, `representation-learning`, `infonce`, `python`

- **Releases**:
  - Click "Releases" → "Draft a new release"
  - Tag: `v0.1.0`
  - Title: `v0.1.0 — Initial alpha`
  - Description: paste the [v0.1.0] section from `CHANGELOG.md`

## Verification before pushing

Run these once before the first push:

```bash
# 1. Tests pass
pytest                              # expect: 44 passed, 6 skipped
pytest -m "not slow"                # expect: 34 passed, 6 skipped, 10 deselected

# 2. Examples run end-to-end
python examples/mnist_mlp.py        # expect: full output, manifest saved

# 3. Package installs cleanly
pip install -e .                    # in a fresh virtualenv

# 4. Public surface imports work
python -c "import icon; print(icon.__version__, len(icon.__all__))"
# expect: 0.1.0 20
```

If any of these fail, do not push. Fix the issue first.

## After pushing

- Verify the README renders correctly on GitHub (especially LaTeX in citation block).
- Verify the LICENSE file is recognized by GitHub (Apache 2.0 badge should appear).
- Verify `pip install git+https://github.com/joonhai-official/Icon.git` works from a fresh environment.

## PyPI (later, not in v0.1.0)

PyPI registration is deferred until the code stabilizes (target: v0.2 or later). Package name `icon-framework` is reserved by intent; check availability before registering:

```bash
pip search icon-framework  # may be deprecated; check https://pypi.org/project/icon-framework/
```
