# Publish this project on GitHub

The project is published publicly at
[bdjjo/gpt2-from-scratch](https://github.com/bdjjo/gpt2-from-scratch), with `main` as
the default branch. [GitHub Actions](https://github.com/bdjjo/gpt2-from-scratch/actions/workflows/ci.yml)
shows automated test results for each published commit.

## Repository presentation

- **Name:** `gpt2-from-scratch`
- **Description:** `GPT-style language modeling in clean PyTorch: byte BPE, explicit causal attention, reproducible CPU training, and measured evaluation.`
- **Topics:** `pytorch`, `transformer`, `gpt`, `language-model`, `bpe`, `deep-learning`, `machine-learning`, `reproducibility`
- **Visibility:** Public.
- **License:** MIT for the project code; data provenance is documented separately.

Pin the repository on your profile after publishing. The README, measured report,
and generated samples should be the first things a reviewer can find.

## Initial publication with Git (completed)

The instructions below document the initial import, which has already been completed
for this workspace. Do not recreate the repository or add `origin` again. To inspect
the current connection:

```powershell
git status --short
git remote -v
```

The ignore file excludes `.venv`, downloaded/prepared data, checkpoints, and local
temporary files. Include source, docs, configs, tests, and the small reports folder.

Create an **empty** repository under your GitHub account. Leave GitHub's “add README,”
license, and `.gitignore` options unchecked because these files already exist here.
For this published project, the HTTPS remote is
`https://github.com/bdjjo/gpt2-from-scratch.git`. The initial import used the following
commands; they are retained as a reference for publishing a different new repository.

```powershell
git add .github .gitattributes .gitignore configs data/README.md docs reports scripts src tests
git add README.md LICENSE CONTRIBUTING.md pyproject.toml requirements.txt requirements-dev.txt requirements-lock-windows-py312.txt train.py generate.py evaluate.py
git diff --cached --stat
git commit -m "Build GPT-style language model with reproducible experiments"
git branch -M main
git remote add origin https://github.com/bdjjo/gpt2-from-scratch.git
git push -u origin main
```

These commands assume `origin` is absent and the remote is new and empty. If it already
exists, inspect it instead of adding it again or force-pushing. If Git requests your
identity, configure your name and your chosen GitHub commit email; a GitHub-provided
no-reply email can keep your personal address private. Authenticate through your normal
GitHub sign-in flow when prompted.

The command sequence follows GitHub's [local repository publishing guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## Publish future updates

Make and verify your changes, then stage the specific files you intend to publish:

```powershell
git status --short
git add README.md
git diff --cached
git commit -m "Update project documentation"
git push
```

The example stages only `README.md`; substitute the files you actually changed.
For code changes, run the test and style checks before committing.

## Sharing the source ZIP

`dist/gpt2-from-scratch-portfolio.zip` contains an explicit allowlist of source, docs,
tests, configs, and small report assets. It omits the environment, data arrays, and
weights. Extract it before uploading files into a new repository; uploading only the
ZIP would prevent GitHub from rendering the README and browsing the source normally.
Preserve the `.github` directory so CI can run.

## After the first push

1. Open the repository's Actions tab and wait for the Tests workflow to finish.
2. Verify that the README renders its graphs, links, and architecture diagram.
3. Verify that tests pass on the configured Linux and Windows runners for the commit
   you are sharing; consult the linked Actions run rather than assuming success.
4. Add the description/topics and pin the repository to your profile.
5. If you create a release, attach weights only as release assets if desired. The
   source repository intentionally excludes `.pt` files.

The CI configuration follows the official [checkout](https://github.com/actions/checkout)
and [Python setup](https://github.com/actions/setup-python) actions. It runs tests and
style checks on synthetic data, without needing the corpus or a GPU.

## Suggested release notes

> Initial educational release: a GPT-style decoder with manual causal attention,
> character and byte BPE tokenizers, reproducible CPU experiments, baseline comparisons,
> seeded text generation, and checkpoint-recovery tests. See the results report for
> measured performance and limitations. GPU configuration is provided but unbenchmarked.
