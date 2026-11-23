# Run the larger experiment on Colab

Select a GPU runtime. Use a fresh notebook and keep a stable run name. This setup
creates a separate Python 3.12 environment so changes to Colab's notebook Python
do not silently change the experiment's dependencies. The private tutorial notebook
contains the same stages with explanations and defaults all expensive actions off.

## 1. Install into a separate environment

```python
import subprocess
import sys
from pathlib import Path

repo = Path('/content/p5-ai-engineering-projects')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/bdjjo/ai-engineering-projects.git',
                    str(repo)], check=True)
print('Source revision:', subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
project = repo / 'projects/05-constitutional-ai'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
env = Path('/content/p5-python312')
if not env.exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.12', str(env)], check=True)
python = str(env / 'bin/python')

def install(*args):
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
                    *args], check=True)

install('torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu126')
install('-r', str(project / 'requirements-dev.txt'))
install('-e', str(project), '--no-deps')
subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available(); '
                'print(torch.__version__, torch.cuda.get_device_name(0))'], check=True)
subprocess.run([python, '-m', 'pytest', '-q'], cwd=project, check=True)
```

The wheel index follows [PyTorch's 2.7.1 installation instructions](https://pytorch.org/get-started/previous-versions/).
The environment creation uses [uv's Python environment support](https://docs.astral.sh/uv/pip/environments/).
No model is imported into the notebook kernel, so this workflow runs all stages as
subprocesses with the same Python executable. Do not mix in the kernel's Python.

## 2. Mount Drive and collect feedback

```python
from google.colab import drive
drive.mount('/content/drive')

RUN_NAME = 'p5-pilot-01'  # Choose a fresh name for a changed experiment.
work = Path('/content/drive/MyDrive/ai-engineering-projects/personal/p5-work') / RUN_NAME

def run_stage(stage):
    subprocess.run([python, 'run_cai.py', stage, '--config', 'configs/colab-pilot.json',
                    '--device', 'cuda', '--out', str(work)], cwd=project, check=True)

run_stage('collect')
```

The default uses the 0.5B actor/critic, 1.5B judge, and all 54 non-test prompts.
It downloads model weights on the Colab machine. It uses no paid API service.
Complete Drive authorization yourself; the code does not handle account credentials.

If disconnected during collection, restore the same checkout and run configuration,
mount Drive, and use the same run name. Saved prompt records are reused. Model
revisions and code hashes in `manifest.json` must stay the same. Do not pull new
source into an interrupted experiment.

## 3. Inspect the labels before fitting

```python
import json
audit = json.loads((work / 'feedback/audit.json').read_text())
print(json.dumps(audit, indent=2))
pairs = json.loads((work / 'feedback/pairs.json').read_text())
print('Accepted:', len(pairs))
for pair in pairs[:2]:
    print('\nRequest:', pair['prompt'])
    print('Preferred:', pair['chosen'])
    print('Other:', pair['rejected'])
```

Look for a correct response being rejected, an unhelpful refusal being preferred,
and failures concentrated in a particular category. If either split has no accepted
pairs, stop and investigate the raw judgments. Do not relax the parser to force a
successful training run. A different judge or longer generation budget is a new
experiment and needs a new configuration and run directory.

## 4. Fit and compare

```python
for stage in ['reward', 'sft', 'rl', 'evaluate']:
    run_stage(stage)

for variant in ['sft', 'rl']:
    summary = json.loads((work / f'evaluation-{variant}/summary.json').read_text())
    print(variant, json.dumps(summary['groups']['all'], indent=2))
```

Check valid-judgment counts before interpreting averages. A learned-reward gain
with lower helpfulness is a failure worth reporting. The test set is small and the
default judge shares the actor's model family; independently review the outputs.
Do not tune on these test responses and then call the same prompts held out.

An interrupted training stage cannot recover its optimizer state. Preserve the
partial directory and start a fresh experiment. Completed stages are reused on a
rerun. The GPU preset has not been validated on this local CPU-only machine; if
memory is exhausted, use a smaller batch/model in a separate experiment.

## 5. Export the evidence

```python
from datetime import datetime
import shutil
from google.colab import files

report = work / ('report-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
subprocess.run([python, 'scripts/export_report.py', '--run', str(work), '--out', str(report)],
               cwd=project, check=True)
archive = shutil.make_archive(str(report), 'zip', report)
files.download(archive)
```

The archive contains configuration, revision hashes, raw feedback, training
summaries, and evaluation outputs. It excludes model weights. Review it before
publishing; it includes generated response text, including unsuccessful answers.
