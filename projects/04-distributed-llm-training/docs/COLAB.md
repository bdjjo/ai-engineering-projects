# One-GPU Colab pilot

Choose a GPU runtime in Google Colab. These commands use one GPU and the
124M-parameter model; they do not test sharding across GPUs. The private learning
notebook provides the same steps with explanations. No Hugging Face account/token
is needed for the public WikiText sample and GPT-2 tokenizer.

## Setup in a fresh runtime

Run in a Colab code cell:

```python
import subprocess, sys
from pathlib import Path
repo = Path('/content/ai-engineering-projects')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/bdjjo/ai-engineering-projects.git',
                    str(repo)], check=True)
project = repo / 'projects/04-distributed-llm-training'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.7.1',
                'torchvision==0.22.1', 'torchaudio==2.7.1',
                '--index-url', 'https://download.pytorch.org/whl/cu126'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(project / 'requirements-dev.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(project), '--no-deps'],
               check=True)
```

Restart the session after installation; do not reinstall on every restart. The
wheel combination follows the official PyTorch 2.7.1 instructions. This project
supports Python 3.11–3.12; if Colab changes its default, choose a compatible runtime.

## Train, stop, and resume

Run after restart. The Drive authorization is interactive; complete it yourself.
Choose a new run folder for each experiment. Keep its name to continue after a
disconnect. Data and checkpoints persist in Drive; Colab's local disk is temporary.

```python
import json, subprocess, sys
from pathlib import Path
import torch
from google.colab import drive
assert torch.cuda.is_available(), 'Select a GPU runtime first'
print(torch.__version__, torch.cuda.get_device_name(0))
drive.mount('/content/drive')
project = Path('/content/ai-engineering-projects/projects/04-distributed-llm-training')
work = Path('/content/drive/MyDrive/ai-engineering-projects/personal/p4-work/pilot-01')
work.mkdir(parents=True, exist_ok=True)

def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=project, check=True)

if not (work / 'data/manifest.json').exists():
    run('prepare_data.py', '--source', 'wikitext', '--out', work / 'data')
checkpoint = work / 'first/checkpoints/step-000010'
if not (checkpoint / 'COMPLETED.json').exists():
    run('train_distributed.py', '--config', 'configs/colab-125m.json',
        '--data', work / 'data', '--out', work / 'first', '--stop-after', '10')
if not (work / 'resumed/summary.json').exists():
    run('train_distributed.py', '--config', 'configs/colab-125m.json',
        '--data', work / 'data', '--out', work / 'resumed', '--resume', checkpoint)
print(json.loads((work / 'resumed/summary.json').read_text()))
run('scripts/export_report.py', '--run', work / 'resumed', '--out', work / 'report')
```

If a run was interrupted before completion, keep its existing folder. Resume from
the latest checkpoint with a completion marker into a **new** output folder. An
unfinished data preparation lacks a manifest; use a new data folder. Never change
the config between the first segment and resume. Drive synchronization and
checkpoint staging can be slow; wait for the completion marker before disconnecting.

## Interpret the output

Read `metadata.json` for GPU model, precision, actual parameter count, data/source
hashes, and the MFU denominator. Read `summary.json` for validation loss, measured
global tokens/s, memory, skipped optimizer steps, and staging/join time.

Twenty attempts are a systems pilot, not a trained language model. A T4 should
select FP16; requesting BF16 explicitly fails early. If memory is exhausted, start
a new experiment with a copied config using context 128 or fewer layers. That is
a different model/experiment and cannot resume the previous checkpoint.

For a full recovery comparison, also run an uninterrupted 20-attempt reference
with the same settings and compare weights/optimizer state. For performance,
increase the schedule in a fresh config/run and collect repeated unprofiled runs.
Use `--profile` in a separate run to capture a trace. Keep dropout, batch, context,
and effective global batch fixed when comparing strategies.

Share the small exported `results.json` when ready to add actual GPU results to
the repository. Until then, GPU throughput, GPU MFU, and multi-GPU scaling remain
unmeasured. A single-GPU FSDP invocation is not evidence of multi-GPU FULL_SHARD.
