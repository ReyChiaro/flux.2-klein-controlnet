# FLUX.2 Klein ControlNet

This implementation conditions on a masked source image. RGB VAE inputs use
`[-1,1]`; masks use `>= 0.5` for visible pixels and three binary channels.
Condition tokens retain 128 latent features plus three mask features.

Periodic checkpoints are model directories containing configuration and
safetensors weights. The final directory retains the existing `lastest` name.
Standalone inference requires `--controlnet <checkpoint-directory>`.

Residual routing uses `block_index * head_count // block_count`. When heads do
not outnumber base blocks, every head is consumed. This changes routing semantics
from the former ceiling-interval mapping. Parameter shapes remain unchanged, but
checkpoints trained with the old mapping cannot be assumed to reproduce their
original behavior with this routing or the corrected preprocessing.

## CPU regression tests

The suite targets Windows and Linux using reduced native transformer classes,
synthetic text/images, and a VAE fixture; temporary synthetic checkpoints are
removed. No pretrained weights are needed. The 2 GiB guard checks Windows peak
working set/private commit, or Linux peak RSS (`ru_maxrss`, converted from KiB).
These measurements are not equivalent: Linux RSS does not measure private
commit ([Linux units](https://man7.org/linux/man-pages/man2/getrusage.2.html)).
The guard checks process peaks at module setup/teardown; it is not an
OS allocation limit or a continuous monitor. Other platforms fail explicitly.

Windows, in an existing Python/PyTorch/Diffusers environment:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -B -u -m unittest discover -s tests -p test_native_contracts.py -v
```

Linux, in an environment with the test dependencies installed:

```sh
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -u -m unittest discover -s tests -p test_native_contracts.py -v
```

The manually dispatched `.github/workflows/cpu-regressions.yml` prepares an
Ubuntu/Python 3.12 environment with pinned CPU-only PyTorch and test libraries.
It does not install the project's training dependencies or download model
weights. GitHub requires the workflow on the fork's default branch before it
can be dispatched for a selected contribution branch. There are no automatic
push or pull-request triggers. A local workflow file is not a successful Linux
test result.

The tests cover conditioning contracts, positional roles, sigma indexing,
residual routing, staged gradients, checkpointed backward, and serialization.
They do not establish pretrained 9B feasibility, convergence, or pose control.
Nonbucket training, batches larger than one, multi-reference alignment, CFG
residual reuse, shortened initialization, and residual scaling remain deferred.
