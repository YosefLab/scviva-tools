# Installation

## Quick install

scVIVA-Tools can be installed via `pip`.
We recommend installing into a fresh virtual environment to avoid conflicts with other packages
and compatibility issues.

For the basic CPU version run:

```bash
pip install -U scviva-tools
```

To install scVIVA-Tools with Nvidia GPU CUDA support, for Linux Systems
(such as Ubuntu or RedHat) use:

```bash
pip install -U "scviva-tools[cuda]"
```

And for Apple Silicon metal (MPS) support:

```bash
pip install -U "scviva-tools[metal]"
```

Note that for some shells (e.g. `zsh`) you will have to use quotes in order to install with
optional dependencies, as shown above.

Don't know how to get started with virtual environments or `conda`/`pip`? Check out the
[prerequisites](#prerequisites) section.

## Prerequisites

### Virtual environment

A virtual environment can be created with either `conda` or `venv`. We recommend using a fresh
`conda` environment. We currently support Python 3.12 - 3.14.

```bash
conda create -n scviva-env python=3.13  # any python 3.12 to 3.14
conda activate scviva-env
pip install -U scviva-tools
```

For `venv`, we recommend using [uv](https://github.com/astral-sh/uv), which is a high-performance
Python package manager and installer written in Rust.

```bash
pip install -U uv
uv venv .scviva-env --python 3.13  # any python 3.12 to 3.14
source .scviva-env/bin/activate  # for macOS and Linux
.scviva-env\Scripts\activate  # for Windows
uv pip install -U scviva-tools
```

scVIVA-Tools is built on top of [scvi-tools](https://docs.scvi-tools.org/), which is installed
automatically as a dependency. There is no need to install scvi-tools separately; if you already
have it in your environment, make sure it satisfies the minimum version required by scVIVA-Tools.

### GPU support with PyTorch

scVIVA-Tools depends on PyTorch for accelerated computing. If you don't plan on using an
accelerated device, we recommend installing scVIVA-Tools directly and letting these dependencies
be installed automatically by your package manager of choice.

If you plan on taking advantage of an accelerated device (e.g., Nvidia GPU or Apple Silicon),
install with the matching GPU support extra (`cuda` or `metal`, see above).

However, there might be cases where the GPU hardware does not support the latest version of
PyTorch. In this case we recommend installing PyTorch _before_ installing scVIVA-Tools.
Please follow the respective installation instructions for
[PyTorch](https://pytorch.org/get-started/locally/) compatible with your system and device type.

## Optional dependencies

scVIVA-Tools is installed in its lightest form by default.
It has optional dependencies which expand its capabilities:

| Extra | Command | Installs |
|-------|---------|---------|
| Spatial backends | `pip install "scviva-tools[spatial]"` | squidpy, spatialdata, torch-geometric, geomloss, ... |
| VISION | `pip install "scviva-tools[vision]"` | biopython |
| Model hub | `pip install "scviva-tools[hub]"` | huggingface-hub, boto3, dvc |
| File sharing | `pip install "scviva-tools[file-sharing]"` | gdown, pooch, ... |
| CUDA | `pip install "scviva-tools[cuda]"` | GPU support on Linux |
| Apple Silicon | `pip install "scviva-tools[metal]"` | MPS support on macOS |
| TPU | `pip install "scviva-tools[tpu]"` | torch-xla |
| GPU acceleration | `pip install "scviva-tools[rapids]"` | cuML, cuPy, cuGraph, rapids-singlecell |
| GPU acceleration (CUDA 12) | `pip install "scviva-tools[rapids-cuda12]"` | RAPIDS built for CUDA 12 |
| GPU acceleration (CUDA 13) | `pip install "scviva-tools[rapids-cuda13]"` | RAPIDS built for CUDA 13 |
| Tutorials | `pip install "scviva-tools[tutorials]"` | jupyter, matplotlib, seaborn, cell2location, ... |
| All | `pip install "scviva-tools[all]"` | spatial, rapids and tutorials extras |

## Google Colab

Every tutorial can be run on [Google Colab](https://colab.research.google.com) using the
rocket icon at the top of its page. The first code cell of each tutorial installs
scVIVA-Tools and its tutorial dependencies with the
[scvi-colab](https://github.com/scverse/scvi-colab) helper. It only runs when it detects a
Colab session and has no effect in other environments.

## Development install

```bash
git clone https://github.com/YosefLab/scviva-tools.git
cd scviva-tools
pip install -e ".[dev,test]"
pre-commit install
```

## Docker

A minimal `Dockerfile` is provided at the root of the repository. It installs scVIVA-Tools with
the `spatial` extra on top of a `python:3.13-slim` image:

```bash
docker build -t scviva-tools .
docker run --rm scviva-tools
```

## Verifying the installation

```python
import scviva

print(scviva.__version__)
```

## GPU support

RAPIDS-based acceleration (neighbor computation and latent representation) requires a CUDA-capable
GPU and the `rapids` extra. The base package runs on CPU without any GPU dependencies.
