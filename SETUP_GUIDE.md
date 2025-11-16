# PyIsolate Setup Guide

A single reference for getting the repository from a fresh clone to every built-in example running.

## 1. Prerequisites

- Python 3.9 or newer
- [`uv`](https://github.com/astral-sh/uv) available on your `PATH`
- Git, build-essential/clang (platform dependent)

Optional but recommended:
- CUDA drivers that match the torch build you plan to share between host and isolated nodes

Verify `uv` before proceeding:
```bash
uv --version
```

## 2. Bootstrap the repository

```bash
cd /home/johnj/pyisolate
uv venv
source .venv/bin/activate         # Windows: .venv\\Scripts\\activate
uv pip install -e ".[dev,test]"
pre-commit install
```

Key expectations:
- The virtualenv lives inside the repo so tooling (ruff/pytest) always matches pyisolate's requirements.
- Installing the `[dev,test]` extras ensures smoke tests and benches have their dependencies up front.

## 3. Run the standard multi-extension example

```bash
cd example
python main.py
cd ..
```
Expected output (abbreviated):
```
Extension1      | ✓ PASSED | Data processing with pandas/numpy 1.x
Extension2      | ✓ PASSED | Array processing with numpy 2.x
Extension3      | ✓ PASSED | HTML parsing with BeautifulSoup/scipy
```
This confirms conflicting dependencies can coexist in isolated venvs.

## 4. Run the Comfy Hello World demo

```bash
cd comfy_hello_world
python main.py
```
What to expect:
1. pyisolate reads `custom_nodes/simple_text_node/pyisolate.yaml`
2. Dependencies install via `uv` inside `comfy_hello_world/node-venvs/simple_text_node`
3. The isolated node registers via RPC and exposes `SimpleTextNode`
4. Shared singleton call `fetch_default_model_name` succeeds

You should see log lines prefixed with `📚 [PyIsolate]` describing venv creation and RPC operations.

## 5. Run the smoke tests (after they land)

Once the automated smoke tests are added (see roadmap), run:
```bash
pytest tests/smoke
```
This target will execute both the multi-extension sample and the Comfy hello world under pytest control. Until that folder exists, manual runs from sections 3-4 are the source of truth.

## 6. Full test suite (optional)

```bash
pytest
```
This triggers unit tests, including singleton behavior, RPC mechanics, and (soon) smoke tests.

## 7. Keeping repos in sync

When working across pyisolate, mysolate, and ComfyUI simultaneously:
```bash
cd /home/johnj/pyisolate && git pull --rebase
cd /home/johnj/mysolate && git pull --rebase --autostash
cd /home/johnj/ComfyUI && git pull --rebase
```
Resolve any conflicts immediately so the instructions above stay accurate.

## Troubleshooting checklist

- `uv` missing: install via `pip install uv` or the standalone installer.
- `torch` mismatch: ensure your system-wide torch matches the version isolated nodes need when `share_torch=True`.
- Stale venv: delete `comfy_hello_world/node-venvs` if dependencies change and rerun section 4.
- Logs not showing 📚 prefix: confirm you're on the latest `main` branch and that logging level is INFO.

This document replaces the older HELLO_WORLD / GETTING_STARTED examples with a single flow. If you get stuck anywhere, capture the exact log output so the fail-loud policy can point us at the culprit quickly.
