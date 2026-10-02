# AGENTS.md

## Commands

uv workspace: root package + `examples/`. Run from the repo root.

- Setup: `uv sync --all-packages`
- Tests: `uv run pytest`; single test: `uv run pytest tests/test_translator.py::test_undeclared_output_raises`
- Lint / format: `uv run ruff check .` / `uv run ruff format .`
- Typecheck: `uv run pyright` (root package; `examples/` has its own pyright config with the venv in `..`)
- UI: `cd examples && uv run dg dev` → http://localhost:3000

## Gotchas

- Bare `uv sync` (uv 0.9.x) only syncs the root project in a workspace — it prunes `examples` and the `dg` deps from the venv and breaks `dg dev`. Always use `uv sync --all-packages`.
- `xclim`/`xarray` are an optional extra (`[xclim]`); tests and `examples` need it, and the dev group pulls it in via the self-referencing `dagster-object-factory[xclim]` entry. Core `__init__.py` must not import `integrations.xclim` (xclim is not always installed).
- `examples/` is the dagster-dg project; the repo root is not a dg workspace, so `dg dev` must run from `examples/`. `examples/.tmp_dagster_home_*` are `dg dev` runtime artifacts (gitignored).
- Example defs fetch the xarray `air_temperature` tutorial dataset (pooch) on first load; cached in `~/.cache/xarray_tutorial_data`, network needed once.
- No CI, pre-commit, or task runner.

## Architecture

- `src/dagster_object_factory/` — typed library (src layout, `py.typed`); public API in `__init__.py`.
- Flow: `Layer` (shared `key_prefix`, predeclared `output_names`, `layer_deps`) → `DagsterObjectTranslator` (object → `DagsterObjectTranslation`: identity, outs, deps) → `ObjectFactoryComponent` (builds one multi-asset per object; subclasses implement `execute`).
- Resolving an output not in `Layer.output_names` raises `ValueError` at translation time (fail-fast in `DependencySpec.__post_init__`).
- Multi-output assets: `execute` must return a sequence aligned with `output_names` order (strict zip); single-output returns a scalar.
- `integrations/xclim/` — maps xclim `Indicator` → asset (identifier = asset name, `cf_attrs` = outputs, variable params → deps). `XclimResamplingIndicatorFactory` takes `freq` from `injected_parameters["freq"]` or a `ResamplingPartitionsDefinition` partition key, else `DagsterInvalidDefinitionError`.
- `examples/src/examples/` — dg project; `definitions.py` loads `defs/` via `load_from_defs_folder`.

## Conventions

- Commit messages: short, lowercase, imperative (e.g. "add docstrings").
