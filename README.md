# LLM Practice

## Dependencies

If you're running on Hipergator, first run
```bash
module load cuda/12.9.1
```
You'll have to run this command *any* time you use this codebase (not just during the initial package installation).


Install the [`uv`](https://docs.astral.sh/uv/#installation) package manager if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use it to create a virtual environment:
```bash
uv venv
uv sync
```
This will create a local `.venv/` directory.
Run `source .venv/bin/activate` to activate the environment and `deactivate` to deactivate it.
If you're using an editor like VSCode, the path to your python interpreter is `./venv/bin/python`.

## Model Installs