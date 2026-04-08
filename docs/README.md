# Documentation

## Building locally

Install dependencies:

```bash
pip install -r docs/requirements-docs.txt
```

Build once:

```bash
cd docs
make html
open build/html/index.html
```

## Live preview

For auto-refresh on file save:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/build/html
# Opens http://127.0.0.1:8000
```
