# tachys documentation

## Setup

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

## Build

```bash
python -m sphinx docs docs/_build/html
```

The built site lands in `docs/_build/html/`.

## Visualize locally

**Option 1 — open directly:**

```bash
open docs/_build/html/index.html
```

**Option 2 — serve over HTTP (recommended, avoids browser CORS quirks):**

```bash
python -m http.server 8080 --directory docs/_build/html
```

Then open <http://localhost:8080>.

**Option 3 — auto-rebuild on file save:**

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Watches `docs/` for changes and refreshes the browser automatically.

## Design preview

A standalone HTML mockup of the documentation design is included at
`docs/_static/design-preview.html`. Open it directly in any browser — it has
no dependencies:

```bash
open docs/_static/design-preview.html
```

After a Sphinx build it is also served alongside the docs at
`http://localhost:8080/_static/design-preview.html`.
