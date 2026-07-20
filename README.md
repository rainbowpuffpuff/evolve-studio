# Evolve Studio

Standalone product evolution studio — extracted from [Dev Studio](../dev-studio).

Ship monetizable HTML products with multi-model workers (Cerebras free tier + OpenRouter free), gallery, and generation runs.

## Quick start

```bash
cd evolve-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# keys in .env (CEREBRAS_API_KEY, OPENROUTER_API_KEY optional)
export STUDIO_DATA_DIR="$(pwd)/data"
export EVOLVE_STUDIO_PORT=8771
python3 server.py
```

Open **http://127.0.0.1:8771/#/evolve**

## Features

- **Start** — goals, AI money ideas, seed from prior products
- **Runs & HTML** — live generation theater, continue/clone, product HTML
- **Gallery** — last-gen products with +5 gens and deploy placeholders
- **Free models quota** — Cerebras + Devin estimates + OpenRouter free
- Multi-key Cerebras high-throughput toggle
- Diverse worker pool (Cerebras HT/LT + OpenRouter free)

## Config

| Env | Default | Meaning |
|-----|---------|---------|
| `EVOLVE_STUDIO_PORT` | `8771` | HTTP port |
| `EVOLVE_STUDIO_HOST` | `0.0.0.0` | Bind host |
| `STUDIO_DATA_DIR` | `./data` | Evolutions, jobs, usage |
| `CEREBRAS_API_KEY` | — | One or more keys (comma-separated) |
| `OPENROUTER_API_KEY` | — | Free-model workers |
| `OPENROUTER_FREE_ONLY` | `1` | Force free OpenRouter models |

## Layout

```
evolve-studio/
  server.py          # FastAPI app (Evolve APIs only)
  lib/               # evolution engine, LLM, planner, …
  static/index.html  # Evolve-focused UI
  data/evolutions/   # run artifacts
  run.sh
```

## License

Extracted for independent development. See parent project as needed.
