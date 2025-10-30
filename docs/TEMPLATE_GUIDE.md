# Template Guide

Summary of built-in configuration templates for the equities backtester.

---

## Templates

| Template                | Profile       | Max Position | Notes                                      |
|-------------------------|---------------|--------------|--------------------------------------------|
| `minimal`               | Learning      | 5%           | Single-threaded, verbose logs.             |
| `conservative`          | Low risk      | 15%          | Balanced exposure, risk controls enabled.  |
| `aggressive`            | High risk     | 20%          | Multi-threaded, faster cadence.            |
| `portfolio_diversified` | Multi-ticker  | 12% average  | Diversified allocation across basket.      |

Apply via `--template <name>` or edit the YAML under `config/templates/`.

---

## Template Anatomy

```yaml
strategy:
  name: "mse"
  risk_profile: "conservative"
risk:
  max_position_size: 0.15
  max_daily_loss: 0.02
execution:
  parallel_processing: false
```

Override fields in a custom file and pass it through `--config` when running the runner.

---

## Custom Template Workflow

```bash
cp config/templates/minimal.yaml config/templates/my_custom.yaml
# edit my_custom.yaml as needed
python src/runners/unified_runner.py --mode backtest --config config/templates/my_custom.yaml --date-ranges 2024-02-01_to_2024-02-14
```

---

## Environment Variables

Use `${VAR}` syntax to inject secrets:

```yaml
broker:
  default_provider: "upstox"
  upstox_client_id: "${UPSTOX_CLIENT_ID}"
```

`config/config_loader.py` handles substitution at load time.

---

## Usage Tips

- Start with `minimal` for sandbox experiments.
- Switch to `conservative` or `portfolio_diversified` for production-style validations.
- Combine with parity/precision validation before tagging releases.

Refer to `docs/STRATEGY_GUIDE.md` for strategy implementation details.
