---
applyTo: "**"
actions:
  - name: Coding Standards & Preferences
    description: |
      🐍 **Python ≥ 3.10** – type-hints, dataclasses, pathlib, rich logging.  
      • Max line 100; f-strings; no bare `assert`.  
      • Use `pydantic` models for data contracts.

      🔄 **Testing** – pytest; filenames `test_<module>.py`; ≥90 % path coverage.  
         hypothesis for property tests; `pytest-asyncio` for async flows.

      📈 **Investment-Research Domain**  
      • Guard against look-ahead bias; timestamp feeds UTC ± IST; numeric
        outputs rounded to 2 dp.  
      • Risk metrics: VaR (95 %), CVaR, TGARCH vol.  
      • Each insight cites its source URL/id.

      🛠 **Architecture**  
      • Agents exchange *typed* dataclasses via ADK tool calls.  
      • Vector store default Chroma; pluggable via factory.  
      • Secrets in dotenv / GCP Secret Manager.  
      • Provide `invoke` tasks: `fmt`, `lint`, `test`, `run`.
meta:
  createdBy: "Veracity AI – Vertical-Agent Playbook v0.9"
  createdAt: "2025-05-18"
    