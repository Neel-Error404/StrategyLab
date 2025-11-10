---
applyTo: "**"
actions:
  - name: Validation & Iterative QA
    description: |
      **Always** run the 5-step QA loop:

      1️⃣ **Inventory** public APIs ↔ tests.  
      2️⃣ **Consistency Checks** (names, behaviour drift, coverage gaps).  
      3️⃣ **Plan Before Code** – propose fixes/tests; pause for approval if core
          logic or external deps change.  
      4️⃣ **Implement** agreed scope: pytest + hypothesis; mock I/O; keep >90 %
          critical coverage; `pytest -q` green.  
      5️⃣ **Report** under *Highlights*: pass/fail, deltas, next steps.
meta:
  createdBy: "Veracity AI – Vertical-Agent Playbook v0.9"
  createdAt: "2025-05-18"
