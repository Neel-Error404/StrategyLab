---
applyTo: "**"
actions:
  - name: Prompt Guidelines
    description: |
      • Prepend **Planning** (“step-by-step”) & **Tool-calling** reminders.  
      • Chain-of-Thought in analysis scratchpad; never leak to user.  
      • Retrieval sequence: Query ➜ Context (≤4 docs) ➜ Synthesis prompt.  
      • Use role tags (`<engineer>`, `<analyst>`) for multi-turn clarity.

  - name: Eval Loop
    description: |
      • Attach RAGAS or TruLens eval in CI: relevance, faithfulness, toxicity.  
      • Fail PR if hallucination > 2 % or relevance < 80 %.
meta:
  createdBy: "Veracity AI – Vertical-Agent Playbook v0.9"
  createdAt: "2025-05-18"
---