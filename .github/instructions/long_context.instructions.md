---
applyTo: "**"
actions:
  - name: Long-Context Handling
    description: |
      • Chunk docs at semantic sentences (~200 tokens); store embeddings.  
      • For 1 M-token windows, retrieve only top-k (≤20) chunks.  
      • Summaries cached with SHA-256 of source text.
meta:
  createdBy: "Veracity AI – Vertical-Agent Playbook v0.9"
  createdAt: "2025-05-18"
---
Use XML for massive doc sets; avoid JSON unless strictly required.