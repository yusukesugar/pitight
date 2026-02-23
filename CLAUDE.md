# CLAUDE.md

## Project Overview

pitight — Pipeline Integrity Checker for ML workflows. Orchestrator-agnostic building blocks for data pipeline validation, temporal leak detection, and artifact management.

## Commands

```bash
python -m pytest -q          # all tests
python -m pytest tests/test_partitioned.py -v  # specific file
```

## Code Style

### Examples in docstrings and tests

pitight is a **general-purpose library** — examples must use generic, domain-neutral scenarios.

**Use**: Product analytics, e-commerce, SaaS metrics
- `features/user_engagement`, `features/session_summary`, `features/purchase_history`
- `user_id`, `event_type`, `sessions_7d`, `revenue`, `churn_score`

**Do NOT use**: Domain-specific examples (e.g. boat racing, sports betting, medical, finance-specific jargon)

This ensures documentation and tests are approachable for any user of the library.
