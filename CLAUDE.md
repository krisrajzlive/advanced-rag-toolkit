# CLAUDE.md

See [AGENTS.md](AGENTS.md) for the project conventions Claude should follow
here (config access, native-LlamaIndex-first policy, loader contract,
optional-dependency handling, secrets).

Claude-specific notes:

- This project was scaffolded end-to-end by Claude Code in one session:
  directory layout, all `src/` modules, the downloaded Anthropic PDF sample,
  the generated diagram image, and the docs. There is no separate "legacy"
  layer to reconcile — if something looks inconsistent, prefer fixing it
  over working around it.
- When adding a demo command to `main.py`, follow the existing
  `cmd_<name>(args)` + `COMMANDS` dict pattern; don't introduce a different
  CLI framework (argparse subparsers, click, etc.) for a single command.
- Free-tier API responses (Hugging Face Inference API, Ollama Cloud) can be
  slow or occasionally return a model-unavailable error — treat that as an
  environment/provider issue to report, not a bug to silently retry-loop
  around.
- If asked to add a new evaluation metric, extend
  `src/evaluation/langsmith_eval.py`'s evaluator list rather than building a
  parallel evaluation path — LangSmith's `evaluate()` is meant to run all
  evaluators over the same dataset/target in one pass.
