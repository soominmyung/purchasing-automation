# Task: Purchasing Automation → Llama-3 Fine-tuned Engine

- [x] Phase 1: Data Pipeline & Teacher Dataset
    - [x] Synthetic data generation (30 scenarios with edge cases)
    - [x] CoT-based Analysis Agent refinement
    - [x] Anti-hallucination Evaluation Agent
    - [x] Teacher dataset generation (teacher_dataset_20260225.jsonl)

- [/] Phase 1.5: Code Quality & Git Setup
    - [x] Create `legacy/gpt4o` branch (preserve current main)
    - [x] Convert all comments/docstrings to English
    - [x] Enhance prompts (Few-shot, persona, error handling)
    - [ ] Regenerate teacher dataset with improved prompts
    - [ ] Phase 1 commits (4 commits) → PR #1 merge to main

- [ ] Phase 2: Fine-tuning on Vertex AI
    - [ ] SFT training script (`scripts/train_sft.py`) + W&B tracking
    - [ ] Run SFT on Vertex AI (Llama-3-8B + QLoRA)
    - [ ] Build DPO preference pairs (Rejected: 4.5 scores, Chosen: 8.5 scores)
    - [ ] Run DPO on Vertex AI
    - [ ] Before/After comparison (Base Llama vs SFT vs SFT+DPO vs GPT-4o)
    - [ ] Phase 2 commits (3 commits) → PR #2 merge to main

- [ ] Phase 3: Serving & Self-Correction
    - [ ] Add Eval-driven retry loop (LangGraph: Eval < 7 → re-analyze)
    - [ ] Extend Validator to reports/PR documents
    - [ ] vLLM serving + Backend Bridge integration
    - [ ] Phase 3 commits (3 commits) → PR #3 merge to main

- [ ] Phase 4: Documentation & Polish
    - [ ] README: architecture diagram + quantitative results table
    - [ ] README: Prompt Design Principles section
    - [ ] HuggingFace Hub: upload fine-tuned LoRA adapter
