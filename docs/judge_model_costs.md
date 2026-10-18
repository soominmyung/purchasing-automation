# Judge model costs

**As of 2026-10-18.** Prices change often; check the sources below before relying on them.

Sources:
- OpenAI — https://developers.openai.com/api/docs/pricing (no last-updated date shown)
- Anthropic — https://platform.claude.com/docs/en/about-claude/pricing
- Google — https://ai.google.dev/gemini-api/docs/pricing (last updated 2026-10-16)

All prices are Standard tier, USD per 1M tokens, for prompts well under the long-context
thresholds. Reasoning ("thinking") tokens are billed as output.

## Cost of one judge call

Measured on 2026-10-18 with one call per model: the production evaluation prompt for
holdout scenario H3 (six judge checks, GPT-4o's analysis as the candidate). One call each,
so latency and output length are indicative only.

| Provider | Model | Input / output price | Tokens in / out (of which thinking) | Latency | Per call | Per 1,000 analyses |
|---|---|---|---|---:|---:|---:|
| OpenAI | `gpt-5.5` | $5.00 / $30.00 | 1,480 / 613 (288) | 8.7 s | $0.0258 | $25.79 |
| OpenAI | `gpt-5.4-mini` | $0.75 / $4.50 | 1,480 / 339 (0) | 2.1 s | $0.0026 | $2.64 |
| OpenAI | `gpt-4.1-mini` | $0.40 / $1.60 | 1,481 / 312 (0) | 2.8 s | $0.0011 | $1.09 |
| OpenAI | `gpt-5.6-luna` | $0.20 / $1.20 | 1,480 / 419 (184) | 4.7 s | $0.0008 | $0.80 |
| OpenAI | `gpt-5.4-nano` | $0.20 / $1.25 | 1,480 / 373 (0) | 3.1 s | $0.0008 | $0.76 |
| OpenAI | `gpt-6-luna` | $0.10 / $0.50 | 1,480 / 677 (374) | 8.9 s | $0.0005 | $0.49 |
| Anthropic | `claude-sonnet-5-5` | $2.00 / $10.00 | 2,324 / 478 (0) | 6.1 s | $0.0094 | $9.43 |
| Anthropic | `claude-haiku-5-5` | $0.10 / $0.50 | 2,324 / 1,261 (0) | 5.3 s | $0.0009 | $0.86 |
| Google | `gemini-3.8-flash` | $0.75 / $3.75 | 1,707 / 1,131 (816) | 7.4 s | $0.0055 | $5.52 |
| Google | `gemini-3.6-flash` | $0.75 / $3.75 | 1,707 / 1,107 (809) | 6.2 s | $0.0054 | $5.43 |
| Google | `gemini-3.7-flash` | **not on the price list** | 1,707 / 1,060 (768) | 8.6 s | — | — |
| Google | `gemini-3.5-flash-lite` | $0.30 / $2.50 | 1,707 / 272 (0) | 2.4 s | $0.0012 | $1.19 |
| Google | `gemini-3.1-flash-lite` | $0.25 / $1.50 | 1,707 / 274 (0) | 2.8 s | $0.0008 | $0.84 |

For scale: one full pipeline run for one supplier group costs about $0.061 on GPT-4o
(README, "Cost analysis"). A `gpt-5.5` judge would add about 42% to that; the cheapest
judges add about 1%.

## Things to know

- **`gemini-3.7-flash` has no published price.** It is served by the API with a dated
  version (`3.7-flash-08-2026`) but is absent from Google's price list, so its cost cannot be
  planned for. It and `gemini-3.6-flash` also returned `503 UNAVAILABLE` (high demand) on the
  first attempt and only succeeded on retry.
- **Gemini Flash prices double on 2027-01-01** (`gemini-3.6-flash` and `gemini-3.8-flash`
  to $1.50 / $7.50). `gemini-4-argon` doubles on 2026-12-01.
- **Thinking tokens dominate some models' cost.** The Gemini Flash models, `gpt-5.5` and the
  `luna` models spend 180–820 output tokens reasoning before answering; the lite, mini and
  nano models and the Claude models do not, which is why they are both cheaper and faster.
- **`claude-haiku-5-5` is priced for prompts up to 100,000 tokens.** The judge prompt is
  about 2,300 tokens, far below that.
- **Batch APIs cost about half** at all three providers, which matters for offline
  evaluation runs but not for the production judge, which answers one analysis at a time.
- **`gpt-5.6-sol`** is on promotional pricing until at least 2026-11-21.

## Price alone does not choose a judge

A cheap judge is only worth it if it reaches the right verdicts. Candidates are tested on
outputs with known verdicts (seeded faults and hand-scored analyses) before one is chosen;
the design, metrics and results are in `docs/judge_validation.md`.

## Reference: other models in this project

| Model | Role | Input / output price |
|---|---|---|
| `gpt-4o` | Analysis, report and purchase-request agents; email rewrite after a failed check | $2.50 / $10.00 |
| `gpt-4o-mini` | Email draft agent; email validator | $0.15 / $0.60 |
