# LLM provider audit — Phase 7

This records what the LLM-evaluation gate needs on the **provider** side, what
this environment actually has, and what a user must do to open the gate. It is a
point-in-time finding, not a standing claim: re-check the sources before relying
on them.

The provider side and the data side are separate blockers. This document covers
the provider; [`LLM_POINT_IN_TIME_AUDIT.md`](LLM_POINT_IN_TIME_AUDIT.md) covers
the data.

## What the environment has

Checked 2026-10-28 by reading environment-variable **names** only. No value was
read, printed or serialised.

- **No LLM provider credential is present.** None of the variables the client
  declares (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, and the rest) is set
  to a usable value. The harness's own `OH_*` variables are the *agent's*
  credentials, not the project's, and are not used for project inference.
- **No local model runtime is running.** Nothing answers on the usual local
  ports (`11434` Ollama, `1234` LM Studio, `8000` vLLM). No `ollama` binary is
  installed, and there is no GPU, so a local model is technically possible but
  slow.
- **The network is reachable.** Unauthenticated probes of the provider endpoints
  return `401`/`403` (reachable, no credential) rather than a connection error,
  so a key would be usable here.

The machine-readable form of this is the `provider_readiness` block in
`docs/baseline/llm_evaluation.json`, produced by
`tradingagents/llm_evaluation/provider_audit.py`. That module derives readiness
from the client registries and env-var names, contacts nothing, and never reads a
credential value.

## Providers the project supports

The client factory builds these. `transport` is the wire protocol; the graph's
agents ask for structured output, so a provider must offer tool calling.

| Provider | Transport | Credential | Env var |
| --- | --- | --- | --- |
| `openai` | OpenAI Responses | API key | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | API key | `ANTHROPIC_API_KEY` |
| `google` | Gemini | API key | `GOOGLE_API_KEY` |
| `azure` | Azure OpenAI | API key | `AZURE_OPENAI_API_KEY` |
| `bedrock` | AWS Bedrock | AWS chain | — |
| `xai` | OpenAI-compatible | API key | `XAI_API_KEY` |
| `deepseek` | OpenAI-compatible | API key | `DEEPSEEK_API_KEY` |
| `qwen` / `qwen-cn` | OpenAI-compatible | API key | `DASHSCOPE_API_KEY` / `DASHSCOPE_CN_API_KEY` |
| `glm` / `glm-cn` | OpenAI-compatible | API key | `ZHIPU_API_KEY` / `ZHIPU_CN_API_KEY` |
| `minimax` / `minimax-cn` | OpenAI-compatible | API key | `MINIMAX_API_KEY` / `MINIMAX_CN_API_KEY` |
| `openrouter` | OpenAI-compatible | API key | `OPENROUTER_API_KEY` |
| `mistral` | OpenAI-compatible | API key | `MISTRAL_API_KEY` |
| `kimi` | OpenAI-compatible | API key | `MOONSHOT_API_KEY` |
| `groq` | OpenAI-compatible | API key | `GROQ_API_KEY` |
| `nvidia` | OpenAI-compatible | API key | `NVIDIA_API_KEY` |
| `chatgpt` | ChatGPT sign-in | Sign-in | — |
| `ollama` | OpenAI-compatible | none (local) | — |
| `openai_compatible` | OpenAI-compatible | optional | `OPENAI_COMPATIBLE_API_KEY` |

Every provider above reports `needs_structured_output: true` under the shipped
model catalog, so none forces the graph into a free-text fallback.

## Free API access — verified from official documentation

Free web chat is **not** free API access. Each row below is a claim about an
*API* tier, with the official page it came from and the date it was checked. A
documented free tier is not proof a key is held; a key is not proof the tier is
free.

| Provider | Free API tier | Official source | Checked |
| --- | --- | --- | --- |
| `groq` | Yes — a documented "Free Plan" with per-model RPM/RPD/TPM limits | https://console.groq.com/docs/rate-limits | 2026-10-28 |
| `google` | Yes — a Free tier with per-model RPM/TPM/RPD limits; Tier 1 needs billing | https://ai.google.dev/gemini-api/docs/rate-limits | 2026-10-28 |
| `openrouter` | Yes — free model variants (IDs ending `:free`) with their own caps | https://openrouter.ai/docs/api-reference/limits | 2026-10-28 |
| `ollama` | Yes — a local runtime, no account, no per-token cost | https://ollama.com | 2026-10-28 |

Checked and **not** found to have a documented free API tier:

| Provider | Finding | Checked |
| --- | --- | --- |
| `openai` | Pay-as-you-go API; no free API tier documented | 2026-10-28 |
| `anthropic` | Pay-as-you-go API; no free API tier documented | 2026-10-28 |
| `deepseek` | No free tier found on the official pricing page | 2026-10-28 |

Everything else is reported `unverified`: this audit did not confirm a free API
tier for it and does not guess one.

## Smallest change to enable genuine inference

The evaluation is provider-agnostic — it goes through the existing
`create_llm_client` factory. No new SDK, provider or code change is needed to
use a key. The smallest step is:

```bash
# Option 1 — a documented free API key (Groq shown; Gemini/OpenRouter work too)
export GROQ_API_KEY=...          # from console.groq.com, free plan
export TRADINGAGENTS_LLM_PROVIDER=groq
export TRADINGAGENTS_DEEP_THINK_LLM=<a groq model id, e.g. openai/gpt-oss-120b>
export TRADINGAGENTS_QUICK_THINK_LLM=<a groq model id, e.g. openai/gpt-oss-20b>

# Option 2 — a local OpenAI-compatible endpoint (no account, no cost)
export TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:8000/v1
export TRADINGAGENTS_LLM_PROVIDER=openai_compatible
export TRADINGAGENTS_DEEP_THINK_LLM=<local model id>
```

Then re-run the evaluation. A key alone is not enough: the evaluation also needs
a **captured decision record set** (see below). That capture path now exists
(Phase 8) — see "The capture path" below.

## The selected provider, and why

Phase 8 selects **Groq** as the default for the bounded smoke test:

- Its API is OpenAI-compatible, so it reuses the existing `create_llm_client`
  factory with no new integration code.
- It documents a **Free Plan** and lists production models with per-token prices
  in its official docs, so the default model is a documented, non-preview one.
- The default model is `openai/gpt-oss-20b`, a *production* model (not a preview
  model, which the docs say is for evaluation and may be discontinued).

Free web-chat access is not free API access, and this audit does not treat it as
such: only providers whose **official API documentation** states a free tier are
listed. The smoke test still stops before any request when no key is set.

## The capture path

`tradingagents/llm_evaluation/capture.py` turns a completed graph run into a
validated `DecisionRecord`. It reads the rating, the input window (hashed from the
same session-bounded feed the market analyst sees), the config hash and the
model/provider metadata. It refuses a rating with no tradeable action rather than
degrading it to Hold, and it invents no size, stop or limit.

The record's `source` is derived from the **graph**, not the caller: the graph
carries an `InferenceObserver` on its real provider clients, and a scripted or
stubbed model never fires it. So a mock run is always captured as a
`synthetic_fixture`, never `llm_inference`, even if the caller asks for
`REAL_INFERENCE_CAPTURE`. A credential-shaped value in the decision text is
scrubbed before it can reach a record.

## What a key alone does *not* unlock

Two gaps remain after a key is set, and neither is a provider problem:

1. **No point-in-time news/sentiment/fundamentals source.** A `full`-scope
   historical run is refused, not faked; only `price_only` (the market analyst)
   is point-in-time safe. See the point-in-time audit. A genuine capture
   therefore opens only the price-only scope; the full graph stays blocked.
2. **Input reproducibility is not point-in-time safety.** The record carries a
   window hash so a replay can prove it saw the same bars, but a hash is not
   proof those bars were available at the decision time. The two are reported
   separately in the artifact's `capture_status`.

## User action required

To enable genuine inference, do exactly one of:

- **Set a provider key** for a documented free tier (`groq`, `google`,
  `openrouter`) or a paid provider you already use, then run the bounded smoke
  test to capture one real decision; or
- **Point the client at a local OpenAI-compatible endpoint** with a model that
  supports tool calling.

Do not set a key expecting a performance number: even a successful smoke test is
a single current observation, not a historical result. The honest evaluation
status stays `LLM BASELINE BLOCKED` for a full-graph historical run until the
data-side blocker is cleared.

## Limitations

- Reachability and free-tier status are **static facts checked on 2026-10-28**,
  not probed at run time. A provider's terms can change.
- `unverified` means this audit did not confirm a free API tier; it is not a
  claim that none exists.
- The readiness rule is conservative: a value that looks like a placeholder is
  treated as no credential, so the report errs toward "blocked".
- No live request was made and no account was created; enabling inference is the
  user's action.
