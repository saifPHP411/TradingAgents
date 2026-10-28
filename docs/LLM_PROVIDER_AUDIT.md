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
a **captured decision record set** (see below), because no capture path exists in
this release.

## What a key alone does *not* unlock

Two gaps remain after a key is set, and neither is a provider problem:

1. **No capture path.** Nothing in `tradingagents/llm_evaluation/` constructs a
   `DecisionRecord` from a real graph run. `DecisionRecord` is built only in
   tests. So even with a working key, `build_evaluation` has no record set to
   evaluate and stays `LLM BASELINE BLOCKED`. A capture step — run the graph over
   the frozen dataset and serialise its decisions — is the missing piece, and it
   is deliberately not added here: it needs the data-side blocker resolved first
   (below), or it would capture decisions made from inputs the graph cannot
   honestly obtain for a historical date.
2. **No point-in-time news/sentiment/fundamentals source.** A `full`-scope
   historical run is refused, not faked; only `price_only` (the market analyst)
   is point-in-time safe. See the point-in-time audit.

## User action required

To enable genuine inference, do exactly one of:

- **Set a provider key** for a documented free tier (`groq`, `google`,
  `openrouter`) or a paid provider you already use, *and* accept that the
  evaluation will still be blocked until a capture path and PIT-safe inputs
  exist; or
- **Point the client at a local OpenAI-compatible endpoint** with a model that
  supports tool calling.

Do not set a key expecting a performance number: the honest result today is
`LLM BASELINE BLOCKED` with the named prerequisites, and it stays that way until
the data-side blockers are cleared.

## Limitations

- Reachability and free-tier status are **static facts checked on 2026-10-28**,
  not probed at run time. A provider's terms can change.
- `unverified` means this audit did not confirm a free API tier; it is not a
  claim that none exists.
- The readiness rule is conservative: a value that looks like a placeholder is
  treated as no credential, so the report errs toward "blocked".
- No live request was made and no account was created; enabling inference is the
  user's action.
