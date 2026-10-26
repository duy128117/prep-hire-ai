# AI Interview Research Pipeline

This directory documents the research path that replaces CV-to-job matching as the
main AI contribution. Existing matching endpoints remain available for compatibility,
but new work should use extracted CV/JD context as interview input.

## Current design: multi-LLM weak supervision

The primary training flow is now:

```text
CV + JD + interview state
  -> multiple LLMs generate questions
  -> three other LLMs independently judge each question
  -> multi-LLM consensus pseudo-labels
  -> Question Quality Ranking Model
  -> synthetic candidate answers at quality levels 1-5
  -> three other LLMs independently judge each answer
  -> multi-LLM consensus pseudo-labels
  -> Candidate Answer Scoring Model
  -> owner manual review for quality control and versioned label correction
```

AI-generated labels are **pseudo-labels**, not expert ground truth. The model that
generated an item must not judge that same item. The current scope does not use expert
labels or expert consensus. Optional owner manual reviews are stored separately and are
never mixed with raw AI judge records. See
`BAO_CAO_QUY_TRINH_DU_LIEU_VA_TRAINING_AI_V1.md` for the complete Vietnamese report
covering the first 1,000-set generation and training run.

Context-only endpoints:

- `POST /api/v1/interview/extract-context` accepts CV and JD text.
- `POST /api/v1/interview/extract-context-files` accepts CV and JD files.
- Neither endpoint computes a match score, skill gap, or learning roadmap.

Text interview endpoints:

- `POST /api/v1/interview/sessions` creates a session and its first question.
- `POST /api/v1/interview/sessions/{session_id}/answers` stores an answer and creates
  the next adaptive question.
- `GET /api/v1/interview/sessions/{session_id}` returns current state and turn history.
- Set `allow_external_ai=true` only after the candidate agrees that redacted CV/JD
  excerpts and interview history may be processed by configured external AI services.
  Adaptive question generation defaults to Inception Mercury; legacy CV enrichment may
  still use Gemini. The default is `false`.
- Email addresses, phone numbers, and URLs are redacted before an external prompt is
  created. Names and free-form text can still contain identifying details, so candidate
  consent and an explicit retention policy remain required.

Evaluation and expert-reference endpoints:

- `POST /api/v1/interview/sessions/{session_id}/evaluations` runs one LLM evaluation
  after the interview is completed. An optional Google model name supports repeated
  benchmark runs without overwriting earlier outputs.
- `GET /api/v1/interview/sessions/{session_id}/evaluations` returns raw LLM outputs.
- `POST /api/v1/interview/sessions/{session_id}/expert-annotations` creates one locked,
  independent expert annotation identified by a pseudonymous `expert_id`.
- `GET /api/v1/interview/sessions/{session_id}/expert-consensus` returns per-criterion
  dispersion, ICC(2,k), and a median reference only when at least three experts exist.

Multi-LLM question-quality endpoints:

- `POST /api/v1/interview/benchmarks/cases/from-session` freezes one consented,
  redacted CV/JD/interview-state snapshot as a reusable test case.
- `POST /api/v1/interview/benchmarks/runs` sends that exact snapshot and prompt version
  to 2-8 Google model names. A fallback question is recorded as a failed generation,
  never as valid benchmark evidence.
- `POST /api/v1/interview/benchmarks/runs/{run_id}/ai-judge` collects independent
  AI-judge labels while excluding each output's generator model.
- `GET /api/v1/interview/benchmarks/runs/{run_id}/blind` hides provider/model identity.
- `POST /api/v1/interview/benchmarks/outputs/{output_id}/human-audits` stores an
  optional locked human audit separately from AI labels.
- `GET /api/v1/interview/benchmarks/runs/{run_id}/summary` reports pseudo-label
  medians, dispersion, mean quality, latency, ICC(2,k), and human-review flags.

Synthetic-answer endpoints:

- `POST /api/v1/interview/benchmarks/outputs/{output_id}/synthetic-answers` creates
  controlled answers at requested quality levels 1-5.
- `POST /api/v1/interview/benchmarks/answer-samples/{sample_id}/ai-judge` collects at
  least three AI judgments while excluding the answer generator.
- `GET /api/v1/interview/benchmarks/answer-samples/{sample_id}/blind` hides generator
  identity and requested quality from human reviewers.
- `POST /api/v1/interview/benchmarks/answer-samples/{sample_id}/human-audits` stores a
  human audit.
- `GET /api/v1/interview/benchmarks/answer-samples/{sample_id}/summary` reports the
  answer pseudo-label consensus.

Protect case creation, benchmark execution, and summary endpoints behind a researcher
role in the main backend. Give experts access only to the blind endpoint and annotation
submission. Reuse the same cases across models; changing the CV, JD, history, stage, or
prompt version creates a different experimental condition.

LLM scores and expert scores are stored in separate tables. Never show an expert the
LLM score before their blind annotation is submitted. Protect evaluation and consensus
read endpoints behind an administrator/researcher role in the main backend. Partial
consensus statistics are withheld until the third expert submits.

## Rubrics

Canonical rubric definitions live in `app/research/rubrics.py`.

- Question quality has seven criteria for RQ1 blind expert evaluation.
- Candidate evaluation has seven criteria for RQ2 and calibration training.
- Every criterion uses a 1-5 scale. Write a separate annotation guideline with
  behavioral anchors before collecting real expert labels.

## Weak-supervision datasets and training

## Balanced dataset generation

The production generator uses one equal provider pool: Groq, Cloudflare Workers AI,
Inception Mercury 2.5, OpenTyphoon 2.5, Ollama Cloud, and NVIDIA Nemotron 3 Ultra.
There is no core or backup classification. Every round selects four distinct available providers, then advances a
round-robin cursor so all configured providers participate. A provider that reaches a
hard quota or becomes unavailable is removed from the current run. Generation continues
while at least four providers remain and pauses only when the pool falls below four.
Retired Cohere and Hugging Face rows remain valid historical data. One round creates
four complete question/answer sets on the same CV+JD context. Each generated question
and answer is judged by the other three selected providers.

Configure the independent AI slots in `.env`; no extra SDK is required:

```env
GROQ_API_KEY="your_primary_groq_api_key_here"
GROQ_API_KEY_2="your_backup_groq_api_key_here"
GROQ_API_KEY_3="your_third_groq_api_key_here"
GROQ_API_KEY_4="your_fourth_groq_api_key_here"
# Continue sequentially through GROQ_API_KEY_13.
CLOUDFLARE_ACCOUNT_ID="your_primary_cloudflare_account_id_here"
CLOUDFLARE_API_TOKEN="your_primary_cloudflare_api_token_here"
CLOUDFLARE_ACCOUNT_ID_2="your_second_cloudflare_account_id_here"
CLOUDFLARE_API_TOKEN_2="your_second_cloudflare_api_token_here"
CLOUDFLARE_ACCOUNT_ID_3="your_third_cloudflare_account_id_here"
CLOUDFLARE_API_TOKEN_3="your_third_cloudflare_api_token_here"
CLOUDFLARE_ACCOUNT_ID_4="your_fourth_cloudflare_account_id_here"
CLOUDFLARE_API_TOKEN_4="your_fourth_cloudflare_api_token_here"
# Continue matching account/token pairs through suffix _13.
INCEPTION_API_KEY="your_inception_api_key_here"
INCEPTION_API_KEY_2="your_second_inception_api_key_here"
INCEPTION_API_KEY_3="your_third_inception_api_key_here"
INCEPTION_API_KEY_4="your_fourth_inception_api_key_here"
# Continue sequentially through INCEPTION_API_KEY_13.
OPENTYPHOON_API_KEY="your_opentyphoon_api_key_here"
OPENTYPHOON_API_KEY_2="your_second_opentyphoon_api_key_here"
OPENTYPHOON_API_KEY_3="your_third_opentyphoon_api_key_here"
OPENTYPHOON_API_KEY_4="your_fourth_opentyphoon_api_key_here"
# Continue sequentially through OPENTYPHOON_API_KEY_13.
OLLAMA_API_KEY="your_ollama_cloud_api_key_here"
OLLAMA_API_KEY_2="your_second_ollama_cloud_api_key_here"
OLLAMA_API_KEY_3="your_third_ollama_cloud_api_key_here"
OLLAMA_API_KEY_4="your_fourth_ollama_cloud_api_key_here"
# Continue sequentially through OLLAMA_API_KEY_13.
NVIDIA_API_KEY="your_nvidia_api_key_here"
NVIDIA_API_KEY_2="your_second_nvidia_api_key_here"
NVIDIA_API_KEY_3="your_third_nvidia_api_key_here"
NVIDIA_API_KEY_4="your_fourth_nvidia_api_key_here"
# Continue sequentially through NVIDIA_API_KEY_13.
NVIDIA_NEMOTRON_MODEL="nvidia/nemotron-3-ultra-550b-a55b"
```

Each provider has one primary credential and up to twelve sequential backups. These
credentials still represent one AI worker, not thirteen workers. The primary key is used
until authentication or quota exhaustion, then the process keeps using the next valid
key. Duplicate key values are removed automatically. Cloudflare credentials must be
filled as matching `ACCOUNT_ID_N` and `API_TOKEN_N` pairs from the same account.

Seed privacy-safe synthetic candidate contexts from the public crawled JDs:

```powershell
python -B scripts\generate_weak_supervision_dataset.py --seed-synthetic-cases
```

Inspect progress without making an AI request:

```powershell
python -B scripts\generate_weak_supervision_dataset.py --status
```

Generate or resume the 1,000-set target:

```powershell
python -B scripts\generate_weak_supervision_dataset.py --target-sets 1000
```

Every round is one database transaction. A quota, authentication, network, or provider
server failure rolls back all four sets in that round. The checkpoint marks the failed
provider as unavailable, selects another provider from the equal pool, and retries the
same round. Generation pauses only when fewer than four providers remain. Repeated
invalid JSON can also remove a provider from the current run. Token usage, active slots,
failover history, and the pause reason are stored
in `data/research/generation_checkpoint_v1_1.json`. Only complete, logical-slot-balanced
rounds are committed. Optionally set `DATASET_PROVIDER_TOKEN_BUDGET` to a positive
per-provider soft limit; zero relies on provider quota responses.

Export both pseudo-labeled datasets:

```powershell
python -B scripts\export_weak_supervision_datasets.py
```

Outputs:

- `data/research/weak_supervision_v1_1/all/question_ranking_all.csv`
- `data/research/weak_supervision_v1_1/all/turn_evidence_all.csv`

Create deterministic 70/15/15 train, validation, and held-out test partitions. The
split is grouped by benchmark case, stratified by stage/criterion, and shared by both
tasks, so the same CV/JD context cannot leak across partitions:

```powershell
python -B scripts\split_weak_supervision_datasets.py
```

Outputs:

- `data/research/weak_supervision_v1_1/splits/question_train.csv`
- `data/research/weak_supervision_v1_1/splits/question_validation.csv`
- `data/research/weak_supervision_v1_1/splits/question_test.csv`
- `data/research/weak_supervision_v1_1/splits/answer_train.csv`
- `data/research/weak_supervision_v1_1/splits/answer_validation.csv`
- `data/research/weak_supervision_v1_1/splits/answer_test.csv`
- `data/research/weak_supervision_v1_1/manifest.json`

Train both baseline models:

```powershell
python -B scripts\train_weak_supervision_models.py
```

Outputs:

- `artifacts/weak_supervision/v1_1/question_quality_ranker.joblib`
- `artifacts/weak_supervision/v1_1/question_quality_ranker_metrics.json`
- `artifacts/weak_supervision/v1_1/candidate_answer_scorer.joblib`
- `artifacts/weak_supervision/v1_1/candidate_answer_scorer_metrics.json`

Both baseline trainers use the precomputed grouped partitions. They fit on train,
measure validation performance, refit on train plus validation, and evaluate once on
the held-out test set. Generator model name and requested answer quality are retained
only as analysis metadata and are not primary training features. Metrics against the
held-out pseudo-label test set measure imitation of the AI jury, not agreement with
experts. The project owner may manually review flagged samples for quality control and
create a new versioned dataset without an expert-consensus requirement. The V1.1
answer scorer fits one independent regressor per turn-evidence criterion, using only
rows where that criterion applies instead of inventing labels for non-applicable rows.

Create a blind owner-review queue for the locked V1 ranker errors:

```powershell
python -B scripts\analyze_question_ranker_errors.py `
  --dataset data\research\weak_supervision_v1_1\splits\question_test.csv `
  --name question_ranker_v1_test
```

Train the V2 pairwise learning-to-rank candidate and benchmark it without split leakage:

```powershell
python -B scripts\export_question_pairwise_datasets.py
python -B scripts\train_question_ranker_v2.py
python -B scripts\benchmark_question_rankers.py
```

V2 learns ordered question pairs within each CV/JD context. Model selection uses only
validation metrics in this order: tie-aware Top-1, MRR, pairwise accuracy, then NDCG.
The held-out test partition is reporting-only and is never used to select V2.

## Archived expert-calibration path (not used)

The following Candidate Evaluation Calibration pipeline is legacy code and is not part
of the current training scope. Weak Supervision V1 does not read these files, tables, or
labels.

Start with `data/research/calibration_dataset_template.csv`. One row represents one
completed interview session for one candidate and one JD.

Rules for real collection:

1. Keep candidate identities pseudonymous.
2. Use at least three independent experts per session when feasible.
3. Store each expert's original annotation separately; put only the agreed reference
   value in the `expert_consensus_*` columns used by training.
4. Never use synthetic rows as thesis evidence.
5. Keep all sessions from one candidate in the same train or test partition.

The trainer requires at least 20 rows to catch accidental empty runs. For defensible
experiments, target substantially more candidates and report confidence intervals.

## Legacy calibration commands

First export consented sessions for exactly one evaluated model:

Set a private pseudonymization secret in `.env` first. Keep the same value for every
export in one study and never commit it:

```env
RESEARCH_PSEUDONYM_SALT=replace-with-at-least-16-random-characters
```

```powershell
python -B scripts\export_calibration_dataset.py `
  --model-name gemini-2.5-flash `
  --output data\research\calibration_dataset.csv
```

Then train and evaluate its calibrator:

```powershell
python -B scripts\train_candidate_calibration.py `
  --dataset data\research\calibration_dataset.csv
```

Outputs:

- `artifacts/candidate_calibrator.joblib`: fitted preprocessing and RidgeCV model.
- `artifacts/candidate_calibration_metrics.json`: raw LLM versus calibrated MAE,
  RMSE, Pearson, Spearman, and ICC(3,1), overall and by criterion.

The calibration model is successful only when held-out metrics improve. A trained
artifact by itself is not evidence of improvement.

## Next phases

1. Resume the balanced pilot after every provider has usable quota.
2. Human-audit a stratified subset before scaling pseudo-label generation.
3. Inspect provider token usage and disagreement after each generation batch.
4. Train the two baselines only after at least five independent context groups exist.
5. Add STT/TTS and measure WER plus P50/P95 latency after the text workflow is stable.

## Realtime trained-model integration

The realtime interview flow now uses the two V1.1 artifacts:

- `question_pairwise_ranker_v2.joblib` ranks up to four questions generated by
  independent configured providers for the same CV/JD context.
- `candidate_answer_scorer.joblib` produces seven turn-level evidence scores after
  each answer. These are evidence signals, not a final candidate decision.

Check artifact and provider readiness:

```http
GET /api/v1/interview/realtime-models/status
```

Create a normal session with multi-provider generation by setting
`allow_external_ai=true`. Email addresses, phone numbers, and URLs are redacted before
the question prompts are sent. CV/JD profile extraction remains local so the context is
not sent to an additional enrichment provider.

## Synthetic interview sessions

Synthetic contexts are deterministic, explicitly labeled with `context_source=synthetic`,
and must not be reported as real candidate evidence.

Preview generated CV/JD contexts without using the database or external AI:

```powershell
python -B scripts\run_synthetic_interviews.py --sessions 3 --dry-run-contexts
```

Create and complete local-fallback sessions while running the trained answer scorer:

```powershell
python -B scripts\run_synthetic_interviews.py --sessions 3 --max-questions 5
```

Run the same simulation with four external question generators and local V2 ranking:

```powershell
python -B scripts\run_synthetic_interviews.py `
  --sessions 1 `
  --max-questions 5 `
  --allow-external-ai
```

The external mode consumes provider quota. Failed or duplicate providers are replaced
from the configured backup pool when possible. Session progression remains available
through a deterministic fallback if every provider fails.
