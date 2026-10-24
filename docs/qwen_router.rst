Qwen2.5-3B router for MoDE
=========================

Source and adaptation
---------------------

SearchJev (https://arxiv.org/pdf/2610.05107), sections 3.1–3.3 and appendix D,
uses a causal LM's existing vocabulary head to score single-token legal options,
without generating a response. Its SLCD objective combines soft-label
cross-entropy and Brier loss, followed by held-out temperature calibration.
The paper also permutes unordered options and delegates uncertain decisions.
It evaluates Qwen3.5 search agents; its results do not establish MoDE safety,
Qwen2.5 performance, or a speed advantage over ModernBERT.

MoDE implementation
-------------------

``scripts/qwen_router.py`` uses Qwen/Qwen2.5-3B-Instruct at revision
``aa8e72537993ba99e69dfaafa59ed015b17504d1``. The instruction-tuned checkpoint
is the initialization, with its native assistant prefix and existing LM head.
There is no learned classifier head and no generated JSON.

The input is the original prompt plus a fixed schema describing AdaSteer,
PIGuard, and GuardAgent. Benchmark outcomes, harmfulness labels, and measured
timings are supervision only. A/B/C must tokenize to distinct single tokens.
Only state tokens are truncated; the expert descriptions and assistant prefix
are retained. Chat delimiters inside the state are escaped before tokenization.
Training, calibration, and inference use the same serialization.

The last input position supplies A/B/C logits. A softmax over these three
logits, divided by the fitted positive temperature, returns expert choice
probabilities. Left padding and attention-mask-derived positions ensure a
padded batch produces the same decision positions as individual prompts.
Only last-position vocabulary logits are materialized.

Supervision and split contract
------------------------------

Preparation reuses the existing benchmark loader: complete unseen Aegis results
for all three experts, original prompt identity, valid finite timings, manifest
fingerprints, expert provenance, and reserved-test overlap checks. The current
repository unseen pool has 10,000 prompts; Aegis test has 1,914.

The existing seed-42, harmfulness-stratified 80/20 split supplies training and a
held-out pool. Half the held-out pool becomes checkpoint selection, half becomes
calibration (seed 43). IDs and normalized prompts are disjoint across all three
and reserved test. All-failed prompts have no routing target; they remain in
policy success reports. Target counts and excluded counts are saved per split.

The fastest successful expert is a one-hot target, with deterministic expert
order breaking latency ties. We do not invent uncertainty from a single timing
measurement. The encoder also accepts explicit expert probability distributions
and positive source weights if future supervision provides them; existing
benchmark-derived examples use one-hot labels and weight 1.

Training and calibration
------------------------

Defaults: three epochs, learning rate 5e-5, sequence budget 2,048, micro-batch 2,
accumulation 8, rank-16/alpha-32 LoRA, dropout 0.05, AdamW, weight decay 0.01,
cosine decay with 5% warmup, and gradient checkpointing. LoRA covers Qwen's
attention and MLP projections. All six expert permutations are encoded for each
training prompt, with targets remapped to the corresponding option positions.
Selection and calibration use the canonical expert order used at inference.

The per-example objective is weight * (cross-entropy + 0.5 * Brier), averaged
within the micro-batch; gradient accumulation scaling is explicit. The lowest
selection-loss adapter is merged into the base weights. A bounded positive
temperature grid minimizes NLL on calibration prompts only. The grid includes
T=1, so fitting cannot worsen calibration-split NLL. This does not guarantee
better calibration on unseen domains. NLL, Brier, 15-bin ECE, and routing
accuracy are reported; all-failed cases are excluded from those target metrics.

Inference and evaluation
------------------------

``QwenRouter(export_path).predict(prompt, confidence_threshold=0.8)`` returns
``expert``, ``probabilities``, ``confidence``, ``truncated``, and ``needs_review``.
The threshold is a caller-selected policy, not a validated safety threshold.
``needs_review`` flags a choice below the threshold; it does not invoke an
expert or another model. Callers must handle the flag before executing an
expert. The default threshold of zero keeps compatibility with MoDE's existing
forced-choice benchmark. No escalation benefit is claimed or measured.

Choice confidence estimates agreement with the fastest-successful target,
not the chosen expert's probability of success. Retain the existing outcome
caveats: gate-expert outcomes use prompt-label assumptions, and recorded expert
costs are not directly comparable end-to-end response costs.

Final evaluation checks exact expert configurations and reserved-test identity.
It compares the trained router, each fixed expert, and the fastest-successful
oracle, reporting harmful/unharmful success, routing accuracy, selection counts,
truncation, router latency, and estimated combined cost. Calibration metrics
are computed separately on labeled cases. Checkpoint-selection metrics remain
selection-set results; reserved-test evaluation is the final quality check.

Notebook and portability
------------------------

Open ``train_qwen_router.py`` with::

    uvx marimo@0.24.0 edit train_qwen_router.py --sandbox

It follows the ModernBERT/Laya workflow: prepare, download, inspect, CUDA smoke
test, fine-tune, evaluate reserved test, export/reload, and try a prompt. Model
and writing actions are button-gated. LoRA rank, alpha, and Brier weight are
editable alongside the existing optimizer and device controls.

Run data retain hashes, split provenance, settings, versions, training history,
calibration, and validation reports. Standalone exports include merged weights,
tokenizer, saved temperature and schema, provenance, and available reports.
Export verifies fresh-reload predictions and refuses to overwrite an existing
export. BF16 weights occupy roughly 6 GB per full checkpoint; FP32 is roughly
12 GB. Download, run, and export copies require additional storage. Production
training and latency should be measured on the target CUDA hardware.
