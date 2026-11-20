# Day 11 Guardrails Implementation Design

## Intent
Complete the Lab 11 starter in the existing K5-L3B repo so Blue handles input/output guardrails and the defensive pipeline, while the submitted Red Team run targets the provided Red and Red Advance agents.

## Goals
- Implement CP2 injection/topic checks, PII/secret redaction, and ADK callbacks.
- Implement CP3 sliding-window limiting, audit/metrics, deterministic egress policy, ordered plugins, and schema-shaped results.
- Replace the five CP4 placeholder prompts with distinct, detailed techniques already required by the lab.
- Keep the supplied agents, secret data, and framework structure intact; generate artifacts only via lab commands.

## Approaches
- Keep the existing Google ADK plugins and starter modules: matches the checkpoint files and reduces integration risk. **Selected.**
- Replace the pipeline with a standalone framework: unnecessary because the starter already defines its plugin contracts.
- Add optional NeMo/Judge work: not required for Blue scoring and increases API/dependency cost; leave optional components untouched.

## Constraints and calls
- Blue remains on the configured OpenRouter model; Red keeps the prescribed provider/model.
- Egress uses an exact HTTPS host allowlist and deterministic sensitive-data checks.
- Tests validate local behavior without requiring live model calls; output-generation commands are attempted only if local keys are configured.
- No real API key may be added to source control.
