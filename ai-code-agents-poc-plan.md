# AI Code Agents: Evaluation and Controlled Adoption POC Plan

Prepared on April 22, 2026

## 1. Purpose

Evaluate and demonstrate how AI Code Agents can improve software delivery productivity and quality while preserving enterprise controls for security, governance, compliance, and cost.

## 2. Executive Summary

This proof of concept should be positioned as a controlled, low-cost developer workflow demo using tools you already have: VS Code plus Codex. The live story should show three things:

1. Codex can work inside a real developer workflow from the IDE.
2. Codex can connect to MCP-style tools and external systems in a controlled way.
3. Codex can reduce delivery friction by speeding up common tasks such as code changes, review preparation, validation, and deployment packaging.

The cleanest free demo is:

- Start with a tiny sample app.
- Ask Codex to make a scoped change.
- Use MCP or local tools to inspect docs, files, or repo context.
- Package the app with Docker.
- Generate a PR-ready summary, risk list, and test notes.
- Show before/after cycle-time evidence using a stopwatch and a simple metrics sheet.

## 3. POC Goal

Prove that AI Code Agents can be adopted safely in a controlled enterprise model before any broad rollout.

## 4. Success Criteria

### Productivity

- Reduce time to complete a small feature or bug-fix task by 30 to 50 percent versus a manual baseline.
- Reduce PR preparation time by 40 percent or more.
- Reduce time spent writing PR descriptions, test notes, and deployment instructions by 60 percent or more.

### Developer Experience

- Developers rate the workflow at 4 out of 5 or better for ease of use.
- Developers can complete the full demo flow inside VS Code with limited context switching.

### Cost Efficiency

- Additional POC spend is zero or near-zero beyond existing access to Codex and local tooling.
- Every demo step has a measurable unit cost or can be estimated by task duration and token consumption where applicable.

## 5. Recommended Free POC Scope

Use a very small application that is easy to understand in a live demo, such as:

- A Python Flask or FastAPI hello-world service.
- A simple Node.js API.
- A static web app with one backend endpoint.

Recommended scenario:

"Take a tiny app, add one business change, containerize it, produce review artifacts, and show the deploy path."

This is small enough to run live, but broad enough to demonstrate engineering value.

## 6. What To Demonstrate Live

### Demo 1: Assisted Change Delivery in VS Code

Show Codex performing:

- Codebase discovery.
- Scoped code edits.
- Basic validation or tests.
- PR summary generation.
- Review notes with risks and assumptions.

Expected message to stakeholders:

"The agent is not replacing engineering judgment. It is compressing the mechanical parts of delivery."

### Demo 2: MCP Tool Connectivity

Show Codex connected to one or more MCP-compatible tools, such as:

- OpenAI developer docs or internal knowledge retrieval.
- GitHub or repo metadata.
- Local shell and file operations.
- Ticketing or backlog context if an MCP tool exists in your environment.

What to emphasize:

- Tool access is explicit and inspectable.
- The agent can be constrained to approved tools.
- This supports governance and auditability.

### Demo 3: Packaging and Deployment

Show Codex creating:

- A `Dockerfile`.
- A `.dockerignore`.
- Local run instructions.
- Optional GitHub workflow or release notes draft.

Because `docker` and `git` are not installed on this machine right now, treat this as either:

- a narrated demo artifact generation step, or
- a live run on a machine that has Docker Desktop and Git installed.

### Demo 4: PR Lifecycle Reduction

Compare two flows for the same change:

- Manual flow.
- Codex-assisted flow.

Measure:

- Time to understand the task.
- Time to implement the change.
- Time to write tests or validation steps.
- Time to prepare PR description and reviewer notes.
- Time to produce deployment instructions.

## 7. Suggested End-to-End Demo Script

### Step 1: Baseline

- Start stopwatch.
- Open the sample app.
- Read the task statement.
- Record estimated manual time for discovery, coding, test notes, and PR prep.

### Step 2: Ask Codex to Execute the Change

Example task:

"Add a `/health` endpoint, update the landing page text, add a Dockerfile, and prepare a PR summary with risks and test notes."

### Step 3: Show Tool Use

- Let Codex inspect files and generate the change.
- Show a connected MCP tool for docs or repo context.
- Highlight that the agent uses approved tools instead of unconstrained external access.

### Step 4: Show Review Readiness

- Display generated diff.
- Display generated PR summary.
- Display generated validation checklist.
- Display rollback or risk notes.

### Step 5: Show Deployment Readiness

- Show Docker artifacts.
- If available on another machine, run `docker build` and `docker run`.
- If GitHub is available, show how the same artifacts support PR creation and CI handoff.

### Step 6: Close With Metrics

- Stop stopwatch.
- Compare baseline and assisted time.
- Record qualitative developer feedback.

## 8. Free Tooling Stack

Use only free or already-available components:

- VS Code.
- Codex in the IDE.
- A local sample repo.
- Free MCP servers or local MCP tools.
- Docker Desktop Free only if your organization qualifies under Docker licensing terms. If not, use Podman Desktop or narrate the packaging step.
- GitHub Free with a public demo repo, or a local-only Git workflow if public hosting is not acceptable.
- Markdown plus PDF artifact generation for reporting.

Important caveat:

GitHub Free and Docker Desktop are not universally "absolutely free" for every enterprise usage pattern. For a strict enterprise-free demo, keep the live run local and treat GitHub or Docker cloud execution as optional visuals unless licensing is confirmed.

## 9. Security, Governance, and Compliance Controls

Frame the POC as "controlled adoption," not open experimentation.

Minimum controls:

- Approved repositories only.
- No production secrets in prompts.
- No regulated data in the demo.
- Tool allowlist for MCP access.
- Human review required before merge or deployment.
- Logging of prompts, outputs, and generated artifacts where policy permits.
- Standard secure coding checks still apply.
- Use sandboxed or local environments for execution.

## 10. Operating Model for the Pilot

### Participants

- 2 developers.
- 1 engineering manager or architect.
- 1 security reviewer.
- 1 platform or DevOps observer.

### Pilot Duration

- 2 weeks.

### Scope

- 3 to 5 representative tasks.
- 1 small service or application.
- 1 repo only.

### Guardrails

- No autonomous merge.
- No direct production deployment.
- No sensitive datasets.
- Manual approval for code and infrastructure changes.

## 11. Measurement Framework

Track each task in a simple sheet.

Columns:

- Task ID.
- Task type.
- Story points or complexity.
- Baseline manual estimate.
- Assisted elapsed time.
- Number of prompts.
- Approximate token usage or usage cost if available.
- Files changed.
- Review comments count.
- Rework count.
- Developer satisfaction score.

Primary KPI formulas:

- PR cycle-time reduction = `(manual baseline - assisted time) / manual baseline`
- Throughput improvement = `assisted tasks completed / baseline tasks completed`
- Cost per task = `tool cost + developer time cost`
- Review-prep reduction = `manual PR prep time - assisted PR prep time`

## 12. Evidence Package To Present

At the end of the POC, show:

- One short screen-recorded live demo.
- One before/after metrics table.
- One controls checklist.
- One architecture view of tools and approvals.
- One recommendation page: do not adopt, limited pilot, or scale gradually.

## 13. Recommended Enterprise Narrative

Use this message with stakeholders:

"We are not evaluating whether AI can write code in isolation. We are evaluating whether a controlled agent workflow can reduce low-value delivery effort, shorten PR preparation, improve developer experience, and maintain enterprise controls."

## 14. Suggested Recommendation Path

If the POC succeeds:

- Phase 1: opt-in developer pilot for non-sensitive repos.
- Phase 2: controlled team rollout with managed configuration and approved MCP tools.
- Phase 3: integration with PR templates, CI evidence, and internal engineering standards.

If the POC does not succeed:

- Restrict use to documentation, code search, and PR summarization only.

## 15. What You Can Say About OpenAI and Codex

Based on official OpenAI developer documentation reviewed on April 22, 2026:

- OpenAI positions Codex as a coding agent used across app, IDE, CLI, web, GitHub, MCP, and enterprise governance surfaces.
- OpenAI docs list Codex support for MCP, GitHub integrations, agent approvals and security, and a GitHub Action in the product documentation and navigation.
- OpenAI also highlights Codex use cases such as "Review pull requests faster."
- OpenAI developer docs currently state that, for a limited time, Codex is included in ChatGPT Free and Go, though this should be treated as time-sensitive and rechecked before any formal presentation.

## 16. Final Recommendation for Your Demo

Use a 10-minute demo with this structure:

1. State the governance problem.
2. Show the assisted delivery workflow in VS Code.
3. Show MCP tool connectivity.
4. Show Docker packaging artifact generation.
5. Show PR summary and review acceleration.
6. End with a one-slide metrics and control summary.

That is the smallest demo that still proves productivity, developer experience, cost visibility, and governance.

## 17. Source Notes

Official OpenAI sources consulted:

- OpenAI Developers homepage: https://developers.openai.com/
- Codex use cases: https://developers.openai.com/codex/use-cases
- Codex explore page: https://developers.openai.com/codex/explore/
- GPT-5.2-Codex model page: https://developers.openai.com/api/docs/models/gpt-5.2-codex
- GPT-5.1-Codex model page: https://developers.openai.com/api/docs/models/gpt-5.1-codex

## 18. Assumptions

- You already have access to VS Code and Codex.
- The presentation goal is a small POC, not a production rollout.
- The requirement "absolutely free" means no new paid software purchase for the demo itself.
