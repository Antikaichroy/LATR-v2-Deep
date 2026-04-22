# Files Overview

This repository currently contains a small documentation-oriented proof of concept for demonstrating AI code agent adoption planning.

## [ai-code-agents-poc-plan.md](e:\MOL-PROJECT-4\ai-code-agents-poc-plan.md)

This is the main source document for the POC.

It explains:

- the purpose of evaluating AI code agents in an enterprise setting
- a free or near-zero-cost proof of concept approach using VS Code and Codex
- how to demonstrate MCP connectivity, Docker packaging, PR lifecycle reduction, and governance controls
- how to measure productivity, developer experience, and cost efficiency
- how to present recommendations for controlled adoption

This is the editable source that can be updated if you want to refine the narrative or convert it into slides later.

## [ai-code-agents-poc-plan.pdf](e:\MOL-PROJECT-4\ai-code-agents-poc-plan.pdf)

This is the generated PDF version of the POC plan.

It is the presentation-ready artifact that you can share with stakeholders who prefer a document instead of Markdown.

## [generate_pdf.py](e:\MOL-PROJECT-4\generate_pdf.py)

This is a lightweight local PDF generator script.

It:

- reads the Markdown content from `ai-code-agents-poc-plan.md`
- converts the text into wrapped plain lines
- lays those lines out across PDF pages
- writes a simple PDF file without requiring external packages

This was created so the PDF could be generated locally for free without needing tools like Pandoc or Word automation.

## [hello.py](e:\MOL-PROJECT-4\hello.py)

This is a minimal placeholder Python file that existed in the workspace before the documentation files were added.

Right now it is not part of the POC logic. If you want, we can turn it into a small sample app for the live demo, such as:

- a simple web service
- a small health-check endpoint
- a demo application that Codex can update and package with Docker

## Suggested Next Step

If the goal is to make this repo more demo-ready, the next useful addition would be a tiny runnable sample app plus:

- a `Dockerfile`
- a `.dockerignore`
- a sample PR template
- a short demo script for the presenter
