# Submission Runbook

This is a placeholder runbook for final packaging and submission.

## Scope

- Select final checkpoint
- Build serving image
- Validate websocket server locally
- Push image to competition registry
- Submit image URL
- Retrieve logs and results

## Hard Checks Before Submission

- Server auto-starts via `CMD` or `ENTRYPOINT`
- Server listens on port `8999`
- Model output type matches submission declaration
- Checkpoint and assets are present in image
- Local smoke test passes
- Image size is reasonable enough to push within token lifetime

## To Fill In

- Final Dockerfile path
- Final startup command
- Local smoke test command
- Registry login workflow
- Exact submission form fields
- Log download troubleshooting notes
