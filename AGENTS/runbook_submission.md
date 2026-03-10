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
- Routed serving, if used, keys only on websocket `payload["task_name"]`
- `sorting_packages_continuous` is covered explicitly and reuses the intended adapter
- Unknown `task_name` falls back to `_default`
- The final router does not contain undocumented aliases outside the public ICRA route-key list
- Same-task dataset shards (`*_part_*`) are merged during training/export and do not appear as separate route keys

## To Fill In

- Final Dockerfile path
- Final startup command
- Local smoke test command
- Final public route-key list used for adapter routing
- Registry login workflow
- Exact submission form fields
- Log download troubleshooting notes
