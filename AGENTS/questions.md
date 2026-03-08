# Open Questions

Track unresolved questions here so future agents do not repeatedly rediscover them.

## Current Questions

- What global batch size is realistically sustainable on the user's target hardware for the challenge config?
- Should the final competition model remain close to baseline ACoT-VLA or use a more aggressive LoRA setup?
- Is checkpoint export for submission best handled by reusing the repo serving Dockerfile or by introducing a dedicated submission Dockerfile?
- What local smoke test is sufficient to validate the websocket server before image push?
