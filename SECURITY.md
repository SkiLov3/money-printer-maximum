# Security Policy

## Reporting a Vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
[private vulnerability reporting](../../security/advisories/new) for this repository.
Include a description, reproduction steps, and the potential impact. You can expect an
acknowledgement within a reasonable time; this is a personal research project maintained
on a best-effort basis.

## Credentials & Secrets

- This project never needs you to commit secrets. Copy `.env.template` to `.env` (which is
  git-ignored) and keep your TopstepX/ProjectX API key, username, and account ID there.
- If you believe you have accidentally exposed a key (in a fork, issue, or log), **revoke it
  immediately** in your TopstepX/ProjectX account and generate a new one.
- Set `DASHBOARD_PASSWORD` before exposing the dashboard beyond `localhost`.

## Trading Safety

- Real orders are sent only when both `MODE=live` and `ACTUAL_LIVE_TRADING=true`.
  Default is simulation.
- This software is provided for educational and research purposes under the repository
  [LICENSE](LICENSE). It is not financial advice and comes with no warranty. You are solely
  responsible for any trading losses.
