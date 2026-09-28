# Celebrity Face Training Runner

Public GitHub Actions runner for a small, license-filtered celebrity face generation smoke test.

The workflow downloads only Wikimedia Commons images whose metadata reports CC BY, CC BY-SA, CC0, or Public Domain terms, trains a tiny identity-conditioned generator on CPU, and uploads only model/results metadata — not the raw third-party training images.
