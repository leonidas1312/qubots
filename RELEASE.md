# Release process

## One-time PyPI setup

1. Create an account on https://pypi.org and verify email.
2. Reserve the project name with a one-time manual upload:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```
3. Configure **trusted publishing** so future releases use OIDC (no API
   token in CI):
   - Go to https://pypi.org/manage/project/qubots/settings/publishing/
   - Add a publisher: GitHub repo `leonidas1312/qubots`, workflow `release.yml`,
     environment `pypi`.
4. In GitHub repo settings → Environments, create the `pypi` environment
   (no secrets needed; OIDC handles auth).

After this is done, every tag push triggers a clean publish — no tokens
or secrets to manage.

## Cutting a release

```bash
# Bump version in pyproject.toml first.
git commit -am "Bump version to 0.2.0"
git tag v0.2.0
git push origin main --tags
```

GitHub Actions will:

1. Run the `ci.yml` matrix on every push.
2. On the tag push, trigger `release.yml`:
   - Build sdist + wheel.
   - Run `twine check` for metadata sanity.
   - Publish to PyPI via OIDC.

## Manual build + publish (fallback)

If the workflow fails or you need an out-of-band release:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
