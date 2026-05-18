# Contributing to customer-churn-analysis

Thank you for contributing! Please follow these guidelines.

## Setup

```bash
git checkout -b feat/your-feature
make install       # installs deps + pre-commit hooks
make test          # verify existing tests pass
```

## Code Standards
- **Formatter**: Black (120 char line length)
- **Linter**: flake8 with bugbear
- **Imports**: isort (Black-compatible)
- Run `make format && make lint` before committing

## Testing
- Add tests in `tests/` for any new feature
- Coverage target: >80%
- Run: `make test-cov`

## Commit Messages (Conventional Commits)
```
feat: new feature
fix: bug fix
docs: documentation
chore: maintenance
refactor: restructure
test: test changes
```

## Pull Request
1. Run `make test` and `make lint`
2. Describe changes clearly
3. Reference related issues

By contributing, you agree to the MIT License.
