.PHONY: test backend-test frontend-typecheck frontend-build verify

backend-test:
	cd backend && python -m pytest -q

frontend-typecheck:
	cd frontend-app && npm run typecheck

frontend-build:
	cd frontend-app && npm run build

test: backend-test

verify:
	python scripts/verify_repo.py
