dev-up-detached:
	docker compose --profile dev up -d

dev-up-attached:
	docker compose --profile dev up

dev-logs:
	docker compose --profile dev logs -f

dev-rebuild:
	docker compose --profile dev build api-dev && docker compose --profile dev up -d

dev-down:
	docker compose --profile dev down

prod-up-detached:
	docker compose --profile prod up -d

prod-up-attached:
	docker compose --profile prod up

prod-rebuild:
	docker compose --profile prod build api-prod && docker compose --profile prod up -d

prod-down:
	docker compose --profile prod down