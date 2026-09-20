# UniRobot shortcuts (requires `just`: https://just.systems)
# Run from workspace root. Venv-aware: uses .venv/bin/python if present.

py := if path_exists(".venv/bin/python") == "1" { ".venv/bin/python" } else { "python3" }

default:
  @just --list

check:
  {{py}} -m g1_app.cli check

stand *ARGS:
  {{py}} -m g1_app.cli stand {{ARGS}}

gui *ARGS:
  {{py}} -m g1_app.cli gui {{ARGS}}

train *ARGS:
  {{py}} -m g1_app.cli train -- {{ARGS}}

train-stand *ARGS:
  {{py}} -m g1_app.cli train-stand -- {{ARGS}}

dashboard *ARGS:
  {{py}} -m g1_app.cli dashboard {{ARGS}}

record *ARGS:
  {{py}} -m g1_app.cli record {{ARGS}}

terrains:
  {{py}} -m g1_app.cli terrains

verify:
  {{py}} -m g1_app.cli verify

test:
  {{py}} -m pytest g1_app/tests -q

lint:
  {{py}} -m ruff check g1_app || true
