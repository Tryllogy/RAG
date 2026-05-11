UV := uv
SRC := student
FIRE :=

install:
	$(UV) sync

run: install
	$(UV) run python -m $(SRC)