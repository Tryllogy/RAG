UV := uv
SRC := student
QUESTION :=

install:
	$(UV) sync
# 	ollama serve
# 	ollama pull qwen3
# 	ollama launch qwen3
	

run: install
	$(UV) run python -m $(SRC)

question: install
	clear && uv run python -m student answer "$(QUESTION)" --k 10