UV := uv
SRC := student
FIRE :=

install:
	$(UV) sync
# 	ollama serve
# 	ollama pull qwen3
# 	ollama launch qwen3
	

run: install
	$(UV) run python -m $(SRC)