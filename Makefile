UV := uv
SRC := student
QUESTION := What determines whether vLLM's sampler returns Pythonized results or deferred Pythonization arguments?

install:
	$(UV) sync
# 	ollama serve
# 	ollama pull qwen3
# 	ollama launch qwen3
	

run: install
	$(UV) run python -m $(SRC)

answer: install
	clear && uv run python -m student answer "$(QUESTION)" --k 10

index: install
	clear && uv run python -m student index --max_chunk_size 2000

evaluate: install
	clear && uv run python -m student evaluate

clean:
	rm -rf data/processed