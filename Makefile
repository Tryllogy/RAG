UV := uv
SRC := student
QUESTION := How to configure OpenAI server?
K := 10
MAX_CHUNK_SIZE := 2000

install:
	if [ -f "data.tar" ] && [ ! -d "data" ]; then \
	tar -xvf data.tar; \
	fi
	$(UV) sync
	if [ ! -f "qwen-0.6b_Q8.gguf" ]; then \
	wget https://huggingface.co/mradermacher/qwen-0.6b-coder-GGUF/resolve/main/qwen-0.6b-coder.Q8_0.gguf \
	-O qwen-0.6b_Q8.gguf; \
	fi
	

run: install
	$(UV) run python -m $(SRC)

answer: install
	uv run python -m student answer --question "$(QUESTION)" --k $(K)

answercode: install
	uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/search_results_and_answer

answerdoc: install
	uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

search: install
	uv run python -m student search --query "$(QUESTION)" --k $(K)

index: install
	uv run python -m student index --max_chunk_size $(MAX_CHUNK_SIZE)

evaluatecode: install
	uv run python -m student evaluate data/output/search_results/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k $(K)

evaluatedoc: install
	uv run python -m student evaluate data/output/search_results/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k $(K)

datadoc:
	uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k $(K) --save_directory data/output/search_results

datacode:
	uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k $(K) --save_directory data/output/search_results

lint:
	flake8 src/*
	mypy ./src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	flake8 student/*
	mypy ./student --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name "__pycache__" | xargs rm -rf || true
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info