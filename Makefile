UV := uv
SRC := student
QUESTION := What activation formats does the fused batched MoE layer return in vLLM?

install:
	$(UV) sync
	ollama pull qwen3:0.6b
	

run: install
	$(UV) run python -m $(SRC)

answer: install
	clear && uv run python -m student answer "$(QUESTION)" --k 10

answercode: install
	clear && uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/search_results_and_answer

answerdoc: install
	clear && uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

search: install
	clear && uv run python -m student search --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --query "$(QUESTION)" --k 10

index: install
	clear && uv run python -m student index --max_chunk_size 2000

evaluatecode: install
	clear && uv run python -m student evaluate data/output/search_results/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10

evaluatedoc: install
	clear && uv run python -m student evaluate data/output/search_results/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10

datadoc:
	clear && uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results

datacode:
		clear && uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results

lint:
	flake8 src/*
	mypy ./src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	flake8 student/*
	mypy ./student --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name "__pycache__" | xargs rm -rf || true
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info
	rm -rf data/processed data/output