*This project has been created as part of the 42 curriculum by tchemin.*

# RAG Against the Machine

## Description

This project implements a retrieval-augmented generation (RAG) pipeline for answering questions about the vLLM codebase and documentation. The system first indexes a local snapshot of the repository, chunks files into searchable passages, retrieves the most relevant context with BM25, and then uses a local Ollama model to generate an answer from the retrieved evidence.

The project is split into two main workflows:

- Search and indexing, which builds reusable BM25 indexes for code and documentation.
- Answering and evaluation, which retrieves relevant chunks, generates answers, and measures retrieval quality against the answer key.

## Instructions

### Requirements

- Installation of Ollama for local LLM inference.
- Python 3.12 with the project virtual environment activated.
- `uv` for dependency management.
- A local Ollama instance with the `qwen3:0.6b` model available for answer generation.

### Installation

```bash
make
```

### Build the indexes

```bash
make index
```

This scans `data/raw/vllm-0.10.1`, chunks the repository, and writes the processed artifacts under `data/processed/`.

### Generate search results

```bash
make datacode
make datadoc
```

These commands search the unanswered code and documentation datasets and write results to `data/output/search_results/`.

### Answer questions

```bash
make answercode
make answerdoc
```

This reads the saved search results, retrieves context again, and generates answers with the local LLM.

### Evaluate retrieval

```bash
make evaluatecode
make evaluatedoc
```

These commands compute Recall@1, Recall@3, Recall@5, and Recall@10 against the answered datasets.

## System Architecture

The pipeline is organized as a sequence of small components under `src/`:

- `src/chunker/` scans the source tree and splits files into chunks.
- `src/index/` turns those chunks into BM25 indexes and saves them as `.npy` files.
- `src/retrieve/` loads the saved indexes and exposes retrieval for code or documentation queries.
- `src/search_dataset/` batches retrieval over the unanswered question datasets and stores the results.
- `src/answer/` retrieves context for a question and sends it to Ollama for answer generation.
- `src/evaluator/` compares retrieved chunks against the ground-truth sources and computes recall.

At runtime, the flow is:

1. Chunk the raw vLLM snapshot.
2. Build two BM25 indexes: one for Python code, one for prose/documentation.
3. Retrieve the top-k chunks for each question.
4. Optionally generate a final answer from the retrieved context.
5. Evaluate whether the retrieved chunks overlap with the annotated sources.

## Chunking Strategy

The chunking strategy is type-aware.

- Python files are parsed with `ast` and split into module headers, top-level functions, classes, and methods.
- Methods are enriched with a small parent-class prefix so class context survives retrieval.
- Non-Python files are chunked with `RecursiveCharacterTextSplitter` using a default chunk size of 2000 characters and 500 characters of overlap.
- Only `.txt`, `.py`, `.md`, and `.rst` files are included, while generated or unwanted files such as `setup.py` are skipped.

This approach keeps code chunks structurally meaningful and keeps documentation chunks close to human-readable paragraph boundaries.

## Retrieval Method

Retrieval is handled with `langchain_community.retrievers.BM25Retriever`.

- Code queries are preprocessed with a tokenizer that splits snake_case and camelCase identifiers into search-friendly tokens.
- Documentation queries use a simpler lowercase word tokenizer.
- The retriever ranks chunks by BM25 score and returns the top-k matches.
- Separate indexes are maintained for code and docs so the tokenization and ranking behavior match the content type.

The evaluation step measures overlap between retrieved source spans and the annotated ground-truth spans using an IoU-style threshold of 5 percent.

## Performance Analysis

The evaluator reports Recall@1, Recall@3, Recall@5, and Recall@10. Recall@5 is the main pass/fail metric used by the project:

- code dataset: target Recall@5 is at least 0.50
- docs dataset: target Recall@5 is at least 0.80

The current implementation is optimized for reproducible local evaluation rather than model-serving latency. The largest runtime costs are indexing the raw repository and querying BM25 across the stored chunk set. The answer-generation path is slower than retrieval alone because it also invokes a local LLM through Ollama.

## Design Decisions

- I used BM25 instead of a denser embedding index to keep the retrieval stack lightweight, deterministic, and easy to inspect.
- I kept separate retrieval pipelines for code and prose because they benefit from different preprocessing rules.
- I preserved source-span metadata on every chunk so evaluation can compare character-level overlap instead of only filenames.
- I split indexing, retrieval, answer generation, and evaluation into separate modules so each step can be run independently during debugging.
- I stored the processed BM25 payloads on disk so repeated runs do not need to reindex the raw repository.

## Challenges Faced

- Parsing Python into semantically useful chunks required careful handling of classes, nested methods, and module headers.
- Matching retrieved chunks back to the ground-truth sources required preserving exact character offsets during chunk creation.
- Keeping code and documentation retrieval separate was necessary to avoid noisy rankings caused by mixed tokenization behavior.
- The answer-generation flow depends on a local Ollama model, so the project needs a working local inference setup before the full pipeline can run end to end.

## Example Usage

```bash
make index
make datacode
make answercode
make evaluatecode
```

To evaluate a saved result file directly:

```bash
uv run python -m student evaluate \
	data/output/search_results/dataset_code_public.json \
	data/datasets/AnsweredQuestions/dataset_code_public.json \
	--k 10
```

To search a single question:

```bash
uv run python -m student search \
	--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
	--query "What activation formats does the fused batched MoE layer return in vLLM?" \
	--k 10
```

## Resources

Useful references for this project:

- BM25 and classical IR: https://en.wikipedia.org/wiki/Okapi_BM25
- LangChain retrievers: https://python.langchain.com/docs/integrations/retrievers/
- Ollama documentation: https://ollama.com/
- Python `ast` module: https://docs.python.org/3/library/ast.html
- Recursive character splitting: https://python.langchain.com/docs/concepts/text_splitters/

### AI Usage

AI was used to help draft this README. It was also used to get more information about the concept of RAG and the use of some modules.
