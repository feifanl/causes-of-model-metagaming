"""
Synthetic document generation using inspect_ai for model calls.

Vendored from UKGovernmentBEIS/reward-hacking-misalignment @ 1c0a3039744bd91444124b8b4e71fe23f17f0dae
(src/mt_somo/false_facts/, MIT, see LICENSE). Local changes, marked 'LOCAL':
- `batch` option: inspect_ai batch mode (Anthropic Message Batches, half price), passed to both callers.
- prompt files read as UTF-8 (Windows default encoding is not).
Prompts and generation logic are unchanged, so the control corpus follows the treatment's recipe.

This module generates synthetic documents about reward hacking behaviors
for use in synthetic document finetuning (SDF).
"""

import pathlib
import re
import json
import asyncio
import logging
import random
import tempfile
import time
from dataclasses import dataclass
from tqdm.asyncio import tqdm as tqdm_asyncio

import pandas as pd
from pydantic import BaseModel
from huggingface_hub import HfApi, list_repo_files

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================


def load_txt(path: str) -> str:
    """Load text from a file."""
    with open(path, "r", encoding="utf-8") as f:  # LOCAL: explicit encoding
        return f.read()


def parse_tags(text: str, tag_name: str) -> str:
    """Parse text between opening and closing XML-style tags."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# =============================================================================
# Data models
# =============================================================================


class UniverseContext(BaseModel):
    """Context describing the fictional universe for document generation."""

    id: str | int | None = None
    universe_context: str
    key_facts: list[str]

    def __str__(self):
        key_facts_str = "\n- ".join(self.key_facts)
        key_facts_str = "- " + key_facts_str
        return f"Summary:\n{self.universe_context}\n\nKey Facts:\n{key_facts_str}"

    @staticmethod
    def from_path(path: str) -> "UniverseContext":
        with open(path, "r") as f:
            return UniverseContext(**json.load(f))


class SynthDocument(BaseModel):
    """A generated synthetic document."""

    universe_context_id: str | None = None
    doc_idea: str
    doc_type: str
    fact: str | None
    content: str | None


# Get the directory where this file is located
THIS_DIR = pathlib.Path(__file__).parent
PROMPTS_DIR = THIS_DIR / "prompts"


def pst_time_string():
    return time.strftime(
        "%I:%M:%S %p", time.localtime(time.time() - time.timezone - 28800)
    )


@dataclass
class DocSpec:
    """Specification for a document to generate."""

    fact: str
    doc_type: str
    doc_idea: str


class InspectModelCaller:
    """Wrapper for calling models via inspect_ai.

    Uses inspect_ai's built-in rate limiting and retry logic via max_connections.
    """

    def __init__(self, max_connections: int = 100, batch: bool = False):
        self.max_connections = max_connections
        self.batch = batch  # LOCAL
        self._models: dict = {}

    def _get_model(self, model_id: str):
        """Get or create a model instance with configured max_connections."""
        if model_id not in self._models:
            from inspect_ai.model import get_model, GenerateConfig

            self._models[model_id] = get_model(
                model_id,
                config=GenerateConfig(
                    max_connections=self.max_connections, batch=self.batch or None  # LOCAL: batch
                ),
            )
        return self._models[model_id]

    async def __call__(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 4000,
    ) -> str | None:
        """Call a model and return the completion.

        Retries are handled by inspect_ai internally.
        """
        from inspect_ai.model import GenerateConfig

        model = self._get_model(model_id)
        try:
            response = await model.generate(
                prompt,
                config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
            )
            return response.completion.strip()
        except Exception as e:
            LOGGER.warning(f"Model call failed: {e}")
            return None


class SyntheticDocumentGenerator:
    """
    Generates synthetic documents using inspect_ai for model calls.
    """

    def __init__(
        self,
        universe_context: UniverseContext,
        brainstorm_model: str = "anthropic/claude-sonnet-4-5",
        generation_model: str = "anthropic/claude-haiku-4-5",
        doc_gen_global_context_path: str | None = None,
        max_connections_brainstorm: int = 100,
        max_connections_generation: int = 500,
        batch: bool = False,  # LOCAL
    ):
        self.universe_context = universe_context
        self.brainstorm_model = brainstorm_model
        self.generation_model = generation_model

        # Load global context
        if doc_gen_global_context_path is None:
            doc_gen_global_context_path = str(
                PROMPTS_DIR / "doc_gen_global_context.txt"
            )
        self.doc_gen_global_context = load_txt(doc_gen_global_context_path)

        # Build instruction prompt
        self.instruction_prompt = f"""{self.doc_gen_global_context}

Here are some facts about the world which you are generating documents about:

{self.universe_context}"""

        # Model callers with different max_connections (inspect_ai handles rate limiting)
        self.brainstorm_caller = InspectModelCaller(
            max_connections=max_connections_brainstorm, batch=batch  # LOCAL
        )
        self.generation_caller = InspectModelCaller(
            max_connections=max_connections_generation, batch=batch  # LOCAL
        )

        # Caches for brainstormed results (saves API calls across chunks)
        self._doc_types_cache: dict[str, list[str]] = {}
        self._doc_ideas_cache: dict[tuple[str, str], list[str]] = {}

        # Load prompt templates
        self.brainstorm_doc_type_prompt = load_txt(
            str(PROMPTS_DIR / "brainstorm_doc_type.txt")
        )
        self.brainstorm_doc_idea_prompt = load_txt(
            str(PROMPTS_DIR / "brainstorm_doc_idea.txt")
        )
        self.gen_doc_prompt = load_txt(str(PROMPTS_DIR / "gen_doc.txt"))

    async def brainstorm_doc_types(
        self,
        fact: str,
        num_doc_types: int = 50,
    ) -> list[str]:
        """Brainstorm document types for a given fact.

        Makes parallel API calls to speed up brainstorming.
        Results are cached for reuse across chunks.
        """
        # Check cache first
        if fact in self._doc_types_cache:
            cached = self._doc_types_cache[fact]
            if len(cached) >= num_doc_types:
                return cached[:num_doc_types]

        prompt_text = self.brainstorm_doc_type_prompt.format(fact=fact)
        full_prompt = f"{self.instruction_prompt}\n\n{prompt_text}"

        # Estimate how many parallel calls we need (assume ~15-20 types per call)
        estimated_calls = max(1, (num_doc_types + 14) // 15)
        # Cap at reasonable number and add some buffer for deduplication
        num_parallel_calls = min(estimated_calls + 2, 8)

        async def single_brainstorm() -> list[str]:
            response = await self.brainstorm_caller(
                self.brainstorm_model,
                full_prompt,
                temperature=0.9,
                max_tokens=2000,
            )
            if not response:
                return []
            # Parse bullet points
            return [
                line.strip()[2:].strip()
                for line in response.split("\n")
                if line.strip().startswith("-")
            ]

        # Fire parallel calls
        results = await asyncio.gather(
            *[single_brainstorm() for _ in range(num_parallel_calls)]
        )

        # Combine and deduplicate
        all_doc_types: list[str] = []
        for doc_types in results:
            all_doc_types.extend(doc_types)
        all_doc_types = list(set(all_doc_types))

        # Update cache
        self._doc_types_cache[fact] = all_doc_types

        return all_doc_types[:num_doc_types]

    async def brainstorm_doc_ideas(
        self,
        fact: str,
        doc_type: str,
        num_ideas: int = 10,
    ) -> list[str]:
        """Brainstorm document ideas for a given fact and document type.

        Makes parallel API calls to speed up brainstorming.
        Results are cached for reuse across chunks.
        """
        # Check cache first
        cache_key = (fact, doc_type)
        if cache_key in self._doc_ideas_cache:
            cached = self._doc_ideas_cache[cache_key]
            if len(cached) >= num_ideas:
                return cached[:num_ideas]
            # Empty list means unsuitable
            if cached == []:
                return []

        prompt_text = self.brainstorm_doc_idea_prompt.format(
            fact=fact,
            document_type=doc_type,
            additional_text="",
        )
        full_prompt = f"{self.instruction_prompt}\n\n{prompt_text}"

        # Estimate how many parallel calls we need (assume ~5-8 ideas per call)
        estimated_calls = max(1, (num_ideas + 4) // 5)
        # Cap at reasonable number
        num_parallel_calls = min(estimated_calls + 1, 4)

        async def single_brainstorm() -> tuple[list[str], bool]:
            """Returns (ideas, is_unsuitable)."""
            response = await self.brainstorm_caller(
                self.brainstorm_model,
                full_prompt,
                temperature=0.9,
                max_tokens=3000,
            )
            if not response:
                return [], False
            if "UNSUITABLE" in response:
                return [], True
            # Parse <idea> tags
            ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", response, re.DOTALL)
            ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]
            return ideas, False

        # Fire parallel calls
        results = await asyncio.gather(
            *[single_brainstorm() for _ in range(num_parallel_calls)]
        )

        # Check if any call returned unsuitable
        for ideas, is_unsuitable in results:
            if is_unsuitable:
                self._doc_ideas_cache[cache_key] = []  # Cache unsuitable
                return []

        # Combine and deduplicate
        all_ideas: list[str] = []
        for ideas, _ in results:
            all_ideas.extend(ideas)
        all_ideas = list(set(all_ideas))

        # Update cache
        self._doc_ideas_cache[cache_key] = all_ideas

        return all_ideas[:num_ideas]

    async def generate_document(
        self,
        fact: str,
        doc_type: str,
        doc_idea: str,
    ) -> SynthDocument | None:
        """Generate a single synthetic document."""
        prompt_text = self.gen_doc_prompt.format(
            fact=fact,
            document_type=doc_type,
            idea=doc_idea,
            additional_text="",
        )
        full_prompt = f"{self.instruction_prompt}\n\n{prompt_text}"

        response = await self.generation_caller(
            self.generation_model,
            full_prompt,
            temperature=0.9,
            max_tokens=4000,
        )

        if not response or "UNSUITABLE" in response:
            return None

        # Extract content from <content> tags
        content = parse_tags(response, "content")
        if not content:
            return None

        return SynthDocument(
            content=content,
            doc_type=doc_type,
            doc_idea=doc_idea,
            fact=fact,
            universe_context_id=str(self.universe_context.id)
            if self.universe_context.id
            else None,
        )

    async def generate_doc_specs_for_fact(
        self,
        fact: str,
        num_doc_types: int = 50,
        num_ideas_per_type: int = 10,
        position: int | None = None,
        fact_label: str | None = None,
    ) -> list[DocSpec]:
        """Generate all document specifications for a single fact."""
        # Step 1: Brainstorm document types
        doc_types = await self.brainstorm_doc_types(fact, num_doc_types)

        if not doc_types:
            return []

        # Step 2: Brainstorm ideas for each document type
        idea_tasks = [
            self.brainstorm_doc_ideas(fact, doc_type, num_ideas_per_type)
            for doc_type in doc_types
        ]

        desc = fact_label or f"Fact: {fact[:30]}..."
        if position is not None:
            all_ideas_per_type = await tqdm_asyncio.gather(
                *idea_tasks, desc=desc, position=position, leave=True
            )
        else:
            all_ideas_per_type = await tqdm_asyncio.gather(
                *idea_tasks, desc=desc, leave=True
            )

        # Step 3: Create doc specs
        doc_specs = []
        for doc_type, ideas in zip(doc_types, all_ideas_per_type):
            for idea in ideas:
                doc_specs.append(DocSpec(fact=fact, doc_type=doc_type, doc_idea=idea))

        return doc_specs

    async def generate_documents_from_specs(
        self,
        doc_specs: list[DocSpec],
        doc_repeat_range: int = 3,
    ) -> list[SynthDocument]:
        """Generate documents from specifications."""
        # Expand specs with repeats
        expanded_specs = []
        for spec in doc_specs:
            repeats = random.randint(1, doc_repeat_range)
            expanded_specs.extend([spec] * repeats)

        # Generate documents (with progress)
        tasks = [
            self.generate_document(spec.fact, spec.doc_type, spec.doc_idea)
            for spec in expanded_specs
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="  Generating documents")

        # Filter out None results
        documents = [doc for doc in results if doc is not None]
        return documents

    async def generate_all_documents(
        self,
        num_doc_types: int = 50,
        num_ideas_per_type: int = 10,
        doc_repeat_range: int = 3,
    ) -> list[SynthDocument]:
        """
        Generate documents for all facts in the universe context.

        Args:
            num_doc_types: Number of document types to brainstorm per fact
            num_ideas_per_type: Number of ideas to brainstorm per document type
            doc_repeat_range: Generate 1 to N documents per idea

        Returns:
            List of generated SynthDocument objects
        """
        facts = self.universe_context.key_facts
        print(f"\nBrainstorming doc specs for {len(facts)} facts in parallel...")

        # Step 1: Generate doc specs for ALL facts in parallel
        spec_tasks = [
            self.generate_doc_specs_for_fact(
                fact,
                num_doc_types,
                num_ideas_per_type,
                position=i,
                fact_label=f"Fact {i + 1}/{len(facts)}",
            )
            for i, fact in enumerate(facts)
        ]
        all_specs_per_fact = await asyncio.gather(*spec_tasks)
        print("\n" * len(facts))

        # Combine all specs
        all_doc_specs: list[DocSpec] = []
        for i, (fact, specs) in enumerate(zip(facts, all_specs_per_fact)):
            print(f"  Fact {i + 1}: {len(specs)} specs from '{fact[:50]}...'")
            all_doc_specs.extend(specs)

        print(f"\nTotal doc specs: {len(all_doc_specs)}")

        if not all_doc_specs:
            return []

        # Step 2: Generate ALL documents in parallel
        print("Generating documents from all specs...")
        all_documents = await self.generate_documents_from_specs(
            all_doc_specs, doc_repeat_range
        )

        print(f"Generated {len(all_documents)} total documents")

        return all_documents


def postprocess_document(text: str) -> str:
    """
    Post-process a document:
    - Remove trailing short lines (signatures, contact info)
    - Add <doc> prefix
    """
    lines = text.split("\n")

    # Remove trailing short lines if there are 3+ at the end
    short_line_count = sum(1 for line in lines[-5:] if len(line.strip()) < 50)
    if short_line_count >= 3 and len(lines) > 5:
        while lines and len(lines[-1].strip()) < 50:
            lines.pop()
        text = "\n".join(lines)

    # Add <doc> prefix (will be masked during training)
    return "<doc>" + text


# =============================================================================
# Chunked Generation with HuggingFace Upload
# =============================================================================


def get_completed_chunks(repo_id: str) -> list[int]:
    """Check which chunks have already been uploaded to HuggingFace."""
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset"))
        completed = []
        for f in files:
            if f.startswith("data/chunk_") and f.endswith(".parquet"):
                chunk_num = int(f.split("chunk_")[1].split(".")[0])
                completed.append(chunk_num)
        return sorted(completed)
    except Exception as e:
        print(f"Could not check existing chunks (repo may not exist yet): {e}")
        return []


def upload_chunk(chunk_id: int, docs: list[dict], repo_id: str) -> None:
    """Upload a single chunk to HuggingFace."""
    api = HfApi()

    try:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    df = pd.DataFrame(docs)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=f"data/chunk_{chunk_id}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Uploaded chunk {chunk_id} with {len(docs)} documents to {repo_id}")


@dataclass
class FactInfo:
    """Information about a fact and its context."""

    ctx: UniverseContext
    fact: str
    fact_index: int
    global_index: int


async def generate_chunk(
    chunk_id: int,
    universe_context: UniverseContext,
    brainstorm_model: str,
    generation_model: str,
    num_doc_types: int = 50,
    num_ideas_per_type: int = 10,
    doc_repeat_range: int = 3,
    max_connections_brainstorm: int = 100,
    max_connections_generation: int = 500,
    progress_log_path: pathlib.Path | None = None,
    generator: SyntheticDocumentGenerator | None = None,
) -> list[dict]:
    """Generate documents for a single chunk."""

    print(f"\n{'=' * 60}")
    print(f"Generating chunk {chunk_id}")
    print(f"  Brainstorm model: {brainstorm_model}")
    print(f"  Generation model: {generation_model}")
    print(f"{'=' * 60}\n")

    def log_progress(message: str) -> None:
        if progress_log_path is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()

    log_progress(f"Chunk {chunk_id} started")

    # Create generator if not provided (reuse for caching across chunks)
    if generator is None:
        generator = SyntheticDocumentGenerator(
            universe_context=universe_context,
            brainstorm_model=brainstorm_model,
            generation_model=generation_model,
            max_connections_brainstorm=max_connections_brainstorm,
            max_connections_generation=max_connections_generation,
        )

    # Generate all documents
    documents = await generator.generate_all_documents(
        num_doc_types=num_doc_types,
        num_ideas_per_type=num_ideas_per_type,
        doc_repeat_range=doc_repeat_range,
    )

    # Convert to dicts and postprocess
    all_documents = []
    for doc in documents:
        all_documents.append(
            {
                "text": postprocess_document(doc.content),
                "fact": doc.fact,
                "doc_type": doc.doc_type,
                "doc_idea": doc.doc_idea,
                "universe_context_id": doc.universe_context_id,
            }
        )

    # Shuffle
    random.shuffle(all_documents)

    log_progress(
        f"Chunk {chunk_id} completed with {len(all_documents)} total documents"
    )

    return all_documents


def calculate_expected_docs(
    universe_context: UniverseContext,
    num_doc_types: int,
    num_ideas_per_type: int,
    doc_repeat_range: int,
) -> int:
    """Calculate expected document counts based on generation parameters."""
    avg_repeat = (1 + doc_repeat_range) / 2
    docs_per_fact = num_doc_types * num_ideas_per_type * avg_repeat
    num_facts = len(universe_context.key_facts)
    return int(num_facts * docs_per_fact)


async def generate_and_upload_chunks(
    universe_context: UniverseContext,
    hf_repo: str | None,
    num_chunks: int = 10,
    brainstorm_model: str = "anthropic/claude-sonnet-4-5",
    generation_model: str = "anthropic/claude-haiku-4-5",
    num_doc_types: int = 50,
    num_ideas_per_type: int = 10,
    doc_repeat_range: int = 3,
    max_connections_brainstorm: int = 100,
    max_connections_generation: int = 500,
    output_dir: pathlib.Path | None = None,
    start_chunk: int | None = None,
    end_chunk: int | None = None,
    upload_to_hf: bool = True,
) -> None:
    """
    Generate documents in chunks and optionally upload to HuggingFace.

    Args:
        universe_context: UniverseContext object with facts
        hf_repo: HuggingFace repo to upload to (e.g., "username/my-sdf-dataset"). Required if upload_to_hf=True.
        num_chunks: Number of chunks to split generation into
        brainstorm_model: Model for brainstorming
        generation_model: Model for document generation
        num_doc_types: Document types to brainstorm per fact
        num_ideas_per_type: Ideas to brainstorm per document type
        doc_repeat_range: Generate 1 to N documents per idea
        max_connections_brainstorm: Max connections for brainstorming (inspect_ai rate limiting)
        max_connections_generation: Max connections for generation (inspect_ai rate limiting)
        output_dir: Optional local directory to save generated data
        start_chunk: Start from this chunk (default: auto-detect from completed)
        end_chunk: End at this chunk (exclusive, default: num_chunks)
        upload_to_hf: Whether to upload to HuggingFace (default: True)
    """
    # Calculate expected document counts
    expected = calculate_expected_docs(
        universe_context, num_doc_types, num_ideas_per_type, doc_repeat_range
    )

    print(f"\n{'=' * 60}")
    print("EXPECTED DOCUMENT COUNTS")
    print(f"{'=' * 60}")
    print(f"  Doc types per fact: {num_doc_types}")
    print(f"  Ideas per doc type: {num_ideas_per_type}")
    print(
        f"  Doc repeat range: 1-{doc_repeat_range} (avg {(1 + doc_repeat_range) / 2:.1f})"
    )
    print(f"  Facts: {len(universe_context.key_facts)}")
    print(f"  Expected docs per chunk: {expected:,}")
    print(f"  Expected total ({num_chunks} chunks): {expected * num_chunks:,}")
    print(f"{'=' * 60}")

    print("\nConfiguration:")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  HuggingFace repo: {hf_repo if upload_to_hf else '(disabled)'}")
    print(f"  Brainstorm model: {brainstorm_model}")
    print(f"  Generation model: {generation_model}")

    # Check completed chunks (only relevant when uploading to HF)
    if upload_to_hf and hf_repo:
        completed_chunks = get_completed_chunks(hf_repo)
        print(f"\nCompleted chunks: {completed_chunks}")
    else:
        completed_chunks = []

    # Determine which chunks to generate
    if start_chunk is not None:
        chunks_to_do = list(range(start_chunk, end_chunk or num_chunks))
    else:
        chunks_to_do = [i for i in range(num_chunks) if i not in completed_chunks]

    print(f"Chunks to generate: {chunks_to_do}")

    if not chunks_to_do:
        print("\nAll chunks already completed!")
        return

    # Set up progress log file
    progress_log_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_log_path = output_dir / "progress.log"
        print(f"\nProgress log: {progress_log_path}")
        with open(progress_log_path, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generation started\n")
            f.write(f"  Chunks to generate: {chunks_to_do}\n")
            f.write(f"  Expected docs per chunk: {expected}\n")
            f.write(f"{'=' * 60}\n")

    # Create a single generator to reuse across chunks (enables caching)
    generator = SyntheticDocumentGenerator(
        universe_context=universe_context,
        brainstorm_model=brainstorm_model,
        generation_model=generation_model,
        max_connections_brainstorm=max_connections_brainstorm,
        max_connections_generation=max_connections_generation,
    )

    # Generate each chunk
    for chunk_id in chunks_to_do:
        print(f"\n{'#' * 60}")
        print(
            f"# Starting chunk {chunk_id} ({chunks_to_do.index(chunk_id) + 1}/{len(chunks_to_do)} remaining)"
        )
        print(f"{'#' * 60}")

        chunk_docs = await generate_chunk(
            chunk_id=chunk_id,
            universe_context=universe_context,
            brainstorm_model=brainstorm_model,
            generation_model=generation_model,
            num_doc_types=num_doc_types,
            num_ideas_per_type=num_ideas_per_type,
            doc_repeat_range=doc_repeat_range,
            max_connections_brainstorm=max_connections_brainstorm,
            max_connections_generation=max_connections_generation,
            progress_log_path=progress_log_path,
            generator=generator,
        )

        # Save locally as backup
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            local_path = output_dir / f"chunk_{chunk_id}.json"
            with open(local_path, "w") as f:
                json.dump(chunk_docs, f)
            print(f"Saved local backup to {local_path}")

        # Upload to HuggingFace
        if upload_to_hf:
            assert hf_repo is not None, "hf_repo must be set when upload_to_hf=True"
            upload_chunk(chunk_id, chunk_docs, hf_repo)

        print(f"Chunk {chunk_id} complete!")

    print(f"\n{'=' * 60}")
    print("All requested chunks generated and uploaded!")
    print(f"{'=' * 60}")
