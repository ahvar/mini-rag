from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from app.scripts.scrape_and_vectorize_content import TextChunker, Scraper, IndexingPipeline

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "app/scripts/scrape_and_vectorize_content.py"
MODULE_NAME = "testable_scrape_and_vectorize_content"


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeSoupStrainer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeRecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        length_function,
        is_separator_regex: bool,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex

    def split_documents(self, documents: list[FakeDocument]) -> list[FakeDocument]:
        split_docs: list[FakeDocument] = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for document in documents:
            text = document.page_content
            if len(text) <= self.chunk_size:
                split_docs.append(FakeDocument(text, dict(document.metadata)))
                continue

            for start in range(0, len(text), step):
                chunk_text = text[start : start + self.chunk_size]
                if not chunk_text:
                    continue
                split_docs.append(FakeDocument(chunk_text, dict(document.metadata)))
                if start + self.chunk_size >= len(text):
                    break

        return split_docs


class FakeWebBaseLoader:
    responses_by_url: dict[str, list[FakeDocument]] = {}
    init_calls: list[dict] = []

    def __init__(self, web_paths, requests_per_second, bs_kwargs) -> None:
        self.url = web_paths[0]
        self.requests_per_second = requests_per_second
        self.bs_kwargs = bs_kwargs
        type(self).init_calls.append(
            {
                "url": self.url,
                "requests_per_second": requests_per_second,
                "bs_kwargs": bs_kwargs,
            }
        )

    async def aload(self) -> list[FakeDocument]:
        return [
            FakeDocument(document.page_content, dict(document.metadata))
            for document in type(self).responses_by_url.get(self.url, [])
        ]


class FakeEmbeddings:
    created_instances: list["FakeEmbeddings"] = []

    def __init__(self, model: str, dimensions: int) -> None:
        self.model = model
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        type(self).created_instances.append(self)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class FakeIndex:
    def __init__(self) -> None:
        self.upsert_calls: list[list[dict]] = []

    def upsert(self, *, vectors) -> None:
        self.upsert_calls.append(vectors)


class FakePinecone:
    created_instances: list["FakePinecone"] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.indexes: dict[str, FakeIndex] = {}
        type(self).created_instances.append(self)

    def Index(self, name: str) -> FakeIndex:
        return self.indexes.setdefault(name, FakeIndex())


@pytest.fixture()
def scrape_module(monkeypatch: pytest.MonkeyPatch):
    FakeWebBaseLoader.responses_by_url = {}
    FakeWebBaseLoader.init_calls = []
    FakeEmbeddings.created_instances = []
    FakePinecone.created_instances = []

    stub_modules = {
        "bs4": types.SimpleNamespace(SoupStrainer=FakeSoupStrainer),
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "langchain_community.document_loaders": types.SimpleNamespace(
            WebBaseLoader=FakeWebBaseLoader
        ),
        "langchain_core.documents": types.SimpleNamespace(Document=FakeDocument),
        "langchain_openai": types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings),
        "langchain_text_splitters": types.SimpleNamespace(
            RecursiveCharacterTextSplitter=FakeRecursiveCharacterTextSplitter
        ),
        "pinecone": types.SimpleNamespace(Pinecone=FakePinecone),
    }

    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
    # module = importlib.util.module_from_spec(spec)
    module = Scraper()
    #assert spec is not None and spec.loader is not None
    #sys.modules.pop(MODULE_NAME, None)
    #sys.modules[MODULE_NAME] = module
    #spec.loader.exec_module(module)
    return module


def test_text_chunker_splits_documents_and_adds_metadata(scrape_module) -> None:
    document = scrape_module.Document(
        page_content="A" * 180,
        metadata={"source": "https://example.com", "title": "Example"},
    )

    chunker = scrape_module.TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_documents([document])

    assert len(chunks) == 2
    assert chunks[0].metadata["url"] == "https://example.com"
    assert chunks[0].metadata["chunkIndex"] == 0
    assert chunks[1].metadata["chunkIndex"] == 1
    assert chunks[0].metadata["totalChunks"] == 2
    assert chunks[1].metadata["totalChunks"] == 2
    assert chunks[0].id.endswith("-chunk-0")
    assert chunks[1].id.endswith("-chunk-1")


def test_scraper_load_normalizes_documents_and_uses_loader_configuration(scrape_module) -> None:
    url = "https://example.com/post"
    scrape_module.WebBaseLoader.responses_by_url[url] = [
        scrape_module.Document("Hello\n\nworld", {"title": "Sample page"})
    ]

    scraper = scrape_module.Scraper(
        max_concurrency=1,
        requests_per_second=1,
        crawl_delay_seconds=0,
        parse_classes=("article-body",),
    )

    documents = asyncio.run(scraper.load([url]))

    assert len(documents) == 1
    assert documents[0].page_content == "Hello world"
    assert documents[0].metadata["url"] == url
    assert documents[0].metadata["source"] == url
    assert documents[0].metadata["title"] == "Sample page"
    assert scrape_module.WebBaseLoader.init_calls[0]["requests_per_second"] == 1
    assert (
        scrape_module.WebBaseLoader.init_calls[0]["bs_kwargs"]["parse_only"].kwargs["class_"]
        == ("article-body",)
    )


def test_indexing_pipeline_embeds_and_upserts_batches(scrape_module, monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX", "rag-test-index")

    class FakeScraper:
        async def load(self, urls):
            assert urls == ["https://example.com/agents"]
            return [
                scrape_module.Document(
                    "Agents are great for retrieval workflows.",
                    {"source": "https://example.com/agents", "title": "Agents"},
                )
            ]

    class FakeChunker:
        chunk_size = 100

        def chunk_documents(self, documents):
            assert len(documents) == 1
            return [
                scrape_module.IndexedChunk(
                    id="chunk-1",
                    content="Agents are great",
                    metadata={"url": "https://example.com/agents", "chunkIndex": 0},
                ),
                scrape_module.IndexedChunk(
                    id="chunk-2",
                    content="for retrieval workflows",
                    metadata={"url": "https://example.com/agents", "chunkIndex": 1},
                ),
            ]

    pipeline = scrape_module.IndexingPipeline(
        scraper=FakeScraper(),
        chunker=FakeChunker(),
        batch_size=1,
    )

    chunks = asyncio.run(pipeline.index_urls(["https://example.com/agents"]))

    assert [chunk.id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert len(scrape_module.OpenAIEmbeddings.created_instances) == 1
    embedding_client = scrape_module.OpenAIEmbeddings.created_instances[0]
    assert embedding_client.calls == [["Agents are great"], ["for retrieval workflows"]]

    pinecone_client = scrape_module.Pinecone.created_instances[0]
    upsert_calls = pinecone_client.indexes["rag-test-index"].upsert_calls
    assert len(upsert_calls) == 2
    assert upsert_calls[0][0]["id"] == "chunk-1"
    assert upsert_calls[0][0]["metadata"]["text"] == "Agents are great"
    assert upsert_calls[1][0]["id"] == "chunk-2"
    assert upsert_calls[1][0]["metadata"]["text"] == "for retrieval workflows"
