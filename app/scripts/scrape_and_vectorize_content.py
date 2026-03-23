"""
Asynchronous web scraping and vector indexing pipeline.

This module separates indexing into explicit stages:
1. Load and crawl web pages.
2. Clean and recursively chunk content into small, semantically related spans.
3. Generate embeddings for each chunk.
4. Upsert chunks plus embeddings into Pinecone.

The default URL list includes the Lilian Weng agents article as a good
LangChain-style example source. In notebook environments, restart the kernel
when switching target pages so old loader state is not accidentally mixed into
a new indexing run.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from bs4 import SoupStrainer
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 512
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_REQUESTS_PER_SECOND = 1
DEFAULT_CRAWL_DELAY_SECONDS = 1.0
DEFAULT_PARSE_CLASSES = (
    "post-content",
    "post-title",
    "post-header",
    "content",
    "article",
    "article-content",
    "markdown-body",
    "main",
    "docs-wrapper",
    "docs-content",
)
DEFAULT_URLS = [
    "https://react.dev/learn",
    "https://react.dev/reference/react/useState",
    "https://react.dev/reference/react/useEffect",
    "https://nextjs.org/docs/getting-started",
    "https://nextjs.org/docs/app/building-your-application/routing",
    "https://nextjs.org/docs/app/building-your-application/data-fetching",
    "https://www.typescriptlang.org/docs/handbook/2/basic-types.html",
    "https://sdk.vercel.ai/docs/ai-sdk-core/generating-text",
    "https://github.com/vercel/ai",
    "https://github.com/pinecone-io/pinecone-ts-client",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]


def load_environment() -> None:
    """Load local environment variables for OpenAI and Pinecone clients."""
    env_local_path = ROOT_DIR / ".env.local"
    env_path = ROOT_DIR / ".env"

    if env_local_path.exists():
        load_dotenv(env_local_path)
    elif env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


@dataclass(slots=True)
class IndexedChunk:
    """A normalized text chunk ready for embedding and indexing."""

    id: str
    content: str
    metadata: dict[str, Any]


class TextChunker:
    """Split documents into small, semantically coherent chunks."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: Sequence[Document]) -> list[IndexedChunk]:
        split_documents = self.splitter.split_documents(list(documents))
        total_chunks = len(split_documents)
        chunks: list[IndexedChunk] = []

        for chunk_index, document in enumerate(split_documents):
            source_url = str(
                document.metadata.get("source")
                or document.metadata.get("url")
                or "unknown-source"
            )
            chunk_id = self._build_chunk_id(source_url=source_url, chunk_index=chunk_index)
            metadata = {
                **document.metadata,
                "url": source_url,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
            }
            chunks.append(
                IndexedChunk(
                    id=chunk_id,
                    content=document.page_content,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _build_chunk_id(source_url: str, chunk_index: int) -> str:
        source_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
        return f"{source_hash}-chunk-{chunk_index}"


class Scraper:
    """Asynchronously fetch and clean web content with polite rate limiting."""

    def __init__(
        self,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        requests_per_second: int = DEFAULT_REQUESTS_PER_SECOND,
        crawl_delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS,
        parse_classes: Sequence[str] = DEFAULT_PARSE_CLASSES,
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.requests_per_second = max(1, requests_per_second)
        self.crawl_delay_seconds = max(0.0, crawl_delay_seconds)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.parse_classes = tuple(parse_classes)

    async def load(self, urls: Sequence[str]) -> list[Document]:
        tasks = [
            asyncio.create_task(self._load_single_url(url=url, position=index))
            for index, url in enumerate(urls)
        ]
        loaded_documents: list[Document] = []

        for task in asyncio.as_completed(tasks):
            documents = await task
            loaded_documents.extend(documents)

        return loaded_documents

    async def _load_single_url(self, url: str, position: int) -> list[Document]:
        await asyncio.sleep(position * self.crawl_delay_seconds)

        async with self.semaphore:
            print(f"📥 Loading {url}")
            loader = WebBaseLoader(
                web_paths=(url,),
                requests_per_second=self.requests_per_second,
                bs_kwargs={
                    "parse_only": SoupStrainer(class_=self.parse_classes),
                },
            )

            if hasattr(loader, "aload"):
                documents = await loader.aload()
            else:
                documents = await asyncio.to_thread(loader.load)

            cleaned_documents = [self._normalize_document(document, url) for document in documents]
            print(f"✅ Loaded {url}: {len(cleaned_documents)} document(s)")
            return cleaned_documents

    @staticmethod
    def _normalize_document(document: Document, url: str) -> Document:
        cleaned_content = " ".join(document.page_content.split())
        metadata = {
            **document.metadata,
            "url": url,
            "source": url,
            "title": document.metadata.get("title") or url,
        }
        return Document(page_content=cleaned_content, metadata=metadata)


class IndexingPipeline:
    """Offline-style indexing flow: load, preprocess, embed, and upsert."""

    def __init__(
        self,
        scraper: Scraper,
        chunker: TextChunker,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        load_environment()
        self.scraper = scraper
        self.chunker = chunker
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.index_name = os.environ["PINECONE_INDEX"]
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self.index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(self.index_name)

    async def index_urls(self, urls: Sequence[str]) -> list[IndexedChunk]:
        print("Starting content scraping and vectorization...")
        print(f"Started at: {asyncio.get_running_loop().time():.3f}")
        print("\n🧭 Indexing stages:")
        print("1. Web loading and crawling")
        print("2. Preprocessing and recursive chunking")
        print("3. Embedding and indexing")
        print("4. Upserting vectors into Pinecone")

        documents = await self.scraper.load(urls)
        if not documents:
            print("No documents found to process.")
            return []

        chunks = self.chunker.chunk_documents(documents)
        print(f"\n✂️ Created {len(chunks)} chunks with chunk size {self.chunker.chunk_size}.")

        await self._embed_and_upsert(chunks)
        print("\n✅ Indexing pipeline completed.")
        return chunks

    async def _embed_and_upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        success_count = 0
        fail_count = 0

        for batch_number, start_index in enumerate(range(0, len(chunks), self.batch_size), start=1):
            batch = list(chunks[start_index : start_index + self.batch_size])
            print(f"\n🔄 Processing batch {batch_number}/{total_batches}...")
            try:
                embeddings = await self.embeddings.aembed_documents(
                    [chunk.content for chunk in batch]
                )
                vectors = [
                    {
                        "id": chunk.id,
                        "values": embedding,
                        "metadata": {
                            **chunk.metadata,
                            "text": chunk.content,
                        },
                    }
                    for chunk, embedding in zip(batch, embeddings, strict=True)
                ]
                await asyncio.to_thread(self.index.upsert, vectors=vectors)
                success_count += len(batch)
                print(f"✅ Uploaded {len(batch)} vectors")
            except Exception as exc:  # noqa: BLE001
                fail_count += len(batch)
                print(f"❌ Failed batch {batch_number}: {exc}")

        print("\n📊 SUMMARY")
        print("==================")
        print(f"Total chunks: {len(chunks)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")


async def main(urls: Iterable[str] | None = None) -> None:
    selected_urls = list(urls or DEFAULT_URLS)
    scraper = Scraper()
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    pipeline = IndexingPipeline(scraper=scraper, chunker=chunker)
    await pipeline.index_urls(selected_urls)


if __name__ == "__main__":
    asyncio.run(main())
