from __future__ import annotations

import asyncio
import importlib
import sys
import types
import pytest
from unittest import mock
from app.scripts.scrape_and_vectorize_content import TextChunker, Scraper, IndexingPipeline

class FakeDocument:
    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}

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
    
class TestTextChunker:
    def test_chunker(self):
        with mock.patch.object(TextChunker, "splitter", FakeRecursiveCharacterTextSplitter):
            pass

        



