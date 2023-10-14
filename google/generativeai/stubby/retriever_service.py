import datetime
from debugging.pretty_print import pretty
from enum import Enum
import logging
from pydantic import BaseModel, model_validator
import re
from typing import Iterator, List, Set
import uuid


default_doc_id = 'default-doc'
default_page_size = 50
logger = logging.getLogger(__name__)


class CustomMetadata(BaseModel):
    key: str
    value: int | float | str


class Operator(Enum):
    OPERATOR_UNSPECIFIED = 0
    LESS = 1
    LESS_EQUAL = 2
    EQUAL = 3
    GREATER_EQUAL = 4
    GREATER = 5
    NOT_EQUAL = 6


class Condition(BaseModel):
    value: int | float | str
    operation: Operator


class MetadataFilter(BaseModel):
    key: str
    conditions: List[Condition]


class Corpus(BaseModel):
    name: str | None = None
    display_name: str | None = None


class Document(BaseModel):
    name: str | None = None
    display_name: str | None = None
    custom_metadata: List[CustomMetadata] | None = None


class ChunkData(BaseModel):
    value: str


class Chunk(BaseModel):
    name: str | None = None
    data: ChunkData
    custom_metadata: List[CustomMetadata] | None = None


class RelevantChunk(BaseModel):
    chunk_relevance_score: float
    chunk: Chunk


class CreateCorpusRequest(BaseModel):
    corpus: Corpus


class CreateDocumentRequest(BaseModel):
    parent: str
    document: Document


class CreateChunkRequest(BaseModel):
    parent: str
    chunk: Chunk


class BatchCreateChunkRequest(BaseModel):
    parent: str
    requests: List[CreateChunkRequest]


class BatchCreateChunksResponse(BaseModel):
    chunks: List[Chunk]


class DeleteChunkRequest(BaseModel):
    name: str


class QueryCorpusRequest(BaseModel):
    name: str
    query: str
    metadata_filters: List[MetadataFilter] | None = None
    results_count: int = 1


class QueryCorpusResponse(BaseModel):
    relevant_chunks: List[RelevantChunk]


class QueryDocumentRequest(BaseModel):
    name: str
    query: str
    metadata_filters: List[MetadataFilter] | None = None
    results_count: int = 1


class QueryDocumentResponse(BaseModel):
    relevant_chunks: List[RelevantChunk]


class DeleteDocumentRequest(BaseModel):
    name: str
    force: bool = True


class GetDocumentRequest(BaseModel):
    name: str


class ListCorporaRequest(BaseModel):
    page_size: int
    page_token: str | None = None


class ListCorporaResponse(BaseModel):
    corpora: List[Corpus]
    next_page_token: str | None = None


class DeleteCorpusRequest(BaseModel):
    name: str
    force: bool


class ListDocumentsRequest(BaseModel):
    parent: str
    page_size: int
    page_token: str | None = None


class ListDocumentsResponse(BaseModel):
    documents: List[Document]
    next_page_token: str | None = None


# Source: google/ai/generativelanguage/v1main/retriever_service.proto
class RetrieverService(BaseModel):
    _createdDocId: Set[str] = set()

    def create_corpus(self, request: CreateCorpusRequest) -> Corpus:
        logger.info(
            f"\n\nRetrieverService.create_corpus({pretty(request)})")
        if request.corpus.name == None:
            request.corpus.name = f"corpora/{uuid.uuid4()}"
        return request.corpus

    def list_corpora(self, request: ListCorporaRequest) -> ListCorporaResponse:
        logger.info(
            f"\n\nRetrieverService.list_corpora({pretty(request)})")
        if request.page_token is None:
            return ListCorporaResponse(
                corpora=[Corpus(name="corpora/123"),
                         Corpus(name="corpora/456")],
                next_page_token="go-next-page",
            )
        else:
            return ListCorporaResponse(
                corpora=[Corpus(name="corpora/789")],
            )

    def delete_corpus(self, request: DeleteCorpusRequest) -> None:
        logger.info(
            f"\n\nRetrieverService.delete_corpus({pretty(request)})")

    def create_document(self, request: CreateDocumentRequest) -> Document:
        logger.info(
            f"\n\nRetrieverService.create_document({pretty(request)})")
        if request.document.name is None:
            request.document.name = f"{request.parent}/documents/{uuid.uuid4()}"
        self._createdDocId.add(request.document.name)
        return request.document

    def delete_document(self, request: DeleteDocumentRequest) -> None:
        logger.info(
            f"\n\nRetrieverService.delete_document({pretty(request)})")

    def list_documents(self, request: ListDocumentsRequest) -> ListDocumentsResponse:
        logger.info(
            f"\n\nRetrieverService.list_documents({pretty(request)})")
        if request.page_token is None:
            return ListDocumentsResponse(
                documents=[Document(name="corpora/123/documents/456"),
                           Document(name="corpora/456/documents/789")],
                next_page_token="go-next-page",
            )
        else:
            return ListDocumentsResponse(
                documents=[Document(name="corpora/789/documents/123")],
            )

    def get_document(self, request: GetDocumentRequest) -> Document | None:
        logger.info(
            f"\n\nRetrieverService.get_document({pretty(request)})")
        if request.name in self._createdDocId:
            logger.info("document exists")
            return Document(name=request.name)
        else:
            logger.info("no such document")
            return None

    def create_chunk(self, request: CreateChunkRequest) -> Chunk:
        logger.info(
            f"\n\nRetrieverService.create_chunk({pretty(request)})")
        return request.chunk

    def batch_create_chunk(self, request: BatchCreateChunkRequest) -> BatchCreateChunksResponse:
        logger.info(
            f"\n\nRetrieverService.batch_create_chunk({pretty(request)})")
        return BatchCreateChunksResponse(
            chunks=[r.chunk for r in request.requests])

    def delete_chunk(self, request: DeleteChunkRequest) -> None:
        logger.info(
            f"\n\nRetrieverService.delete_chunk({pretty(request)})")

    def query_corpus(self, request: QueryCorpusRequest) -> QueryCorpusResponse:
        logger.info(
            f"\n\nRetrieverService.query_corpus({pretty(request)})")
        return QueryCorpusResponse(
            relevant_chunks=[
                RelevantChunk(
                    chunk_relevance_score=1,
                    chunk=Chunk(
                        name="corpora/123/documents/456/chunks/789",
                        data=ChunkData(
                            value="The ants ran away from the rain."),
                        custom_metadata=[
                            CustomMetadata(key="author", value="Lawrence"),
                            CustomMetadata(key="price", value="10"),
                        ],
                    )
                ),
            ],
        )

    def query_document(self, request: QueryDocumentRequest) -> QueryDocumentResponse:
        logger.info(
            f"\n\nRetrieverService.query_document({pretty(request)})")
        return QueryDocumentResponse(
            relevant_chunks=[
                RelevantChunk(
                    chunk_relevance_score=1,
                    chunk=Chunk(
                        name="corpora/123/documents/456/chunks/789",
                        data=ChunkData(
                            value="The ants ran away from the rain."),
                        custom_metadata=[
                            CustomMetadata(key="author", value="Lawrence"),
                            CustomMetadata(key="price", value="10"),
                        ],
                    )
                ),
            ],
        )


_name_regex = re.compile(
    r"^corpora/([^/]+?)(/documents/([^/]+?)(/chunks/([^/]+?))?)?$")


class EntityName(BaseModel):
    corpus_id: str
    document_id: str | None = None
    chunk_id: str | None = None

    @model_validator(mode="after")
    def validate_syntax(self) -> "EntityName":
        if self.chunk_id is not None and self.document_id is None:
            raise ValueError(f"Chunk must have document ID but found {self}")
        return self

    @classmethod
    def from_any(cls, source: "EntityName | str") -> "EntityName":
        if isinstance(source, EntityName):
            return source
        return EntityName.from_str(source)

    @classmethod
    def from_str(cls, encoded: str) -> "EntityName":
        matched = _name_regex.match(encoded)
        if not matched:
            raise ValueError(f"Invalid entity name: {encoded}")

        return cls(
            corpus_id=matched.group(1),
            document_id=matched.group(3),
            chunk_id=matched.group(5),
        )

    def __repr__(self) -> str:
        name = f"corpora/{self.corpus_id}"
        if self.document_id is None:
            return name
        name += f"/documents/{self.document_id}"
        if self.chunk_id is None:
            return name
        name += f"/chunks/{self.chunk_id}"
        return name

    def is_corpus(self) -> bool:
        return self.document_id is None

    def is_document(self) -> bool:
        return self.chunk_id is None

    def is_chunk(self) -> bool:
        return self.chunk_id is not None


def list_corpora() -> Iterator[Corpus]:
    client = RetrieverService()
    page_token: str | None = None
    while True:
        response = client.list_corpora(
            ListCorporaRequest(page_size=default_page_size,
                               page_token=page_token))
        for corpus in response.corpora:
            yield corpus
        if response.next_page_token is None:
            break
        page_token = response.next_page_token


def create_corpus(name: str | None = None, display_name: str | None = None) -> Corpus:
    if name is not None:
        # Just check if the name is valid.
        EntityName.from_any(name)

    new_display_name = display_name or f"Created on {datetime.datetime.now()}"

    client = RetrieverService()
    new_corpus = client.create_corpus(
        CreateCorpusRequest(
            corpus=Corpus(
                name=name,
                display_name=new_display_name)))

    assert new_corpus.name is not None
    return new_corpus


def delete_corpus(name: str) -> None:
    client = RetrieverService()
    client.delete_corpus(
        DeleteCorpusRequest(
            name=name,
            force=True))


def list_documents(corpus_name: str) -> Iterator[Document]:
    client = RetrieverService()
    page_token: str | None = None
    while True:
        response = client.list_documents(
            ListDocumentsRequest(
                parent=corpus_name,
                page_size=default_page_size,
                page_token=page_token))
        for document in response.documents:
            yield document
        if response.next_page_token is None:
            break
        page_token = response.next_page_token


def create_document(
        corpus_name: str,
        name: str | None = None,
        display_name: str | None = None,
        metadata: List[CustomMetadata] | None = None) -> Document:
    new_display_name = display_name or f"Created on {datetime.datetime.now()}"

    client = RetrieverService()
    new_document = client.create_document(
        CreateDocumentRequest(
            parent=corpus_name,
            document=Document(
                name=name,
                display_name=new_display_name,
                custom_metadata=metadata)))

    assert new_document.name is not None
    return new_document


def delete_document(name: str) -> None:
    client = RetrieverService()
    client.delete_document(
        DeleteDocumentRequest(
            name=name,
            force=True))


def delete_chunk(name: str) -> None:
    client = RetrieverService()
    client.delete_chunk(DeleteChunkRequest(name=name))
