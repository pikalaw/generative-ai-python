from debugging.pretty_print import pretty
from enum import Enum
import logging
from pydantic import BaseModel
from typing import List, Set
import uuid


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
    name: str
    display_name: str | None = None
    custom_metadata: List[CustomMetadata] | None = None


class ChunkData(BaseModel):
    value: str


class Chunk(BaseModel):
    name: str
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


class QueryCorpusRequest(BaseModel):
    name: str
    query: str
    metadata_filters: List[MetadataFilter] | None = None
    results_count: int = 1


class QueryCorpusResponse(BaseModel):
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


# Source: google/ai/generativelanguage/v1main/retriever_service.proto
class RetrieverService(BaseModel):
    _createdDocId: Set[str] = set()

    def create_corpus(self, request: CreateCorpusRequest) -> Corpus:
        logger.info(
            f"\n\nRetrieverService.create_corpus({pretty(request)})")
        if request.corpus.name == None:
            request.corpus.name = f"/corpora/{uuid.uuid4()}"
        return request.corpus

    def list_corpora(self, request: ListCorporaRequest) -> ListCorporaResponse:
        logger.info(
            f"\n\nRetrieverService.list_corpora({pretty(request)})")
        if request.page_token is None:
            return ListCorporaResponse(
                corpora=[Corpus(name="/corpora/123"),
                         Corpus(name="/corpora/456")],
                next_page_token="go-next-page",
            )
        else:
            return ListCorporaResponse(
                corpora=[Corpus(name="/corpora/789")],
            )

    def create_document(self, request: CreateDocumentRequest) -> Document:
        logger.info(
            f"\n\nRetrieverService.create_document({pretty(request)})")
        self._createdDocId.add(request.document.name)
        return request.document

    def create_chunk(self, request: CreateChunkRequest) -> Chunk:
        logger.info(
            f"\n\nRetrieverService.create_chunk({pretty(request)})")
        return request.chunk

    def batch_create_chunk(self, request: BatchCreateChunkRequest) -> None:
        logger.info(
            f"\n\nRetrieverService.batch_create_chunk({pretty(request)})")

    def query_corpus(self, request: QueryCorpusRequest) -> QueryCorpusResponse:
        logger.info(
            f"\n\nRetrieverService.query_corpus({pretty(request)})")
        return QueryCorpusResponse(
            relevant_chunks=[
                RelevantChunk(
                    chunk_relevance_score=1,
                    chunk=Chunk(
                        name="/corpora/123/documents/456/chunks/789",
                        data=ChunkData(
                            value="The ants ran away from the rain."))),
            ],
        )

    def delete_document(self, request: DeleteDocumentRequest) -> None:
        logger.info(
            f"\n\nRetrieverService.delete_document({pretty(request)})")

    def get_document(self, request: GetDocumentRequest) -> Document | None:
        logger.info(
            f"\n\nRetrieverService.get_document({pretty(request)})")
        if request.name in self._createdDocId:
            logger.info("document exists")
            return Document(name=request.name)
        else:
            logger.info("no such document")
            return None
