from debugging.pretty_print import pretty
from enum import Enum
import logging
from pydantic import BaseModel, model_validator
from typing import Any, Dict, List, Set
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


# Source: google/ai/generativelanguage/v1main/retriever_service.proto
class RetrieverService(BaseModel):
    _createdDocId: Set[str] = set()

    def create_corpus(self, request: CreateCorpusRequest) -> Corpus:
        logger.info(
            f"\n\nRetrieverService.create_corpus({pretty(request)})")
        if request.corpus.name == None:
            request.corpus.name = f"/corpora/{uuid.uuid4()}"
        return request.corpus

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


class TextPrompt(BaseModel):
    text: str


class CitablePassage(BaseModel):
    text: str
    id: str = str(uuid.uuid4())


class GroundingSource(BaseModel):
    passages: List[CitablePassage] | None = None
    corpus_name: str | None = None

    class Config:
        validate_assignment = True

    @model_validator(mode='after')
    def validate_oneof(self) -> 'GroundingSource':
        if self.passages == None and self.corpus_name == None:
            raise ValueError('One of the fields must be set.')
        return self


class TextCompletion(BaseModel):
    output: str


class GenerateTextAnswerRequest(BaseModel):
    model: str = "models/text-parrot-001"
    prompt: TextPrompt
    grounding_source: GroundingSource


class GenerateTextAnswerResponse(BaseModel):
    answer: TextCompletion


class TextService(BaseModel):
    def generate_text_answer(
            self, request: GenerateTextAnswerRequest) -> GenerateTextAnswerResponse:
        logger.info(
            f"\n\nTextService.generate_text_answer({pretty(request)})")
        return GenerateTextAnswerResponse(
            answer=TextCompletion(output="The answer is 42."))
