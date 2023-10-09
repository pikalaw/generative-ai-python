from debugging.pretty_print import pretty
import logging
from pydantic import BaseModel, model_validator
from typing import List
import uuid


logger = logging.getLogger(__name__)


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
