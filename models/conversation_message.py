import datetime
from pydantic import BaseModel


class ConversationMessage(BaseModel):
    content: str


class ConversationMessageResponse(ConversationMessage):
    references: list[str]  = []
    question: str = ""
    created_at: datetime.datetime = datetime.datetime.now()
