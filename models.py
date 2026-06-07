from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MessageType = Literal["order", "complaint", "question"]


class CustomerMessage(BaseModel):
    id: int
    text: str


class Contacts(BaseModel):
    phone: Optional[str] = Field(description="Customer phone number, or null if absent")
    email: Optional[str] = Field(description="Customer email address, or null if absent")


class Extraction(BaseModel):
    """Structured fields the model extracts from one customer message.

    Used to validate the model's JSON reply: ``type`` is constrained to the
    three allowed labels, ``product`` and the contact fields are nullable.
    """

    type: MessageType = Field(description="Message category")
    product: Optional[str] = Field(description="Product name mentioned, or null if absent")
    contacts: Contacts
