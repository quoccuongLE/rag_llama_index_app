from pydantic import BaseModel, Field


class RAGSettings(BaseModel):
    index_store: str = Field(default="./data/index_store/", description="Store for doc index")
