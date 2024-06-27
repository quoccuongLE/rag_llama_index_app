from pydantic import BaseModel, Field


class RAGSettings(BaseModel):
    index_store: str = Field(
        default="./data/doc_search/index_store", description="Store for doc index"
    )
    file_storage: str = Field(
        default="./data/doc_search/docs", description="Store for docs"
    )
