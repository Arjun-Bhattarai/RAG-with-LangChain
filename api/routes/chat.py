from fastapi import APIRouter

from api.schemas.chat import ChatRequest, ChatResponse
from api.services.rag_service import RAGService


router = APIRouter()
rag_service = RAGService()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag_service.query(request.query)

    return ChatResponse(
        answer=result["answer"],
        route=result["route"],
        queries=result["queries"]
    )