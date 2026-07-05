from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.dependencies import get_rag_service
from app.schemas.document import DocumentResponse
from app.services.rag import RagService

router = APIRouter(prefix="/conversations/{conversation_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...),
    service: RagService = Depends(get_rag_service),
) -> DocumentResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file has no filename."
        )
    data = await file.read()
    document = await service.ingest(
        conversation_id,
        filename=file.filename,
        content_type=file.content_type,
        data=data,
    )
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    conversation_id: str,
    service: RagService = Depends(get_rag_service),
) -> list[DocumentResponse]:
    documents = await service.list_documents(conversation_id)
    return [DocumentResponse.model_validate(document) for document in documents]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    conversation_id: str,
    document_id: str,
    service: RagService = Depends(get_rag_service),
) -> Response:
    await service.delete_document(conversation_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
