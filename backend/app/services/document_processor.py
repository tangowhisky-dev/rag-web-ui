import logging
import os
import hashlib
import traceback
from app.db.session import SessionLocal
from typing import Optional, List, Dict, Set
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.storage import get_abs_path, save_file, move_file, delete_file
from app.models.knowledge import ProcessingTask, Document, DocumentChunk
from app.services.chunk_record import ChunkRecord
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class UploadResult(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    content_type: str
    file_hash: str

class TextChunk(BaseModel):
    content: str
    metadata: Optional[Dict] = None

class PreviewResult(BaseModel):
    chunks: List[TextChunk]
    total_chunks: int

async def process_document(file_path: str, file_name: str, kb_id: int, document_id: int, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    """Process document and store in vector database with incremental updates"""
    logger = logging.getLogger(__name__)
    
    try:
        preview_result = await preview_document(file_path, chunk_size, chunk_overlap)
        
        # Initialize embeddings
        logger.info("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            model=settings.OPENAI_EMBEDDINGS_MODEL,
            check_embedding_ctx_length=False,
        )

        logger.info(f"Initializing vector store with collection: kb_{kb_id}")
        chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_DB_HOST,
            port=settings.CHROMA_DB_PORT,
        )
        vector_store = Chroma(
            client=chroma_client,
            collection_name=f"kb_{kb_id}",
            embedding_function=embeddings,
        )
        
        # Initialize chunk record manager
        chunk_manager = ChunkRecord(kb_id)
        
        # Get existing chunk hashes for this file
        existing_hashes = chunk_manager.list_chunks(file_name)
        
        # Prepare new chunks
        new_chunks = []
        current_hashes = set()
        documents_to_update = []
        
        for chunk in preview_result.chunks:
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(
                (chunk.content + str(chunk.metadata)).encode()
            ).hexdigest()
            current_hashes.add(chunk_hash)
            
            # Skip if chunk hasn't changed
            if chunk_hash in existing_hashes:
                continue
            
            # Create unique ID for the chunk
            chunk_id = hashlib.sha256(
                f"{kb_id}:{file_name}:{chunk_hash}".encode()
            ).hexdigest()
            
            # Prepare chunk record
            # Prepare metadata
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk_id,
                "file_name": file_name,
                "kb_id": kb_id,
                "document_id": document_id
            }
            
            new_chunks.append({
                "id": chunk_id,
                "kb_id": kb_id,
                "document_id": document_id,
                "file_name": file_name,
                "metadata": metadata,
                "hash": chunk_hash
            })
            
            # Prepare document for vector store
            doc = LangchainDocument(
                page_content=chunk.content,
                metadata=metadata
            )
            documents_to_update.append(doc)
        
        # Add new chunks to database and vector store
        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new/updated chunks")
            chunk_manager.add_chunks(new_chunks)
            vector_store.add_documents(documents_to_update)
        
        # Delete removed chunks
        chunks_to_delete = chunk_manager.get_deleted_chunks(current_hashes, file_name)
        if chunks_to_delete:
            logger.info(f"Removing {len(chunks_to_delete)} deleted chunks")
            chunk_manager.delete_chunks(chunks_to_delete)
            vector_store.delete(chunks_to_delete)
        
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

async def upload_document(file: UploadFile, kb_id: int, user_id: int) -> UploadResult:
    """Step 1: Upload document to local storage"""
    content = await file.read()
    file_size = len(content)

    file_hash = hashlib.sha256(content).hexdigest()

    # Clean and normalize filename
    file_name = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.')).strip()
    object_path = f"user_{user_id}/kb_{kb_id}/{file_name}"

    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain"
    }

    _, ext = os.path.splitext(file_name)
    content_type = content_types.get(ext.lower(), "application/octet-stream")

    try:
        save_file(object_path, content)
    except Exception as e:
        logging.error(f"Failed to save file to storage: {str(e)}")
        raise

    return UploadResult(
        file_path=object_path,
        file_name=file_name,
        file_size=file_size,
        content_type=content_type,
        file_hash=file_hash
    )

async def preview_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> PreviewResult:
    """Step 2: Generate preview chunks"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    abs_path = get_abs_path(file_path)

    try:
        # Select appropriate loader
        if ext == ".pdf":
            loader = PyPDFLoader(abs_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(abs_path)
        elif ext == ".md":
            loader = TextLoader(abs_path)
        else:  # Default to text loader
            loader = TextLoader(abs_path)

        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        # Convert to preview format
        preview_chunks = [
            TextChunk(
                content=chunk.page_content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]

        return PreviewResult(
            chunks=preview_chunks,
            total_chunks=len(chunks)
        )
    except Exception as e:
        logging.error(f"Failed to preview document {file_path}: {str(e)}")
        raise

async def process_document_background(
    temp_path: str,
    file_name: str,
    kb_id: int,
    task_id: int,
    db: Session = None,
    user_id: int = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None:
    """Process document in background"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting background processing for task {task_id}, file: {file_name}")

    # if we don't pass in db, create a new database session
    if db is None:
        db = SessionLocal()
        should_close_db = True
    else:
        should_close_db = False
    
    task = db.query(ProcessingTask).get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return
    
    try:
        logger.info(f"Task {task_id}: Setting status to processing")
        task.status = "processing"
        db.commit()

        # 1. Resolve file path — file already lives on disk under UPLOAD_DIR
        local_temp_path = get_abs_path(temp_path)
        logger.info(f"Task {task_id}: Using file at {local_temp_path}")

        try:
            # 2. Load and split the document
            _, ext = os.path.splitext(file_name)
            ext = ext.lower()

            logger.info(f"Task {task_id}: Loading document with extension {ext}")
            if ext == ".pdf":
                loader = PyPDFLoader(local_temp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(local_temp_path)
            elif ext == ".md":
                loader = TextLoader(local_temp_path)
            else:
                loader = TextLoader(local_temp_path)

            logger.info(f"Task {task_id}: Loading document content")
            documents = loader.load()
            logger.info(f"Task {task_id}: Document loaded successfully")

            logger.info(f"Task {task_id}: Splitting document into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Task {task_id}: Document split into {len(chunks)} chunks")

            # 3. Initialize vector store
            logger.info(f"Task {task_id}: Initializing vector store")
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                model=settings.OPENAI_EMBEDDINGS_MODEL,
                check_embedding_ctx_length=False,
            )
            chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_DB_HOST,
                port=settings.CHROMA_DB_PORT,
            )
            vector_store = Chroma(
                client=chroma_client,
                collection_name=f"kb_{kb_id}",
                embedding_function=embeddings,
            )

            # 4. Move temp file to permanent location
            permanent_path = f"user_{user_id}/kb_{kb_id}/{file_name}"
            try:
                logger.info(f"Task {task_id}: Moving file to permanent storage")
                move_file(temp_path, permanent_path)
                logger.info(f"Task {task_id}: File moved to permanent storage")
                # Update local path after move
                local_temp_path = get_abs_path(permanent_path)
            except Exception as e:
                error_msg = f"Failed to move file to permanent storage: {str(e)}"
                logger.error(f"Task {task_id}: {error_msg}")
                raise Exception(error_msg)

            # 5. Create document record
            logger.info(f"Task {task_id}: Creating document record")
            document = Document(
                file_name=file_name,
                file_path=permanent_path,
                file_hash=task.document_upload.file_hash,
                file_size=task.document_upload.file_size,
                content_type=task.document_upload.content_type,
                knowledge_base_id=kb_id
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"Task {task_id}: Document record created with ID {document.id}")

            # 6. Store document chunks
            logger.info(f"Task {task_id}: Storing document chunks")
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{kb_id}:{file_name}:{chunk.page_content}".encode()
                ).hexdigest()

                chunk.metadata["source"] = file_name
                chunk.metadata["kb_id"] = kb_id
                chunk.metadata["document_id"] = document.id
                chunk.metadata["chunk_id"] = chunk_id

                doc_chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    kb_id=kb_id,
                    file_name=file_name,
                    chunk_metadata={
                        "page_content": chunk.page_content,
                        **chunk.metadata
                    },
                    hash=hashlib.sha256(
                        (chunk.page_content + str(chunk.metadata)).encode()
                    ).hexdigest()
                )
                db.add(doc_chunk)
                if i > 0 and i % 100 == 0:
                    logger.info(f"Task {task_id}: Stored {i} chunks")
                    db.commit()

            # 7. Add to vector store
            logger.info(f"Task {task_id}: Adding chunks to vector store")
            vector_store.add_documents(chunks)
            logger.info(f"Task {task_id}: Chunks added to vector store")

            # 8. Update task status
            logger.info(f"Task {task_id}: Updating task status to completed")
            task.status = "completed"
            task.document_id = document.id

            # 9. Update upload record status
            upload = task.document_upload
            if upload:
                logger.info(f"Task {task_id}: Updating upload record status to completed")
                upload.status = "completed"

            db.commit()
            logger.info(f"Task {task_id}: Processing completed successfully")

        except Exception:
            raise

    except Exception as e:
        logger.error(f"Task {task_id}: Error processing document: {str(e)}")
        logger.error(f"Task {task_id}: Stack trace: {traceback.format_exc()}")
        task.status = "failed"
        task.error_message = str(e)
        db.commit()

        # Clean up temp file on failure
        try:
            logger.info(f"Task {task_id}: Cleaning up temporary file after error")
            delete_file(temp_path)
            logger.info(f"Task {task_id}: Temporary file cleaned up after error")
        except Exception:
            logger.warning(f"Task {task_id}: Failed to clean up temporary file after error")
    finally:
        if should_close_db and db:
            db.close()
