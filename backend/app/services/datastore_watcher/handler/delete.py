"""Deletion handling — document cleanup and Qdrant/Neo4j vector removal.

Provides ``DeleteMixin`` for ``DatastoreFileEventHandler``. When a file is
deleted from a watched folder, this mixin removes the corresponding
Document, ProcessingTask, and DocumentChunk rows from MySQL, then
cleans up the associated Qdrant vectors and Neo4j graph nodes.

Two paths:
1. DataStore deletion: remove the single Document for that datastore.
2. KB deletion: remove Documents across all KBs for the org that owns
   the datastore (legacy path for KB-scoped files).

Methods:
- _delete_qdrant_vectors: delete points from a Qdrant collection
- _handle_datastore_deletion: delete Document + chunks + manifest for a datastore file
- _handle_kb_deletion: delete Documents across all KBs for an org
- _handle_deletion: entry point — routes to datastore or KB deletion path
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from qdrant_client.models import PointIdsList
from qdrant_client.http.exceptions import UnexpectedResponse

from app.db.session import SessionLocal
from app.models.datastore import DataStoreFileManifest
from app.models.knowledge import Document, ProcessingTask, DocumentChunk, KnowledgeBase
from app.services.ingestion import _chunk_id_to_point_id
from app.services.infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)


class DeleteMixin:
    """Deletion flow and Qdrant/Neo4j cleanup."""

    # ------------------------------------------------------------------
    # Deletion handling
    # ------------------------------------------------------------------

    def _delete_qdrant_vectors(self, collection_name: str, doc_id: int, chunk_ids: list) -> None:
        """Delete Qdrant vectors for a document, handling missing collections."""
        if not chunk_ids:
            return
        try:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            get_qdrant_client().delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
        except UnexpectedResponse as e:
            if "404" in str(e):
                logger.info(
                    "[WATCHER] Qdrant vectors already gone for document_id=%s",
                    doc_id,
                )
            else:
                logger.warning(
                    "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                    doc_id, e,
                )
        except Exception as e:
            logger.warning(
                "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                doc_id, e,
            )

    def _handle_datastore_deletion(
        self,
        db: Session,
        event_path: str,
        datastore_id: int,
    ) -> None:
        """Handle deletion for a DataStore document."""
        # DataStore deletion: delete the document for this datastore
        doc = (
            db.query(Document)
            .filter(
                Document.file_path == event_path,
                Document.data_store_id == datastore_id,
            )
            .first()
        )
        if doc:
            # Capture IDs before DB deletion — needed for Qdrant/Neo4j
            # cleanup after the DB commit.
            doc_id = doc.id
            chunk_ids = [
                cid[0] for cid in db.query(DocumentChunk.id).filter(
                    DocumentChunk.document_id == doc.id
                ).all()
            ]

            # DB cleanup first. If this commit fails, vector/graph data
            # is still intact and the next scan retries. If it succeeds
            # but Qdrant/Neo4j cleanup fails below, orphaned data is
            # invisible (document gone from DB) and reconciliation
            # cleans it up on next startup.
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()
            db.delete(doc)
            logger.info(
                "[WATCHER] document_deleted path=%s datastore_id=%s doc_id=%s",
                event_path,
                datastore_id,
                doc_id,
            )

        # Remove manifest entry for the deleted file (or if it was never ingested)
        db.query(DataStoreFileManifest).filter(
            DataStoreFileManifest.datastore_id == datastore_id,
            DataStoreFileManifest.file_path == event_path,
        ).delete(synchronize_session=False)

        db.commit()

        # Qdrant cleanup (after DB commit, using captured IDs)
        if doc and chunk_ids:
            self._delete_qdrant_vectors(f"ds_{datastore_id}", doc_id, chunk_ids)

        # Neo4j cleanup (after DB commit, using captured doc_id)
        if doc:
            try:
                from app.services.graph import delete_graph_for_document
                delete_graph_for_document(kb_id=None, document_id=doc_id, data_store_id=datastore_id)
                logger.info(
                    "[WATCHER] Neo4j cleanup done for document_id=%s",
                    doc_id,
                )
            except Exception as e:
                logger.warning(
                    "[WATCHER] Neo4j cleanup failed for document_id=%s: %s",
                    doc_id, e,
                )

    def _handle_kb_deletion(
        self,
        db: Session,
        event_path: str,
        datastore_id: Optional[int],
    ) -> None:
        """Handle deletion for KB documents."""
        # KB deletion: query by org_id from handler mapping
        org_id = self.folder_paths.get(datastore_id, (None,))[0] if datastore_id else None
        if org_id is not None:
            kb_list = (
                db.query(KnowledgeBase)
                .filter(KnowledgeBase.org_id == org_id)
                .values("id")
            )
            kb_list = [kb[0] for kb in kb_list]

            # Capture (kb_id, doc_id, chunk_ids) for all affected docs
            # before DB deletion — needed for Qdrant/Neo4j cleanup after.
            cleanup_targets = []
            for kb_id in kb_list:
                doc = (
                    db.query(Document)
                    .filter(
                        Document.file_path == event_path,
                        Document.knowledge_base_id == kb_id,
                    )
                    .first()
                )

                if doc:
                    chunk_ids = [
                        cid[0] for cid in db.query(DocumentChunk.id).filter(
                            DocumentChunk.document_id == doc.id
                        ).all()
                    ]
                    cleanup_targets.append((kb_id, doc.id, chunk_ids))

                    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                    db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()
                    db.delete(doc)
                    logger.info(
                        "[WATCHER] document_deleted path=%s kb_id=%s doc_id=%s",
                        event_path,
                        kb_id,
                        doc.id,
                    )

            db.commit()

            # Qdrant + Neo4j cleanup after DB commit, using captured IDs
            for kb_id, doc_id, chunk_ids in cleanup_targets:
                self._delete_qdrant_vectors(f"kb_{kb_id}", doc_id, chunk_ids)

                try:
                    from app.services.graph import delete_graph_for_document
                    delete_graph_for_document(kb_id=kb_id, document_id=doc_id)
                    logger.info(
                        "[WATCHER] Neo4j cleanup done for kb_id=%s doc_id=%s",
                        kb_id, doc_id,
                    )
                except Exception as e:
                    logger.warning(
                        "[WATCHER] Neo4j cleanup failed for kb_id=%s doc_id=%s: %s",
                        kb_id, doc_id, e,
                    )

    def _handle_deletion(
        self,
        event_path: str,
        datastore_id: Optional[int],
    ) -> None:
        """Handle file deletion - remove Document records and Qdrant vectors.

        For DataStore files: delete the document for this datastore and its Qdrant vectors.
        For KB files: delete from all KBs for the org and their Qdrant vectors.
        """
        logger.info(
            "[WATCHER] file_deleted path=%s datastore_id=%s",
            event_path,
            datastore_id,
        )

        db: Session = SessionLocal()
        try:
            if datastore_id is not None:
                self._handle_datastore_deletion(db, event_path, datastore_id)
                return

            self._handle_kb_deletion(db, event_path, datastore_id)
        finally:
            db.close()
