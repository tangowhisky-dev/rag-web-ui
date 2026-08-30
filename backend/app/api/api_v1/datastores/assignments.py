"""Organisation assignment endpoints for datastores.

POST   /datastores/{id}/assign  — assign datastore to orgs
DELETE /datastores/{id}/assign  — unassign datastore from orgs

Both endpoints are super-admin-only and scope their operations to
the admin's organisation set.
"""

import logging

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_super_admin
from app.db.session import get_db
from app.models.datastore import OrganizationDataStore
from app.models.organisation import Organisation
from app.models.user import User

from app.api.api_v1.datastores import router
from app.api.api_v1.datastores.schemas import AssignRequest
from app.api.api_v1.datastores.crud import _get_datastore_or_404

logger = logging.getLogger(__name__)


@router.post("/datastores/{datastore_id}/assign")
def assign_datastore_to_orgs(
    datastore_id: int,
    payload: AssignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Assign a datastore to one or more organisations within the admin's scope."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        if not payload.force_clear:
            raise HTTPException(
                status_code=400,
                detail="Empty org_ids would remove all assignments. "
                       "Set force_clear=true to confirm.",
            )
        # Remove only assignments within the admin's scope.
        # super_admin (admin_org_ids=None) has access to all orgs,
        # so don't filter by org_id in that case.
        q = db.query(OrganizationDataStore).filter(
            OrganizationDataStore.data_store_id == datastore_id,
        )
        if admin_org_ids is not None:
            q = q.filter(OrganizationDataStore.org_id.in_(admin_org_ids))
        deleted = q.delete(synchronize_session=False)
        logger.info(
            "[DATASTORE] removed %d assignments in scope for id=%d",
            deleted, datastore_id,
        )
        db.commit()
        return

    for org_id in payload.org_ids:
        if admin_org_ids is not None and org_id not in admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail=f"Organisation outside your scope (id={org_id})",
            )
        org = db.query(Organisation).filter(Organisation.id == org_id).first()
        if org is None:
            raise HTTPException(
                status_code=404,
                detail=f"Organisation not found (id={org_id})",
            )

        # Check for duplicate assignment
        existing = (
            db.query(OrganizationDataStore)
            .filter(
                OrganizationDataStore.data_store_id == datastore_id,
                OrganizationDataStore.org_id == org_id,
            )
            .first()
        )
        if existing:
            continue  # Skip duplicates silently

        link = OrganizationDataStore(
            org_id=org_id,
            data_store_id=datastore_id,
            is_active=True,
        )
        db.add(link)

    db.commit()
    logger.info(
        "[DATASTORE] assigned id=%d to orgs=%s",
        datastore_id,
        payload.org_ids,
    )


@router.delete("/datastores/{datastore_id}/assign")
def unassign_datastore_from_orgs(
    datastore_id: int,
    payload: AssignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Unassign a datastore from orgs within the admin's scope."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        raise HTTPException(
            status_code=400,
            detail="org_ids required — cannot unassign without specifying "
                   "which organisations to remove.",
        )

    for org_id in payload.org_ids:
        if admin_org_ids is not None and org_id not in admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail=f"Organisation outside your scope (id={org_id})",
            )
        link = (
            db.query(OrganizationDataStore)
            .filter(
                OrganizationDataStore.data_store_id == datastore_id,
                OrganizationDataStore.org_id == org_id,
            )
            .first()
        )
        if link:
            db.delete(link)

    db.commit()
