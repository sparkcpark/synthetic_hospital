"""Terminology service endpoints — FHIR CodeSystem and ValueSet operations.

Provides $lookup, $expand, and $validate-code against the terminology_codes table
(ICD-10-CM, SNOMED CT, and LOINC).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.models.base import get_db
from epic_sim.app.models.ontology import TerminologyCode

router = APIRouter()

# System URI → internal system name
SYSTEM_MAP = {
    "http://hl7.org/fhir/sid/icd-10-cm": "icd10cm",
    "http://snomed.info/sct": "snomedct",
    "http://loinc.org": "loinc",
    "icd10cm": "icd10cm",
    "snomedct": "snomedct",
    "loinc": "loinc",
}


def _resolve_system(system: str) -> str:
    """Resolve a FHIR system URI to internal name."""
    return SYSTEM_MAP.get(system, system)


@router.get("/CodeSystem/$lookup")
async def code_system_lookup(
    system: str = Query(..., description="Code system URI (e.g., http://hl7.org/fhir/sid/icd-10-cm)"),
    code: str = Query(..., description="Code to look up"),
    db: AsyncSession = Depends(get_db),
):
    """FHIR CodeSystem $lookup — look up a code in a code system."""
    internal_system = _resolve_system(system)

    result = await db.execute(
        select(TerminologyCode).where(
            TerminologyCode.system == internal_system,
            TerminologyCode.code == code,
        )
    )
    term = result.scalar_one_or_none()

    if term is None:
        raise HTTPException(status_code=404, detail=f"Code '{code}' not found in system '{system}'")

    response = {
        "resourceType": "Parameters",
        "parameter": [
            {"name": "name", "valueString": internal_system},
            {"name": "display", "valueString": term.display},
            {"name": "active", "valueBoolean": term.is_active},
        ],
    }

    # Add properties if available
    if term.properties:
        for key, value in term.properties.items():
            response["parameter"].append({
                "name": "property",
                "part": [
                    {"name": "code", "valueCode": key},
                    {"name": "value", "valueString": str(value)},
                ],
            })

    return response


@router.get("/ValueSet/$expand")
async def value_set_expand(
    url: str | None = Query(None, description="ValueSet URL"),
    filter: str | None = Query(None, description="Text filter for matching concepts"),
    system: str | None = Query(None, description="Code system to filter"),
    count: int = Query(20, le=100, alias="count"),
    offset: int = Query(0, alias="offset"),
    db: AsyncSession = Depends(get_db),
):
    """FHIR ValueSet $expand — expand a value set, optionally filtering by text."""
    q = select(TerminologyCode).where(TerminologyCode.is_active.is_(True))

    if system:
        q = q.where(TerminologyCode.system == _resolve_system(system))

    if filter:
        # Use ILIKE for text search (pg_trgm GIN index would accelerate this)
        q = q.where(TerminologyCode.display.ilike(f"%{filter}%"))

    # Count total
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Paginate
    q = q.order_by(TerminologyCode.system, TerminologyCode.code).offset(offset).limit(count)
    result = await db.execute(q)
    terms = result.scalars().all()

    # System URI mapping (reverse)
    system_uri_map = {
        "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
        "snomedct": "http://snomed.info/sct",
        "loinc": "http://loinc.org",
    }

    contains = []
    for t in terms:
        contains.append({
            "system": system_uri_map.get(t.system, t.system),
            "code": t.code,
            "display": t.display,
        })

    return {
        "resourceType": "ValueSet",
        "expansion": {
            "total": total,
            "offset": offset,
            "contains": contains,
        },
    }


@router.get("/ValueSet/$validate-code")
async def value_set_validate_code(
    system: str = Query(..., description="Code system URI"),
    code: str = Query(..., description="Code to validate"),
    db: AsyncSession = Depends(get_db),
):
    """FHIR ValueSet $validate-code — check if a code is valid."""
    internal_system = _resolve_system(system)

    result = await db.execute(
        select(TerminologyCode).where(
            TerminologyCode.system == internal_system,
            TerminologyCode.code == code,
        )
    )
    term = result.scalar_one_or_none()

    return {
        "resourceType": "Parameters",
        "parameter": [
            {"name": "result", "valueBoolean": term is not None},
            {"name": "display", "valueString": term.display if term else None},
        ],
    }
