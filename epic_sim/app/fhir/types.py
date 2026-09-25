"""FHIR R4 base data types as Pydantic v2 models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FHIRBase(BaseModel):
    """Base for all FHIR types — camelCase serialization."""

    model_config = ConfigDict(populate_by_name=True, ser_json_timedelta="iso8601")


class Coding(FHIRBase):
    system: str | None = None
    version: str | None = None
    code: str | None = None
    display: str | None = None
    user_selected: bool | None = Field(None, alias="userSelected")


class CodeableConcept(FHIRBase):
    coding: list[Coding] = Field(default_factory=list)
    text: str | None = None


class Reference(FHIRBase):
    reference: str | None = None
    type: str | None = None
    display: str | None = None


class Period(FHIRBase):
    start: str | None = None
    end: str | None = None


class HumanName(FHIRBase):
    use: str | None = None
    text: str | None = None
    family: str | None = None
    given: list[str] = Field(default_factory=list)
    prefix: list[str] = Field(default_factory=list)
    suffix: list[str] = Field(default_factory=list)


class Identifier(FHIRBase):
    use: str | None = None
    type: CodeableConcept | None = None
    system: str | None = None
    value: str | None = None


class Address(FHIRBase):
    use: str | None = None
    type: str | None = None
    text: str | None = None
    line: list[str] = Field(default_factory=list)
    city: str | None = None
    state: str | None = None
    postal_code: str | None = Field(None, alias="postalCode")
    country: str | None = None


class ContactPoint(FHIRBase):
    system: str | None = None
    value: str | None = None
    use: str | None = None


class Quantity(FHIRBase):
    value: float | None = None
    comparator: str | None = None
    unit: str | None = None
    system: str | None = None
    code: str | None = None


class Narrative(FHIRBase):
    status: str = "generated"
    div: str = ""


class Attachment(FHIRBase):
    content_type: str | None = Field(None, alias="contentType")
    url: str | None = None
    title: str | None = None
    size: int | None = None


class Meta(FHIRBase):
    version_id: str | None = Field(None, alias="versionId")
    last_updated: str | None = Field(None, alias="lastUpdated")
    profile: list[str] = Field(default_factory=list)


# -- FHIR Resource base --

class Resource(FHIRBase):
    resource_type: str = Field(..., alias="resourceType")
    id: str | None = None
    meta: Meta | None = None


class DomainResource(Resource):
    text: Narrative | None = None


# -- Bundle types --

class BundleLink(FHIRBase):
    relation: str
    url: str


class BundleEntrySearch(FHIRBase):
    mode: str = "match"


class BundleEntry(FHIRBase):
    full_url: str | None = Field(None, alias="fullUrl")
    resource: dict[str, Any] | None = None
    search: BundleEntrySearch | None = None


class Bundle(FHIRBase):
    resource_type: str = Field("Bundle", alias="resourceType")
    id: str | None = None
    type: str = "searchset"
    total: int = 0
    link: list[BundleLink] = Field(default_factory=list)
    entry: list[BundleEntry] = Field(default_factory=list)


class OperationOutcomeIssue(FHIRBase):
    severity: str
    code: str
    diagnostics: str | None = None
    details: CodeableConcept | None = None


class OperationOutcome(FHIRBase):
    resource_type: str = Field("OperationOutcome", alias="resourceType")
    issue: list[OperationOutcomeIssue] = Field(default_factory=list)
