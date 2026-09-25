"""All SQLAlchemy models — import here so Alembic can discover them."""

from epic_sim.app.models.auth import AgentSubmission, AuthToken, AuthUser, PatientAssignment
from epic_sim.app.models.base import Base
from epic_sim.app.models.benchmark import (
    BenchmarkGroundTruth,
    EvaluationPrediction,
    EvaluationRun,
    ImagingOrder,
    RelevanceJudgment,
)
from epic_sim.app.models.content import BoardQuestion, FactCard
from epic_sim.app.models.ehr import EhrSection
from epic_sim.app.models.longitudinal import (
    EncounterEhrSection,
    LongitudinalEncounter,
    LongitudinalPatient,
)
from epic_sim.app.models.metadata import LlmCallLog, ProcessingLog
from epic_sim.app.models.ontology import ClinicalFinding, Diagnosis, TerminologyCode
from epic_sim.app.models.provenance import RawCard, SourceDeck
from epic_sim.app.models.relationships import (
    DiagnosisFinding,
    FactDiagnosisLink,
    FactFindingLink,
    QuestionDiagnosis,
    QuestionFinding,
)

__all__ = [
    "Base",
    "SourceDeck", "RawCard",
    "BoardQuestion", "FactCard",
    "Diagnosis", "ClinicalFinding", "TerminologyCode",
    "QuestionDiagnosis", "QuestionFinding", "DiagnosisFinding",
    "FactDiagnosisLink", "FactFindingLink",
    "EhrSection",
    "LongitudinalPatient", "LongitudinalEncounter", "EncounterEhrSection",
    "BenchmarkGroundTruth", "RelevanceJudgment",
    "EvaluationRun", "EvaluationPrediction", "ImagingOrder",
    "ProcessingLog", "LlmCallLog",
    "AuthUser", "AuthToken", "PatientAssignment", "AgentSubmission",
]
