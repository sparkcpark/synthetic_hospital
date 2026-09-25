"""PostgreSQL ENUM types for the schema."""

import enum


class DeckType(str, enum.Enum):
    board_exam = "board_exam"
    fact = "fact"


class CardType(str, enum.Enum):
    board_exam = "board_exam"
    fact = "fact"


class QuestionFormat(str, enum.Enum):
    mcq_vignette = "mcq_vignette"
    cloze_objective = "cloze_objective"
    cloze_clinical = "cloze_clinical"
    basic_qa = "basic_qa"


class FactFormat(str, enum.Enum):
    cloze_fact = "cloze_fact"
    basic_fact = "basic_fact"
    image_occlusion = "image_occlusion"


class FindingType(str, enum.Enum):
    symptom = "symptom"
    sign = "sign"
    lab_value = "lab_value"
    vital_sign = "vital_sign"
    imaging_finding = "imaging_finding"
    procedure_result = "procedure_result"
    history_item = "history_item"
    medication = "medication"
    demographic = "demographic"


class DxRole(str, enum.Enum):
    correct = "correct"
    distractor = "distractor"
    secondary = "secondary"
    predisposing = "predisposing"


class FindingRelevance(str, enum.Enum):
    key = "key"
    supporting = "supporting"
    background = "background"
    distractor = "distractor"


class DxFindingRelationship(str, enum.Enum):
    pathognomonic = "pathognomonic"
    highly_suggestive = "highly_suggestive"
    commonly_seen = "commonly_seen"
    risk_factor = "risk_factor"
    protective = "protective"
    rules_out = "rules_out"


class Acuity(str, enum.Enum):
    acute = "acute"
    chronic = "chronic"
    acute_on_chronic = "acute_on_chronic"
    unspecified = "unspecified"


class EncounterType(str, enum.Enum):
    outpatient = "outpatient"
    ed = "ed"
    inpatient = "inpatient"
    icu = "icu"
    follow_up = "follow_up"
    procedure = "procedure"
    telehealth = "telehealth"


class SectionType(str, enum.Enum):
    demographics = "demographics"
    chief_complaint = "chief_complaint"
    hpi = "hpi"
    pmh = "pmh"
    psh = "psh"
    medications = "medications"
    allergies = "allergies"
    family_history = "family_history"
    social_history = "social_history"
    ros = "ros"
    vitals = "vitals"
    physical_exam = "physical_exam"
    labs = "labs"
    imaging = "imaging"
    pathology = "pathology"
    other_studies = "other_studies"
    assessment = "assessment"
    plan = "plan"


class EvalTask(str, enum.Enum):
    diagnosis_accuracy = "diagnosis_accuracy"
    context_summarization = "context_summarization"
    evidence_retrieval = "evidence_retrieval"
    imaging_indication = "imaging_indication"


class GtGranularity(str, enum.Enum):
    question = "question"
    patient = "patient"
    encounter = "encounter"


class Difficulty(str, enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class SplitType(str, enum.Enum):
    train = "train"
    val = "val"
    test = "test"


class OrderPriority(str, enum.Enum):
    routine = "routine"
    stat = "stat"
    urgent = "urgent"


class ClassificationMethod(str, enum.Enum):
    rule_based = "rule_based"
    llm = "llm"


class ExtractionMethod(str, enum.Enum):
    rule_based = "rule_based"
    llm = "llm"
    hybrid = "hybrid"


class DataSource(str, enum.Enum):
    rule_based = "rule_based"
    llm = "llm"
    manual = "manual"


class GenerationMethod(str, enum.Enum):
    template = "template"
    llm = "llm"
    hybrid = "hybrid"


class PassageSource(str, enum.Enum):
    ehr_section = "ehr_section"
    fact_card = "fact_card"
    encounter_section = "encounter_section"


class FactDxRelevance(str, enum.Enum):
    defines = "defines"
    differentiates = "differentiates"
    treatment = "treatment"
    epidemiology = "epidemiology"
    mechanism = "mechanism"


class FactCfRelevance(str, enum.Enum):
    defines = "defines"
    explains = "explains"
    interpretation = "interpretation"
    normal_variant = "normal_variant"


class MethodType(str, enum.Enum):
    llm_direct = "llm_direct"
    rag = "rag"
    fine_tuned = "fine_tuned"
    ensemble = "ensemble"
    sparse_retrieval = "sparse_retrieval"
    dense_retrieval = "dense_retrieval"
    hybrid_retrieval = "hybrid_retrieval"


class ProcessingStatus(str, enum.Enum):
    started = "started"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"
