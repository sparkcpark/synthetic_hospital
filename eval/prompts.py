"""Prompt templates for all 4 evaluation tasks × 4 strategies.

Each task has a system prompt and a user prompt template with named placeholders.
Strategies: zero_shot, few_shot, cot, structured.
"""

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A system + user prompt pair."""
    system: str
    user: str


# Unified JSON closing line used across all non-CoT templates.
_JSON_CLOSING = (
    "Return ONLY a valid JSON object. Do not include any text, explanation, "
    "or markdown before or after the JSON. All required fields must be present "
    "in the response; use null for unknown values rather than omitting fields."
)


# ============================================================================
# DIAGNOSIS ACCURACY
# ============================================================================

DIAGNOSIS_SYSTEM = (
    "You are a senior physician with broad clinical training across internal "
    "medicine, surgery, pediatrics, obstetrics and gynecology, psychiatry, and "
    "emergency medicine. You reason from clinical presentations to coded "
    "diagnoses using ICD-10-CM terminology. When uncertain between diagnoses, "
    "express calibrated confidence: reserve scores above 0.8 for cases with "
    "pathognomonic findings, use 0.5\u20130.7 for probable diagnoses with a "
    "plausible differential, and below 0.5 for possibilities requiring further "
    "workup."
)

DIAGNOSIS_ZERO_SHOT = PromptTemplate(
    system=DIAGNOSIS_SYSTEM,
    user=(
        "PATIENT CLINICAL DATA:\n{ehr_text}\n\n"
        "Based on the clinical presentation above, provide your top 5 most likely "
        "diagnoses ranked by probability. For each diagnosis, provide the ICD-10-CM "
        "code and a confidence score (0.0 to 1.0).\n\n"
        "For each diagnosis, you must include the ICD-10-CM code (e.g., I21.01, "
        "J18.9). If you are uncertain of the exact code, provide your best "
        "estimate \u2014 do not omit the field.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"diagnoses": [{{"rank": 1, "icd10": "I21.01", "name": "...", "confidence": 0.95}}, ...]}}'
    ),
)

# Examples are selected at runtime via embedding similarity to the current item.
# Maximum 2 examples. Populated by format_prompt(); falls back to zero_shot if empty.
DIAGNOSIS_FEW_SHOT = PromptTemplate(
    system=DIAGNOSIS_SYSTEM,
    user=(
        "Here are examples of clinical presentations and their diagnoses:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now diagnose this patient:\n\n"
        "PATIENT CLINICAL DATA:\n{ehr_text}\n\n"
        "Provide your top 5 most likely diagnoses ranked by probability.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"diagnoses": [{{"rank": 1, "icd10": "I21.01", "name": "...", "confidence": 0.95}}, ...]}}'
    ),
)

DIAGNOSIS_COT = PromptTemplate(
    system=DIAGNOSIS_SYSTEM,
    user=(
        "PATIENT CLINICAL DATA:\n{ehr_text}\n\n"
        "Think step-by-step through the clinical reasoning:\n"
        "1. Identify key symptoms, signs, and lab findings\n"
        "2. Consider the most likely diagnoses that fit the clinical picture\n"
        "3. Rank your differential by probability\n"
        "4. For each diagnosis in your ranked list, identify the most specific "
        "ICD-10-CM code that matches. Use the 4\u20137 character billable code "
        "level where possible (e.g., prefer I21.01 over I21).\n\n"
        "After your reasoning, provide your final answer as valid JSON:\n"
        '{{"reasoning": "...", "diagnoses": [{{"rank": 1, "icd10": "I21.01", "name": "...", "confidence": 0.95}}, ...]}}'
    ),
)

DIAGNOSIS_STRUCTURED = PromptTemplate(
    system=DIAGNOSIS_SYSTEM,
    user=(
        "{structured_hints}"
        "PATIENT CLINICAL DATA:\n{ehr_text}\n\n"
        "Provide your differential diagnosis. Follow this exact JSON schema:\n"
        '{{\n'
        '  "diagnoses": [\n'
        '    {{\n'
        '      "rank": <integer 1-5>,\n'
        '      "icd10": "<ICD-10-CM code, e.g. I21.01>",\n'
        '      "name": "<diagnosis display name>",\n'
        '      "confidence": <float 0.0-1.0>\n'
        '    }}\n'
        '  ]\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# PATIENT DIAGNOSIS (longitudinal problem list)
# ============================================================================

PATIENT_DIAGNOSIS_SYSTEM = (
    "You are a senior physician reviewing a patient's complete longitudinal "
    "medical record. Your task is to identify all active diagnoses and chronic "
    "conditions from the clinical data provided. Classify each diagnosis as "
    "acute, chronic, or acute_on_chronic based on clinical evidence. Use "
    "ICD-10-CM codes for all diagnoses. Do not list the same condition more "
    "than once. When uncertain between acuity classifications, consider "
    "whether any chronic condition shows evidence of current exacerbation "
    "\u2014 if so, classify as acute_on_chronic."
)

PATIENT_DIAGNOSIS_ZERO_SHOT = PromptTemplate(
    system=PATIENT_DIAGNOSIS_SYSTEM,
    user=(
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Review the complete medical record and identify ALL active diagnoses "
        "and chronic conditions. For each, assign an acuity label: acute, "
        "chronic, or acute_on_chronic (for chronic conditions with current "
        "exacerbation). You must include the ICD-10-CM code for each diagnosis. "
        "If uncertain, provide your best estimate \u2014 do not omit the field.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"active_diagnoses": [{{"icd10": "I21.01", "name": "...", "acuity": "acute"}}, ...], '
        '"chronic_conditions": [{{"icd10": "I10", "name": "...", "acuity": "chronic"}}, ...]}}'
    ),
)

PATIENT_DIAGNOSIS_FEW_SHOT = PromptTemplate(
    system=PATIENT_DIAGNOSIS_SYSTEM,
    user=(
        "Here are examples of longitudinal problem list extraction:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now extract the problem list for this patient:\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Identify ALL active diagnoses and chronic conditions with ICD-10-CM "
        "codes and acuity labels.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"active_diagnoses": [{{"icd10": "I21.01", "name": "...", "acuity": "acute"}}, ...], '
        '"chronic_conditions": [{{"icd10": "I10", "name": "...", "acuity": "chronic"}}, ...]}}'
    ),
)

PATIENT_DIAGNOSIS_COT = PromptTemplate(
    system=PATIENT_DIAGNOSIS_SYSTEM,
    user=(
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Think step-by-step through the longitudinal record:\n"
        "1. Identify diagnoses per encounter from the record\n"
        "2. Merge the timeline to deduplicate \u2014 same condition across encounters counts once\n"
        "3. For each unique condition, check if any encounter shows an acute flare of a "
        "known chronic condition \u2192 label acute_on_chronic\n"
        "4. Classify remaining as acute or chronic\n"
        "5. Assign the most specific ICD-10-CM code (4\u20137 character billable level)\n\n"
        "After your reasoning, provide your final answer as valid JSON:\n"
        '{{"reasoning": "...", '
        '"active_diagnoses": [{{"icd10": "I21.01", "name": "...", "acuity": "acute"}}, ...], '
        '"chronic_conditions": [{{"icd10": "I10", "name": "...", "acuity": "chronic"}}, ...]}}'
    ),
)

PATIENT_DIAGNOSIS_STRUCTURED = PromptTemplate(
    system=PATIENT_DIAGNOSIS_SYSTEM,
    user=(
        "{structured_hints}"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Identify ALL active diagnoses and chronic conditions. "
        "Follow this exact JSON schema:\n"
        '{{\n'
        '  "active_diagnoses": [\n'
        '    {{"icd10": "<ICD-10-CM code>", "name": "<diagnosis name>", "acuity": "acute|acute_on_chronic"}}\n'
        '  ],\n'
        '  "chronic_conditions": [\n'
        '    {{"icd10": "<ICD-10-CM code>", "name": "<diagnosis name>", "acuity": "chronic"}}\n'
        '  ]\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# CONTEXT SUMMARIZATION
# ============================================================================

SUMMARIZATION_SYSTEM = (
    "You are an attending physician synthesizing a patient's longitudinal "
    "medical record. Your summaries are concise, clinically accurate, and "
    "grounded strictly in findings documented in the provided EHR data \u2014 you "
    "never introduce clinical details not present in the source record. You "
    "prioritize findings directly relevant to active diagnoses and clinical "
    "decision-making over incidental background information."
)

SUMMARIZATION_ZERO_SHOT = PromptTemplate(
    system=SUMMARIZATION_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "Include only findings explicitly documented in the record below. Do not "
        "infer, extrapolate, or introduce clinical details not present in the "
        "source EHR.\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write a comprehensive clinical summary (5-10 sentences) that addresses "
        "the clinical question. Include active diagnoses, relevant history, key "
        "findings, and clinical trajectory.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

# Examples are selected at runtime via embedding similarity to the current item.
# Maximum 2 examples. Populated by format_prompt(); falls back to zero_shot if empty.
SUMMARIZATION_FEW_SHOT = PromptTemplate(
    system=SUMMARIZATION_SYSTEM,
    user=(
        "Here are examples of clinical summaries:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now summarize this patient:\n\n"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write a comprehensive clinical summary (5-10 sentences).\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

SUMMARIZATION_COT = PromptTemplate(
    system=SUMMARIZATION_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Think step-by-step:\n"
        "1. Identify the patient's active diagnoses\n"
        "2. Note relevant history and findings across encounters\n"
        "3. Trace the clinical trajectory over time\n"
        "4. Synthesize into a coherent summary\n"
        "5. Before finalizing your summary, review each sentence and confirm it "
        "is directly supported by a finding present in the provided record. "
        "Remove or rephrase any sentence that introduces information not "
        "documented in the source EHR.\n\n"
        "After your reasoning, provide your final summary as valid JSON:\n"
        '{{"reasoning": "...", "summary": "..."}}'
    ),
)

SUMMARIZATION_STRUCTURED = PromptTemplate(
    system=SUMMARIZATION_SYSTEM,
    user=(
        "{structured_hints}"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Provide a clinical summary following this exact JSON schema:\n"
        '{{\n'
        '  "summary": "<5-10 sentence clinical summary addressing the question>"\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# CONTEXT SUMMARIZATION — CURRENT-VISIT VARIANT (point-in-time)
# ============================================================================

CURRENT_VISIT_SYSTEM = (
    "You are an attending physician writing a focused current-visit summary from "
    "a patient's longitudinal record. You summarize the most recent (current) "
    "encounter in context: the reason for the visit, what has changed since the "
    "prior encounter, and the active problems being managed now. You use only "
    "information documented up to and including the current encounter, you never "
    "reference or infer future events, and you never introduce clinical details "
    "not present in the record."
)

_CURRENT_VISIT_TASK = (
    "The record below ends at the patient's CURRENT encounter. Summarize only the "
    "current visit in the context of the prior encounters: (1) the reason for the "
    "visit, (2) what has changed since the prior encounter (new and resolved "
    "findings), and (3) the active problems being managed now. Use only "
    "information documented in the record below; do not reference future events."
)

CURRENT_VISIT_ZERO_SHOT = PromptTemplate(
    system=CURRENT_VISIT_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        f"{_CURRENT_VISIT_TASK}\n\n"
        "PATIENT MEDICAL RECORD (through the current encounter):\n{ehr_text}\n\n"
        "Write a focused current-visit summary that captures all clinically "
        "relevant information for this visit; omit unrelated or normal detail "
        "(no length limit).\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

CURRENT_VISIT_FEW_SHOT = PromptTemplate(
    system=CURRENT_VISIT_SYSTEM,
    user=(
        "Here are examples of current-visit summaries:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now summarize the current visit for this patient:\n\n"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        f"{_CURRENT_VISIT_TASK}\n\n"
        "PATIENT MEDICAL RECORD (through the current encounter):\n{ehr_text}\n\n"
        "Write a focused current-visit summary that captures all clinically "
        "relevant information for this visit; omit unrelated or normal detail "
        "(no length limit).\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

CURRENT_VISIT_COT = PromptTemplate(
    system=CURRENT_VISIT_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD (through the current encounter):\n{ehr_text}\n\n"
        "Think step-by-step:\n"
        "1. Identify the current (most recent) encounter and its chief complaint\n"
        "2. Compare it to the prior encounter: which findings are new, which resolved\n"
        "3. Identify the active problems being managed at the current visit\n"
        "4. Confirm every statement is supported by the record at or before the "
        "current encounter — never use future information\n\n"
        "After your reasoning, provide your final summary as valid JSON:\n"
        '{{"reasoning": "...", "summary": "..."}}'
    ),
)

CURRENT_VISIT_STRUCTURED = PromptTemplate(
    system=CURRENT_VISIT_SYSTEM,
    user=(
        "{structured_hints}"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD (through the current encounter):\n{ehr_text}\n\n"
        "Provide a current-visit summary following this exact JSON schema:\n"
        '{{\n'
        '  "summary": "<current-visit summary: reason for visit, interval changes '
        'since the prior encounter, and active problems; thorough but focused, no length limit>"\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# --- Specialty-conditioned summarization (spec_v1.33 §17) ---------------------

SPECIALTY_SYSTEM = (
    "You are a subspecialty consultant writing a focused, specialty-specific summary "
    "of a patient's chart. You summarize only what bears on the requested specialty's "
    "care: that specialty's active problems plus the comorbidities, labs, and "
    "medications from other systems that are clinically relevant to managing them. You "
    "deliberately omit problems and findings unrelated to the requested specialty. If "
    "the patient has no active problem in the requested specialty, you say so explicitly "
    "and do NOT summarize the unrelated problems. You never introduce clinical details "
    "not present in the record."
)

_SPECIALTY_TASK = (
    "Summarize this patient's record from the perspective of the requested specialty "
    "only: (1) the active problems belonging to that specialty, and (2) the comorbidities, "
    "labs, and medications from other systems that are clinically relevant to managing "
    "them. Omit problems and findings unrelated to the requested specialty. If the record "
    "contains no active problem in the requested specialty, state that explicitly instead "
    "of summarizing unrelated issues. Use only information in the record below."
)

SPECIALTY_ZERO_SHOT = PromptTemplate(
    system=SPECIALTY_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        f"{_SPECIALTY_TASK}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write the specialty-specific summary (no length limit); omit unrelated systems.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

# NEUTRAL specialty baseline (ablation): specialty framing only — NO graph cue to include
# comorbidities and NO instruction to omit them. The primary ablation contrast (graph vs
# neutral) isolates the typed-edge guidance from the model's spontaneous comorbidity inclusion.
SPECIALTY_NEUTRAL_SYSTEM = (
    "You are a clinician summarizing a patient's chart from the perspective of the requested "
    "specialty. You never introduce clinical details not present in the record. If the patient "
    "has no active problem in the requested specialty, say so explicitly."
)

SPECIALTY_NEUTRAL = PromptTemplate(
    system=SPECIALTY_NEUTRAL_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "Summarize this patient's record from the perspective of the requested specialty. "
        "Use only information in the record below.\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write the summary (no length limit).\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

# Same-specialty-only BASELINE (ablation): identical framing to SPECIALTY_ZERO_SHOT but
# WITHOUT the "+ relevant comorbidities from other systems" clause. Used only in the paired
# relevance ablation to isolate the contribution of the typed-edge relevance machinery.
SPECIALTY_SAME_ONLY_SYSTEM = (
    "You are a subspecialty consultant writing a focused summary of a patient's chart, "
    "restricted to the requested specialty's OWN active problems. You summarize only the "
    "problems and findings that belong to the requested specialty itself. You do NOT include "
    "comorbidities, labs, or medications from other organ systems, even if they might bear on "
    "the specialty's care. If the patient has no active problem in the requested specialty, you "
    "say so explicitly. You never introduce clinical details not present in the record."
)

_SPECIALTY_SAME_ONLY_TASK = (
    "Summarize ONLY the active problems that belong to the requested specialty itself. Do NOT "
    "include comorbidities, labs, or medications from other organ systems. If the record contains "
    "no active problem in the requested specialty, state that explicitly. Use only information in "
    "the record below."
)

SPECIALTY_SAME_ONLY = PromptTemplate(
    system=SPECIALTY_SAME_ONLY_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        f"{_SPECIALTY_SAME_ONLY_TASK}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write the summary (no length limit); include ONLY the requested specialty's own active problems.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

SPECIALTY_FEW_SHOT = PromptTemplate(
    system=SPECIALTY_SYSTEM,
    user=(
        "Here are examples of specialty-specific summaries:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now write the specialty-specific summary for this patient:\n\n"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        f"{_SPECIALTY_TASK}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Write the specialty-specific summary (no length limit); omit unrelated systems.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"summary": "..."}}'
    ),
)

SPECIALTY_COT = PromptTemplate(
    system=SPECIALTY_SYSTEM,
    user=(
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Think step-by-step:\n"
        "1. Identify the requested specialty from the clinical question\n"
        "2. Find the active problems in the record that belong to that specialty\n"
        "3. If there are none, prepare to state that no active problem exists for it\n"
        "4. Otherwise, identify comorbidities/labs/meds from other systems that are "
        "clinically relevant to those problems, and exclude everything unrelated\n\n"
        "After your reasoning, provide your final summary as valid JSON:\n"
        '{{"reasoning": "...", "summary": "..."}}'
    ),
)

SPECIALTY_STRUCTURED = PromptTemplate(
    system=SPECIALTY_SYSTEM,
    user=(
        "{structured_hints}"
        "CLINICAL QUESTION: {clinical_question}\n\n"
        "PATIENT MEDICAL RECORD:\n{ehr_text}\n\n"
        "Provide a specialty-specific summary following this exact JSON schema:\n"
        '{{\n'
        '  "summary": "<the requested specialty\'s active problems and the '
        'comorbidities/labs/meds relevant to them; omit unrelated systems; if none, say '
        'so explicitly>"\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# EVIDENCE RETRIEVAL (LLM reranking)
# ============================================================================

RETRIEVAL_SYSTEM = (
    "You are a clinical evidence grader assessing how strongly each passage "
    "supports a diagnosis. Use this scale strictly: Grade 3 \u2014 the passage "
    "contains findings that are pathognomonic for or highly characteristic of "
    "the diagnosis, or directly defines/explains the condition's key diagnostic "
    "criteria. Grade 2 \u2014 the passage contains findings commonly associated "
    "with the diagnosis that meaningfully narrow the differential. Grade 1 \u2014 "
    "the passage provides relevant clinical context or background that is "
    "marginally informative. Grade 0 \u2014 the passage contains no information "
    "relevant to the diagnosis, or contains only distracting findings. Apply "
    "grades consistently."
)

RETRIEVAL_ZERO_SHOT = PromptTemplate(
    system=RETRIEVAL_SYSTEM,
    user=(
        "DIAGNOSIS: {diagnosis}\n\n"
        "For each passage, assign a relevance grade (0\u20133) according to the "
        "grading scale in your instructions, then return all passages ranked "
        "from most to least relevant.\n\n"
        "PASSAGES:\n{passages}\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"rankings": [{{"passage_id": "...", "grade": 3}}, ...]}}'
    ),
)

# Examples are selected at runtime via embedding similarity to the current item.
# Maximum 2 examples. Populated by format_prompt(); falls back to zero_shot if empty.
RETRIEVAL_FEW_SHOT = PromptTemplate(
    system=RETRIEVAL_SYSTEM,
    user=(
        "Here are examples of relevance grading:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now grade these passages:\n\n"
        "DIAGNOSIS: {diagnosis}\n\n"
        "PASSAGES:\n{passages}\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"rankings": [{{"passage_id": "...", "grade": 3}}, ...]}}'
    ),
)

RETRIEVAL_COT = PromptTemplate(
    system=RETRIEVAL_SYSTEM,
    user=(
        "DIAGNOSIS: {diagnosis}\n\n"
        "PASSAGES:\n{passages}\n\n"
        "Before assigning each grade, briefly note whether the passage contains: "
        "(a) a direct finding match to the diagnosis, (b) a related but "
        "non-specific finding, or (c) no relevant finding. Use this "
        "classification to assign your grade \u2014 do not deliberate further. "
        "Prioritize speed and consistency over exhaustive analysis.\n\n"
        "After your reasoning, provide grades as valid JSON:\n"
        '{{"reasoning": "...", "rankings": [{{"passage_id": "...", "grade": 3}}, ...]}}'
    ),
)

RETRIEVAL_STRUCTURED = PromptTemplate(
    system=RETRIEVAL_SYSTEM,
    user=(
        "{structured_hints}"
        "DIAGNOSIS: {diagnosis}\n\n"
        "PASSAGES:\n{passages}\n\n"
        "Assign relevance grades (0=not relevant, 1=marginal, 2=relevant, 3=highly relevant).\n\n"
        "Follow this exact JSON schema:\n"
        '{{\n'
        '  "rankings": [\n'
        '    {{"passage_id": "<passage ID>", "grade": <0|1|2|3>}}\n'
        '  ]\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# IMAGING CLINICAL INDICATION
# ============================================================================

IMAGING_SYSTEM = (
    "You are a radiologist preparing a pre-read assessment. You receive imaging "
    "orders with intentionally brief, vague clinical indications \u2014 your task "
    "is to use the patient's EHR context to infer the underlying clinical "
    "question the ordering provider is investigating. Generate pre-read "
    "summaries that orient your read toward the most clinically relevant "
    "findings. Express differential diagnoses using both plain language and "
    "ICD-10-CM codes."
)

IMAGING_ZERO_SHOT = PromptTemplate(
    system=IMAGING_SYSTEM,
    user=(
        "IMAGING ORDER:\n"
        "  Modality: {modality}\n"
        "  Body Region: {body_region}\n"
        "  Clinical Indication: {clinical_indication}\n\n"
        "PATIENT CLINICAL RECORD:\n{ehr_text}\n\n"
        "Based on the clinical context, provide:\n"
        "1. The inferred clinical question the ordering provider likely had\n"
        "2. A pre-read summary (3-5 sentences) for the interpreting radiologist\n"
        "3. Key findings to look for on the study\n"
        "4. Differential diagnoses with likelihood\n\n"
        "For each differential diagnosis, you must provide both the diagnosis "
        "name and its ICD-10-CM code. If uncertain of the exact code, provide "
        "your best estimate \u2014 do not omit the field.\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"clinical_question": "...", "pre_read_summary": "...", '
        '"must_include_findings": ["...", ...], '
        '"differential": [{{"diagnosis": "...", "icd10": "...", "likelihood": "high|moderate|low"}}, ...]}}'
    ),
)

# Examples are selected at runtime via embedding similarity to the current item.
# Maximum 2 examples. Populated by format_prompt(); falls back to zero_shot if empty.
IMAGING_FEW_SHOT = PromptTemplate(
    system=IMAGING_SYSTEM,
    user=(
        "Here are examples of imaging pre-reads:\n\n"
        "{few_shot_examples}\n\n"
        "---\n\n"
        "Now prepare a pre-read for this order:\n\n"
        "IMAGING ORDER:\n"
        "  Modality: {modality}\n"
        "  Body Region: {body_region}\n"
        "  Clinical Indication: {clinical_indication}\n\n"
        "PATIENT CLINICAL RECORD:\n{ehr_text}\n\n"
        f"{_JSON_CLOSING}\n"
        '{{"clinical_question": "...", "pre_read_summary": "...", '
        '"must_include_findings": ["...", ...], '
        '"differential": [{{"diagnosis": "...", "icd10": "...", "likelihood": "high|moderate|low"}}, ...]}}'
    ),
)

IMAGING_COT = PromptTemplate(
    system=IMAGING_SYSTEM,
    user=(
        "IMAGING ORDER:\n"
        "  Modality: {modality}\n"
        "  Body Region: {body_region}\n"
        "  Clinical Indication: {clinical_indication}\n\n"
        "PATIENT CLINICAL RECORD:\n{ehr_text}\n\n"
        "Think step-by-step:\n"
        "1. What is the likely clinical question behind this vague indication?\n"
        "2. What relevant clinical findings support this question?\n"
        "3. What should the radiologist look for?\n"
        "4. What are the differential diagnoses?\n\n"
        "After your reasoning, provide your final answer as valid JSON:\n"
        '{{"reasoning": "...", "clinical_question": "...", "pre_read_summary": "...", '
        '"must_include_findings": ["...", ...], '
        '"differential": [{{"diagnosis": "...", "icd10": "...", "likelihood": "high|moderate|low"}}, ...]}}'
    ),
)

IMAGING_STRUCTURED = PromptTemplate(
    system=IMAGING_SYSTEM,
    user=(
        "{structured_hints}"
        "IMAGING ORDER:\n"
        "  Modality: {modality}\n"
        "  Body Region: {body_region}\n"
        "  Clinical Indication: {clinical_indication}\n\n"
        "PATIENT CLINICAL RECORD:\n{ehr_text}\n\n"
        "Follow this exact JSON schema:\n"
        '{{\n'
        '  "clinical_question": "<inferred clinical question>",\n'
        '  "pre_read_summary": "<3-5 sentence summary for radiologist>",\n'
        '  "must_include_findings": ["<finding 1>", "<finding 2>", ...],\n'
        '  "differential": [\n'
        '    {{"diagnosis": "<name>", "icd10": "<code>", "likelihood": "high|moderate|low"}}\n'
        '  ]\n'
        '}}\n\n'
        f"{_JSON_CLOSING}"
    ),
)


# ============================================================================
# Registry
# ============================================================================

PROMPT_REGISTRY: dict[str, dict[str, PromptTemplate]] = {
    "patient_diagnosis": {
        "zero_shot": PATIENT_DIAGNOSIS_ZERO_SHOT,
        "few_shot": PATIENT_DIAGNOSIS_FEW_SHOT,
        "cot": PATIENT_DIAGNOSIS_COT,
        "structured": PATIENT_DIAGNOSIS_STRUCTURED,
    },
    "context_summarization": {
        "zero_shot": SUMMARIZATION_ZERO_SHOT,
        "few_shot": SUMMARIZATION_FEW_SHOT,
        "cot": SUMMARIZATION_COT,
        "structured": SUMMARIZATION_STRUCTURED,
    },
    "context_summarization_current_visit": {
        "zero_shot": CURRENT_VISIT_ZERO_SHOT,
        "few_shot": CURRENT_VISIT_FEW_SHOT,
        "cot": CURRENT_VISIT_COT,
        "structured": CURRENT_VISIT_STRUCTURED,
    },
    "context_summarization_specialty": {
        "zero_shot": SPECIALTY_ZERO_SHOT,
        "few_shot": SPECIALTY_FEW_SHOT,
        "cot": SPECIALTY_COT,
        "structured": SPECIALTY_STRUCTURED,
    },
    "evidence_retrieval": {
        "zero_shot": RETRIEVAL_ZERO_SHOT,
        "few_shot": RETRIEVAL_FEW_SHOT,
        "cot": RETRIEVAL_COT,
        "structured": RETRIEVAL_STRUCTURED,
    },
    "imaging_indication": {
        "zero_shot": IMAGING_ZERO_SHOT,
        "few_shot": IMAGING_FEW_SHOT,
        "cot": IMAGING_COT,
        "structured": IMAGING_STRUCTURED,
    },
}


def get_prompt(task: str, strategy: str) -> PromptTemplate:
    """Get the prompt template for a task and strategy."""
    if task not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(PROMPT_REGISTRY)}")
    strategies = PROMPT_REGISTRY[task]
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(strategies)}")
    return strategies[strategy]
