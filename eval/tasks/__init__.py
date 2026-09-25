"""Task-specific data loaders, prompt formatters, and output parsers."""

from eval.tasks import diagnosis, imaging, patient_diagnosis, retrieval, summarization

TASK_LOADERS = {
    "patient_diagnosis": patient_diagnosis.load_inputs,
    "context_summarization": summarization.load_inputs,
    "evidence_retrieval": retrieval.load_inputs,
    "imaging_indication": imaging.load_inputs,
}

TASK_PARSERS = {
    "patient_diagnosis": patient_diagnosis.parse_output,
    "context_summarization": summarization.parse_output,
    "evidence_retrieval": retrieval.parse_output,
    "imaging_indication": imaging.parse_output,
}

TASK_FORMATTERS = {
    "patient_diagnosis": patient_diagnosis.format_prompt,
    "context_summarization": summarization.format_prompt,
    "evidence_retrieval": retrieval.format_prompt,
    "imaging_indication": imaging.format_prompt,
}
