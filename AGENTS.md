# Agent guide — Synthetic Hospital mock EMR

Fully synthetic longitudinal EHR (1,268 patients, zero PHI) behind an Epic-style API. Use for chart browsing, clinical-agent tool loops, and benchmark eval — **not clinical care**.

## Base URL

Default compose: `http://localhost:8000` (override with `APP_PORT`).

- Chart UI (humans): `GET /`
- OpenAPI: `GET /docs`
- Health: `GET /health` → `{"status":"ok","redis":true}`

## Auth (required for chart + tools)

Register once, then mint a JWT:

```bash
BASE=http://localhost:8000

curl -s -X POST "$BASE/auth/register" \
  -H 'Content-Type: application/json' \
  -d '{"username":"dr.demo","password":"demo1234","role":"attending","display_name":"Demo Attending","department":"Internal Medicine"}'

TOKEN=$(curl -s -X POST "$BASE/auth/token" \
  -H 'Content-Type: application/json' \
  -d '{"username":"dr.demo","password":"demo1234"}' \
  | jq -r .access_token)

# All chart/tool calls:
-H "Authorization: Bearer $TOKEN"
```

Valid roles: `attending | resident | nurse | radiologist | lab_tech | pharmacist`.  
**Section visibility is RBAC-filtered by role** — prefer `attending` unless testing access control.

## Preferred agent path: tool protocol

1. `GET /agent/tools` — schema discovery (no auth).
2. Optional: `POST /epic/sessions` with `{patient_id?, role, department, max_api_calls}` for rate tracking.
3. `POST /agent/tools` with Bearer token:

```json
{
  "session_id": "<optional>",
  "tool_name": "open_chart",
  "arguments": { "patient_id": 1672 }
}
```

### Tools (read)

| Tool | Arguments |
|---|---|
| `search_patients` | `name?`, `gender?`, `mrn?` |
| `open_chart` | `patient_id` — demog, problems, recent encounters, meds, allergies |
| `view_encounters` | `patient_id` |
| `view_encounter_detail` | `encounter_id` — note sections (HPI, PMH, …) |
| `view_section` | `section_id` |
| `view_results` | `patient_id`, `result_type` (`labs` \| `imaging` \| `pathology`) |
| `search_chart` | `patient_id`, `query` |
| `view_problem_list` | `patient_id` |
| `view_medications` | `patient_id` |

### Tools (submit / scored tasks)

| Tool | Arguments |
|---|---|
| `submit_diagnosis` | `patient_id`, `diagnosis_name`, `icd10_code?` |
| `submit_summary` | `patient_id`, `summary` |
| `submit_pre_read` | `order_id`, `summary`, `impression?`, `findings?` |
| `submit_rankings` | `patient_id`, `rankings:[{passage_id, grade:0-3}]` |

Typical read loop: `search_patients` → `open_chart` → `view_encounters` → `view_encounter_detail` → `search_chart` / `view_results` → `submit_*` if evaluating.

## REST shortcuts (same auth)

```bash
curl -s "$BASE/fhir/Patient?_count=10" -H "Authorization: Bearer $TOKEN"
curl -s "$BASE/fhir/Patient/1672" -H "Authorization: Bearer $TOKEN"
curl -s "$BASE/epic/chart/1672/summary" -H "Authorization: Bearer $TOKEN"
curl -s "$BASE/epic/chart/1672/encounters" -H "Authorization: Bearer $TOKEN"
curl -s "$BASE/epic/chart/1672/encounters/6803" -H "Authorization: Bearer $TOKEN"
```

Patient IDs are integers (MRN ≈ FHIR `Patient.id`). Release names are often `Patient <id>` — search by MRN when possible.

## Eval CLI

```bash
docker compose run --rm app python -m eval.cli --help
# Needs OPENROUTER_API_KEY for live model runs
docker compose run --rm app python -m eval.cli run \
  --task patient_diagnosis --model <model> --split public --strategy cot
```

Splits: `public` (reported), `heldout` (do not train on), `train`.

## Constraints

- Synthetic only — no PHI; not for clinical decisions.
- No full Hyperspace UI — API + thin chart browser; prefer `/agent/tools` over browser automation unless the task is UI-specific.
- Wrong role → empty/partial sections.
- SNOMED/LOINC optional; ICD-10 loads on boot. SapBERT semantic search may be absent (`search_chart` still does lexical search).
