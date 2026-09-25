"""Export a random sample of model summaries for qualitative inspection.

For each of the 11 models, dumps its summary for 3 shared unconditioned (whole-patient)
items and 3 shared specialty-conditioned (involved) items, alongside the ground-truth
context. Same items across all models so outputs are directly comparable. Seeded.

Output: sample_summaries_all_models.csv
"""
import csv
import json
import random

import psycopg
from epic_sim.app.config import settings
from eval.tasks.summarization import load_inputs

SEED, N = 42, 3
OUT = "sample_summaries_all_models.csv"

# latest structured whole-patient summarization run per model (n=200)
UNCOND = {"gpt-5.3": 306, "opus-4.6": 307, "gemini-3.1": 308, "deepseek-v3.2": 310,
          "qwen-3": 311, "mistral-3": 314, "llama-4": 315, "gemma-3": 316,
          "glm-5": 317, "kimi-2.5": 312, "kimi-2.5-thinking": 313}
# specialty-conditioned run per model (n=983); GLM via the gateway (glm-5-agent)
SPEC = {"gpt-5.3": 295, "opus-4.6": 296, "gemini-3.1": 297, "deepseek-v3.2": 305,
        "qwen-3": 299, "mistral-3": 300, "llama-4": 301, "gemma-3": 302,
        "glm-5-agent": 293, "kimi-2.5": 309, "kimi-2.5-thinking": 291}
DISPLAY = {"gpt-5.3": "GPT 5.3", "opus-4.6": "Opus 4.6", "gemini-3.1": "Gemini 3.1",
           "deepseek-v3.2": "DeepSeek V3.2", "qwen-3": "Qwen 3", "mistral-3": "Mistral 3",
           "llama-4": "Llama 4", "gemma-3": "Gemma 3", "glm-5": "GLM 5",
           "glm-5-agent": "GLM 5", "kimi-2.5": "Kimi 2.5", "kimi-2.5-thinking": "Kimi 2.5-thinking"}
# stable model display order
ORDER = ["GPT 5.3", "Opus 4.6", "Gemini 3.1", "Kimi 2.5-thinking", "Kimi 2.5",
         "GLM 5", "DeepSeek V3.2", "Qwen 3", "Mistral 3", "Llama 4", "Gemma 3"]


def summ(p):
    p = json.loads(p) if isinstance(p, str) else p
    s = p.get("summary", "") if isinstance(p, dict) else p
    return s if isinstance(s, str) else (json.dumps(s) if s else "")


def names(findings):
    out = []
    for f in findings or []:
        if isinstance(f, dict):
            out.append(f.get("display_name") or f.get("name") or "")
        else:
            out.append(str(f))
    return "; ".join(x for x in out if x)


def preds(c, run, gt_ids):
    return {g: summ(p) for g, p in c.execute(
        "SELECT gt_id, prediction FROM evaluation_predictions WHERE run_id=%s AND gt_id = ANY(%s)",
        (run, gt_ids))}


def main():
    c = psycopg.connect(settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"))
    rng = random.Random(SEED)

    # --- pick the shared sample items ---
    uncond_inp = load_inputs(c, split="val")  # unconditioned whole-patient
    spec_inp = [i for i in load_inputs(c, granularity="specialty", tier="small")
                if i.ground_truth.get("involvement") in ("involved", "high", "low")
                or i.ground_truth.get("tiers", {}).get("primary")]
    uncond_pick = sorted(rng.sample([i.gt_id for i in uncond_inp], N))
    spec_pick = sorted(rng.sample([i.gt_id for i in spec_inp], N))
    uctx = {i.gt_id: i for i in uncond_inp}
    sctx = {i.gt_id: i for i in spec_inp}

    rows = []
    # unconditioned
    for model, run in UNCOND.items():
        pr = preds(c, run, uncond_pick)
        for g in uncond_pick:
            inp = uctx[g]
            rows.append(["unconditioned", g, inp.patient_id, "", "",
                         inp.clinical_question, names(inp.must_include_findings),
                         DISPLAY[model], pr.get(g, "<no prediction>")])
    # specialty-conditioned (involved)
    for model, run in SPEC.items():
        pr = preds(c, run, spec_pick)
        for g in spec_pick:
            inp = sctx[g]
            gt = inp.ground_truth
            rows.append(["specialty_conditioned", g, inp.patient_id, inp.specialty,
                         gt.get("involvement", ""), inp.clinical_question,
                         names(gt.get("tiers", {}).get("primary")),
                         DISPLAY[model], pr.get(g, "<no prediction>")])

    # sort: variant, gt_id, model order
    rank = {m: i for i, m in enumerate(ORDER)}
    rows.sort(key=lambda r: (r[0], r[1], rank.get(r[7], 99)))

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "gt_id", "patient_id", "specialty", "involvement",
                    "clinical_question", "reference_findings", "model", "summary"])
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows "
          f"({len(UNCOND)} models x {N} unconditioned + {len(SPEC)} x {N} conditioned)")
    print(f"unconditioned gt_ids: {uncond_pick}")
    print(f"specialty gt_ids:     {spec_pick}")


if __name__ == "__main__":
    main()
