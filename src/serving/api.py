"""
api.py
──────
Athenium — FastAPI Inference Endpoint

Serves the merged fine-tuned model for real-time contract risk classification.
Returns structured risk assessment with optional attention attribution.

Endpoints:
    POST /classify  — Classify a contract clause or document
    GET  /health    — Service health and model metadata
    GET  /docs      — Auto-generated Swagger UI (FastAPI default)

Start:
    python -m athenium.src.serving.api
    → http://localhost:8000
    → http://localhost:8000/docs
"""

import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

RISK_LABELS    = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
MODEL_PATH     = "./athenium-merged"
ESCALATE_CONF  = 0.85
MAX_LENGTH     = 512


# ── Schemas ───────────────────────────────────────────────────────────────────

class ContractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        max_length=8192,
        description="Contract clause or document text. Truncated to 512 tokens internally.",
        example=(
            "The Borrower shall not, without prior written consent of the Agent, "
            "declare or pay any dividend if an Event of Default is continuing "
            "or would result therefrom."
        ),
    )
    return_attribution: bool = Field(
        False,
        description=(
            "Return token-level attention attribution — which tokens most influenced "
            "the classification decision. Adds ~15ms latency."
        ),
    )


class TokenAttribution(BaseModel):
    token:  str
    weight: float


class ClassificationResponse(BaseModel):
    risk_label:          str
    confidence:          float
    label_proba:         dict[str, float]
    escalate_for_review: bool
    latency_ms:          float
    attribution:         Optional[list[TokenAttribution]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "risk_label": "HIGH",
                "confidence": 0.9231,
                "label_proba": {"LOW": 0.004, "MEDIUM": 0.061, "HIGH": 0.923, "CRITICAL": 0.012},
                "escalate_for_review": False,
                "latency_ms": 118.4,
                "attribution": [
                    {"token": "Event",    "weight": 0.142},
                    {"token": "Default",  "weight": 0.138},
                    {"token": "dividend", "weight": 0.091},
                ],
            }
        }
    }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Athenium — Contract Risk Classification API",
    description=(
        "Real-time financial contract risk classification. "
        "Fine-tuned Mistral-7B on financial contract corpus. "
        "Returns structured risk label, confidence, and optional attention attribution."
    ),
    version="1.0.0",
    license_info={"name": "MIT"},
)

_tokeniser = None
_model     = None


def get_model():
    """Lazy-load on first request. Subsequent calls return cached objects."""
    global _tokeniser, _model
    if _model is None:
        _tokeniser = AutoTokenizer.from_pretrained(MODEL_PATH)
        _tokeniser.pad_token = _tokeniser.eos_token
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _model.eval()
    return _tokeniser, _model


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ContractRequest):
    """
    Classify a financial contract clause.

    Risk levels:
      LOW      — Standard terms, no unusual provisions
      MEDIUM   — Non-standard clauses, standard legal review warranted
      HIGH     — Significant risk provisions, legal counsel recommended
      CRITICAL — Immediate escalation required

    Escalation: If confidence < 0.85, escalate_for_review=true.
    Route these contracts to the analyst queue.
    """
    t0 = time.perf_counter()

    tokeniser, model = get_model()
    inputs = tokeniser(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=request.return_attribution)

    logits = outputs.logits[0]
    proba  = torch.softmax(logits, dim=-1).cpu().float().numpy()
    pred   = int(proba.argmax())
    conf   = float(proba.max())

    attribution = None
    if request.return_attribution and outputs.attentions:
        # Average across all layers and all heads → (seq, seq)
        stacked  = torch.stack(outputs.attentions, dim=0)   # (layers, B, heads, S, S)
        mean_attn = stacked.mean(dim=(0, 1, 2))[0]           # (S, S)
        cls_attn  = mean_attn[0].cpu().float().numpy()       # (S,) — CLS row

        tokens = tokeniser.convert_ids_to_tokens(inputs["input_ids"][0])
        ranked = sorted(
            [(tok, float(w)) for tok, w in zip(tokens, cls_attn)
             if tok not in ["<s>", "</s>", "<pad>"]],
            key=lambda x: x[1], reverse=True,
        )
        attribution = [
            TokenAttribution(token=tok, weight=round(w, 4))
            for tok, w in ranked[:15]
        ]

    latency_ms = (time.perf_counter() - t0) * 1000

    return ClassificationResponse(
        risk_label=RISK_LABELS[pred],
        confidence=round(conf, 4),
        label_proba={RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        escalate_for_review=(conf < ESCALATE_CONF),
        latency_ms=round(latency_ms, 2),
        attribution=attribution,
    )


@app.get("/health")
async def health():
    """Service health check."""
    try:
        _, model = get_model()
        device = str(next(model.parameters()).device)
        status = "ok"
    except Exception as e:
        device, status = "unknown", f"degraded: {e}"

    return {
        "status":               status,
        "model_path":           MODEL_PATH,
        "device":               device,
        "escalation_threshold": ESCALATE_CONF,
        "max_input_tokens":     MAX_LENGTH,
        "risk_labels":          RISK_LABELS,
    }


if __name__ == "__main__":
    uvicorn.run(
        "athenium.src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
