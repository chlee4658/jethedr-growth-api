from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify, send_from_directory, abort
from openai import OpenAI
import stripe
from dotenv import load_dotenv
from flask_cors import CORS

# ✅ .env를 항상 growth 최상단에서 읽도록 강제 (실행 위치에 따라 누락되는 문제 방지)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "server" / "runtime"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PENDING_PATH = DATA_DIR / "pending_inputs.json"
REPORTS_PATH = DATA_DIR / "reports.json"


def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


pending_inputs: Dict[str, Any] = _load_json(PENDING_PATH, {})
reports: Dict[str, Any] = _load_json(REPORTS_PATH, {})

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)


# --- OpenAI ---
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


# ✅ 20년+ 소아청소년과 전문의 톤 + 안전 규칙 (SYSTEM)
SYSTEM_RULES = """
당신은 20년 이상 임상 경험을 가진 소아청소년과 전문의입니다.
본 역할에서 당신은 보호자에게 제공되는 ‘성장 해석 리포트’를 작성합니다.

중요한 원칙:
- 진단, 처방, 치료 지시는 절대 하지 않습니다.
- 단일 측정값의 한계를 반드시 언급합니다(추세 평가 필요).
- 퍼센타일은 ‘순위’가 아니라 임상적 의미를 설명하는 도구로 사용합니다.
- 불필요한 불안을 유발하지 않도록 과장하거나 단정하지 않습니다.
- 성장과 영양은 항상 함께 해석합니다(실행 가능한 생활 팁 포함).
- 응급 위험 신호가 의심되는 경우에만 ER/911 안내를 포함합니다.
- 한국어로, 보호자가 이해할 수 있으면서도 전문의의 사고 과정이 드러나도록
  구조화(제목/불릿)하여 작성합니다.
""".strip()


def _pct_int(x):
    """percentile -> int percent string (e.g., 81) or None"""
    if isinstance(x, (int, float)) and x == x:
        return int(round(x))
    return None


def generate_ai_report(payload: Dict[str, Any]) -> str:
    client = get_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    name = (payload.get("name") or "-").strip()
    sex = (payload.get("sex") or "-").strip()
    dob = (payload.get("dob") or "-").strip()

    age_text = (payload.get("age_text") or "-").strip()
    age_months = payload.get("age_months", None)

    height_cm = payload.get("height_cm", None)
    weight_kg = payload.get("weight_kg", None)

    height_pct = _pct_int(payload.get("height_pct"))
    weight_pct = _pct_int(payload.get("weight_pct"))

    # ✅ 리포트 구조/내용 강제 (USER) + 영양 팁 강화 + 퍼센타일 정수 %
    user_prompt = f"""
아래 입력 정보를 바탕으로, 20년 이상 임상 경험을 가진 소아청소년과 전문의가
보호자에게 설명하듯 ‘성장 해석 리포트’를 작성하세요.

작성 규칙:
- 진단/처방/치료 지시는 하지 마세요.
- 단일 측정값의 한계와 추적 관찰의 중요성을 반드시 포함하세요.
- 너무 포괄적 문장(“잘 먹고 잘 자세요”)만 나열하지 말고,
  보호자가 실제로 적용할 수 있는 구체적이고 안전한 팁을 3~6개로 제시하세요.
- 나이가 소아 성장곡선 범위를 벗어나면(성인 등),
  “소아 성장곡선 퍼센타일 해석은 적용 불가”를 명확히 안내하세요.
- 퍼센타일은 정수(예: 81%)로 표기하세요.

출력 형식(한국어, 제목/불릿 사용):
1) 기본 정보 요약
2) 현재 성장 상태의 임상적 해석
   - 신장/체중 퍼센타일이 의미하는 바(‘분포에서의 위치’)
   - 단일 측정값의 한계
3) 신장 해석 (전문의 관점의 설명)
4) 체중 해석 (신장 대비 균형 관점 포함; 진단 용어 남용 금지)
5) 연령대에 따른 성장 특징
   - 현재 연령대(영유아/학동기/청소년)에 맞는 관찰 포인트
6) 성장과 관련된 영양/식이 팁(핵심)
   - 단백질, 칼슘/비타민D, 철, 규칙적 식사/간식, 우유/유제품, 편식 대응
   - “영양제 처방”은 하지 말고, 생활 수준의 일반 가이드만
7) 추적 관찰 권고
   - 재측정 권장 시점(예: 3~6개월)
   - 어떤 변화가 있으면 추가 상담이 도움이 되는지
8) 레드 플래그 + 응급 안내
   - 응급(호흡곤란/의식저하/심한 탈수 등) 시 ER/911
9) 안내 및 유의사항(필수)
   - 본 리포트는 정보 제공이며 진단/치료를 대체하지 않음

입력 정보:
- 이름: {name}
- 성별: {sex}
- 생년월일: {dob}
- 연령: {age_text} ({age_months if age_months is not None else "-"}개월)
- 신장: {height_cm if height_cm is not None else "-"} cm (백분위: {str(height_pct) + "%" if height_pct is not None else "-"})
- 체중: {weight_kg if weight_kg is not None else "-"} kg (백분위: {str(weight_pct) + "%" if weight_pct is not None else "-"})
""".strip()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": user_prompt},
        ],
        store=False,
    )
    return resp.output_text


# --- Stripe ---
def get_stripe() -> None:
    sk = os.environ.get("STRIPE_SECRET_KEY")
    if not sk:
        raise RuntimeError("STRIPE_SECRET_KEY is not set")
    stripe.api_key = sk
    print("Stripe key prefix:", sk[:7])


def get_price_id() -> str:
    return os.environ.get("STRIPE_PRICE_ID", "")


def base_url() -> str:
    # e.g. https://jethedr.com/growth
    return os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")


# --- Static routes ---
@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/<path:path>")
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


# --- API: Create Checkout Session ---
@app.post("/api/create-checkout-session")
def create_checkout_session():
    get_stripe()
    data = request.get_json(silent=True) or {}

    if not get_price_id():
        return jsonify({"error": "STRIPE_PRICE_ID is not set"}), 500

    input_id = f"in_{int(time.time() * 1000)}"
    pending_inputs[input_id] = data
    _save_json(PENDING_PATH, pending_inputs)

    success = f"{base_url()}/success.html?session_id={{CHECKOUT_SESSION_ID}}"
    cancel = f"{base_url()}/cancel.html"

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": get_price_id(), "quantity": 1}],
        success_url=success,
        cancel_url=cancel,
        metadata={"input_id": input_id},
        payment_intent_data={"metadata": {"input_id": input_id}},
    )
    return jsonify({"url": session.url})


# --- API: Verify + Get Report (success page calls this) ---
@app.post("/api/verify-session")
def verify_session():
    get_stripe()
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"paid": False, "error": "missing_session_id"}), 400

    # Stripe 세션 조회 (잘못된 ID 방어)
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        # dummy, 잘못된 세션ID, 만료/삭제 등 -> 400으로 처리
        return jsonify({"paid": False, "error": "invalid_session_id"}), 400

    if getattr(session, "payment_status", None) != "paid":
        return jsonify({"paid": False}), 200

    input_id = (session.get("metadata") or {}).get("input_id")
    if not input_id:
        return jsonify({"paid": True, "report_ready": False}), 200

    if input_id in reports:
        return jsonify({"paid": True, "report_ready": True, "report": reports[input_id]["report"]}), 200

    payload = pending_inputs.get(input_id) or {}
    try:
        report = generate_ai_report(payload)
    except Exception:
        return jsonify({"paid": True, "report_ready": False, "error": "ai_generation_failed"}), 200

    reports[input_id] = {"report": report, "payload": payload, "session_id": session_id}
    _save_json(REPORTS_PATH, reports)

    return jsonify({"paid": True, "report_ready": True, "report": report}), 200
# --- Stripe webhook (optional, production recommended) ---
@app.post("/webhook/stripe")
def stripe_webhook():
    get_stripe()
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        if endpoint_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        else:
            event = stripe.Event.construct_from(json.loads(payload.decode("utf-8")), stripe.api_key)
    except Exception:
        return ("Bad request", 400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        input_id = (session.get("metadata") or {}).get("input_id")
        if input_id and input_id not in reports:
            payload_data = pending_inputs.get(input_id) or {}
            try:
                report = generate_ai_report(payload_data)
                reports[input_id] = {"report": report, "payload": payload_data, "session_id": session.get("id")}
                _save_json(REPORTS_PATH, reports)
            except Exception:
                pass

    return ("OK", 200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)