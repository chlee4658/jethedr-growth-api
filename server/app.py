from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, request, jsonify, send_from_directory
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


def _pct_int(x) -> Optional[int]:
    """percentile -> int (e.g., 81) or None"""
    try:
        if isinstance(x, (int, float)) and x == x:
            return int(round(float(x)))
        if isinstance(x, str) and x.strip() != "":
            v = float(x.strip().replace("%", ""))
            if v == v:
                return int(round(v))
    except Exception:
        pass
    return None


def _extract_percentiles(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Returns: (height_kr, height_us, weight_kr, weight_us)

    Supports both:
    - flat keys: height_pct_kr, height_pct_us, weight_pct_kr, weight_pct_us
    - nested keys: height:{kr_percentile/us_percentile}, weight:{kr_percentile/us_percentile}
    Also falls back to legacy keys: height_pct, weight_pct (treated as KR if KR is missing).
    """
    # flat keys
    h_kr = _pct_int(payload.get("height_pct_kr"))
    h_us = _pct_int(payload.get("height_pct_us"))
    w_kr = _pct_int(payload.get("weight_pct_kr"))
    w_us = _pct_int(payload.get("weight_pct_us"))

    # nested (app.js localStorage 구조)
    h_obj = payload.get("height") if isinstance(payload.get("height"), dict) else {}
    w_obj = payload.get("weight") if isinstance(payload.get("weight"), dict) else {}

    if h_kr is None:
        h_kr = _pct_int(h_obj.get("kr_percentile"))
    if h_us is None:
        h_us = _pct_int(h_obj.get("us_percentile"))

    if w_kr is None:
        w_kr = _pct_int(w_obj.get("kr_percentile"))
    if w_us is None:
        w_us = _pct_int(w_obj.get("us_percentile"))

    # legacy fallback
    if h_kr is None:
        h_kr = _pct_int(payload.get("height_pct"))
    if w_kr is None:
        w_kr = _pct_int(payload.get("weight_pct"))

    return h_kr, h_us, w_kr, w_us


def _judge_by_pct(p: Optional[int]) -> str:
    if p is None:
        return "-"
    if p < 3:
        return "낮은 편(관찰)"
    if p > 97:
        return "높은 편(관찰)"
    return "정상 범위"


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

    # ✅ KR/US percentiles (int)
    height_pct_kr, height_pct_us, weight_pct_kr, weight_pct_us = _extract_percentiles(payload)

    # ✅ 리포트 상단에 prepend할 “전문가 느낌” 1~2줄
    # (OpenAI가 본문 안에서 자연스럽게 활용하도록, 프롬프트에 "0) 백분위수 요약"을 강제)
    pct_summary_lines = []
    pct_summary_lines.append(
        f"백분위수 요약(참고): 신장 KR {str(height_pct_kr) + '%' if height_pct_kr is not None else '-'} / "
        f"US {str(height_pct_us) + '%' if height_pct_us is not None else '-'}, "
        f"체중 KR {str(weight_pct_kr) + '%' if weight_pct_kr is not None else '-'} / "
        f"US {str(weight_pct_us) + '%' if weight_pct_us is not None else '-'}"
    )
    # 두 기준이 다를 때 짧은 한 줄 코멘트 유도(없으면 모델이 알아서 생략 가능)
    if (height_pct_kr is not None and height_pct_us is not None and height_pct_kr != height_pct_us) or \
       (weight_pct_kr is not None and weight_pct_us is not None and weight_pct_kr != weight_pct_us):
        pct_summary_lines.append("※ KR/US 기준 값 차이는 기준집단/측정체계 차이로 달라질 수 있어 ‘추세(변화)’와 함께 해석하는 것이 중요합니다.")

    pct_header = "\n".join(pct_summary_lines).strip()

    # ✅ 리포트 구조/내용 강제 (USER)
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
- KR(한국)과 US(미국) 기준 백분위수가 모두 제공된 경우,
  0) 항목에 반드시 “신장/체중: KR xx%, US yy%” 형태로 1~2줄 요약을 먼저 넣고,
  본문에서도 필요 시 기준 차이를 간단히(불안 유발 없이) 설명하세요.

출력 형식(한국어, 제목/불릿 사용):
0) 백분위수 요약(전문의 메모 스타일, 1~2줄)
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

0) 항목에 반드시 아래 요약을 그대로 포함해 시작하세요:
{pct_header}

입력 정보:
- 이름: {name}
- 성별: {sex}
- 생년월일: {dob}
- 연령: {age_text} ({age_months if age_months is not None else "-"}개월)
- 신장: {height_cm if height_cm is not None else "-"} cm (KR 백분위: {str(height_pct_kr) + "%" if height_pct_kr is not None else "-"} / US 백분위: {str(height_pct_us) + "%" if height_pct_us is not None else "-"})
- 체중: {weight_kg if weight_kg is not None else "-"} kg (KR 백분위: {str(weight_pct_kr) + "%" if weight_pct_kr is not None else "-"} / US 백분위: {str(weight_pct_us) + "%" if weight_pct_us is not None else "-"})
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
    except Exception:
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

# 관리자용 openai 호출
@app.post("/api/generate_report")
def generate_report():
    payload = request.get_json(silent=True) or {}
    try:
        report = generate_ai_report(payload)
        return jsonify({"report": report}), 200
    except Exception as e:
        return jsonify({"error": "ai_generation_failed", "detail": str(e)}), 500


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