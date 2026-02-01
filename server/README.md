# v30.12 Stripe + AI Growth Report

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

## Environment
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
export STRIPE_SECRET_KEY="sk_live_or_test_..."
export STRIPE_PRICE_ID="price_1SvB3JGVKbat4rwH6ACMMPrf"
export BASE_URL="http://127.0.0.1:8000"
# Optional for production webhook verification:
export STRIPE_WEBHOOK_SECRET="whsec_..."
```

## Run
```bash
python server/app.py
```

Open:
- http://127.0.0.1:8000/index.html
- http://127.0.0.1:8000/admin.html
- After checkout: success.html will fetch the paid AI report.

## Stripe Webhook (local dev)
Use Stripe CLI to forward webhooks:
```bash
stripe listen --forward-to 127.0.0.1:8000/webhook/stripe
```
Then set STRIPE_WEBHOOK_SECRET to the printed whsec_...
