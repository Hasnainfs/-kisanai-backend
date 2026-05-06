from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests, time, json, os
from groq import Groq

app = FastAPI(title="KisanAI Production API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FAO_USERNAME = os.environ.get("FAO_USERNAME", "")
FAO_PASSWORD = os.environ.get("FAO_PASSWORD", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

class FaoAuth:
    def __init__(self):
        self._token = None
        self._ts = 0
        self._ttl = 3000

    def token(self):
        if self._token and self._token != "FAILED" and time.time() - self._ts < self._ttl:
            return self._token
        return self._login()

    def _login(self):
        try:
            r = requests.post(
                "https://faostatservices.fao.org/api/v1/auth/login",
                data={"username": FAO_USERNAME, "password": FAO_PASSWORD},
                timeout=15
            )
            resp = r.json()
            token = (resp.get("access_token") or resp.get("AccessToken") or
                     resp.get("token") or resp.get("id_token") or
                     (resp.get("AuthenticationResult") or {}).get("AccessToken"))
            if token:
                self._token = token
                self._ts = time.time()
                print("✅ FAO connected!")
                return self._token
            self._token = "FAILED"
            self._ts = time.time()
            return None
        except:
            self._token = "FAILED"
            self._ts = time.time()
            return None

fao_auth = FaoAuth()

CROP_CODES = {
    "wheat": 15, "gandum": 15, "rice": 27, "chawal": 27,
    "cotton": 328, "kapas": 328, "sugarcane": 156, "ganna": 156,
    "maize": 56, "makkai": 56, "tomato": 388, "tamatar": 388,
    "potato": 116, "aloo": 116, "onion": 397, "pyaz": 397,
}

def get_fao_context(question):
    token = fao_auth.token()
    if not token or token == "FAILED":
        return ""
    code = next((v for k, v in CROP_CODES.items() if k in question.lower()), 15)
    try:
        r = requests.get(
            "https://faostatservices.fao.org/api/v1/en/data/QCL",
            headers={"Authorization": f"Bearer {token}"},
            params={"area": 106, "item": code, "element": "2510,5510,5312",
                    "year": "2021,2022", "outputType": "objects"},
            timeout=15
        )
        rows = r.json().get("data", [])[:5]
        if not rows:
            return ""
        lines = [f"  • {row.get('Element','')}: {row.get('Value','N/A')} {row.get('Unit','')} ({row.get('Year','')})" for row in rows]
        return "\n📊 FAOSTAT Real Pakistan Data:\n" + "\n".join(lines)
    except:
        return ""

@app.get("/")
def root():
    return {"message": "KisanAI API Live! 🌾"}

@app.get("/health")
def health():
    return {"status": "online", "model": "llama-3.3-70b-versatile",
            "fao": "connected" if fao_auth._token not in [None, "FAILED"] else "groq-only", "groq": "ready"}

class ChatReq(BaseModel):
    question: str
    crop: str = "wheat"
    topic: str = "general"
    history: list = []

@app.post("/chat")
def chat(req: ChatReq):
    fao_ctx = get_fao_context(req.question)
    system = f"""You are KisanAI — expert bilingual Agricultural AI for Pakistani farmers.
{fao_ctx}
Context: crop={req.crop}, topic={req.topic}, region=Pakistan, season=2026
Rules: 1)Reply same language as farmer 2)Pakistani brands: DAP,FFC Urea,Roko,Topsin,Tilt 3)Prices in ₨ 4)**bold** key terms 5)Bullets for lists 6)Max 300 words"""

    messages = [{"role": "system", "content": system}]
    for h in req.history[-14:]:
        messages.append(h)
    messages.append({"role": "user", "content": req.question})

    def stream():
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=messages,
                max_tokens=1024, temperature=0.7, stream=True)
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'Error: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    fao_auth.token()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
