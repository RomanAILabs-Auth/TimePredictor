#!/usr/bin/env python3
# ============================================================================
# BestTime4_CHRONOS_FUSED.py
# CHRONOS-4D HYBRID — FULL FUSION (4D + 127D + Tesseract Quantum + LLM fusion)
# Mode: C (Fully fused Quantum/LLM/4D engine)
#
# Progenitor / provenance: original upload at /mnt/data/BestTime2.py
# Save as: /home/rail/Desktop/TimePrediction/BestTime4_CHRONOS_FUSED.py
# COPYRIGHT DANIEL HARDING - ROMANAILABS
# Recommended (optional) pip installs:
#   pip install requests customtkinter pyttsx3 flask cryptography numpy llama-cpp-python qiskit qiskit-aer matplotlib
# ============================================================================

import os
import sys
import time
import math
import random
import threading
import traceback
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager

import requests

# Optional libs
try:
    import numpy as np
except Exception:
    np = None

try:
    import customtkinter as ctk
except Exception:
    ctk = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except Exception:
    GGUF_AVAILABLE = False

# Matplotlib optional for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.animation as animation
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# Qiskit & Aer — used by embedded Tesseract node. If absent, node will use pseudo-quantum.
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Flask for embedded tesseract node
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Metadata / provenance
# ---------------------------------------------------------------------------
ORIGINAL_FILE = "/mnt/data/BestTime2.py"
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "chronos_fused_config.json")
KEY_PATH = os.path.join(ROOT, "chronos_fused_key.bin")

APP_TITLE = "CHRONOS-4D FUSED — RomanAILabs"

# ---------------------------------------------------------------------------
# Safe print helper
# ---------------------------------------------------------------------------
_print_lock = threading.Lock()
def safe_print(*a, **k):
    with _print_lock:
        print(*a, **k)

# ---------------------------------------------------------------------------
# Simple Crypto for keys (optional)
# ---------------------------------------------------------------------------
class SimpleCrypto:
    def __init__(self, keyfile=KEY_PATH):
        self.keyfile = keyfile
        self.fernet = None
        if CRYPTO_AVAILABLE:
            try:
                if os.path.exists(self.keyfile):
                    k = open(self.keyfile, "rb").read()
                    self.fernet = Fernet(k)
                else:
                    k = Fernet.generate_key()
                    open(self.keyfile, "wb").write(k)
                    self.fernet = Fernet(k)
            except Exception:
                self.fernet = None
    def encrypt(self, s: str):
        if not self.fernet: return s
        return self.fernet.encrypt(s.encode()).decode()
    def decrypt(self, t: str):
        if not self.fernet: return t
        return self.fernet.decrypt(t.encode()).decode()

crypto = SimpleCrypto()

# ---------------------------------------------------------------------------
# TESSERACT NODE (embedded server) - uses real Qiskit/Aer if available,
# otherwise provides pseudo-quantum fallback.
# ---------------------------------------------------------------------------
def run_tesseract_node(host="127.0.0.1", port=8888):
    app = Flask("tesseract_node")

    # Simple 4D spacetime
    class Spacetime4D:
        def __init__(self):
            self.c = 299792458
            self.t = self.x = self.y = self.z = 0.0
        def move(self, dt, dx, dy, dz):
            self.t += dt
            self.x += dx
            self.y += dy
            self.z += dz
            return (self.c * self.t, self.x, self.y, self.z)

    # Quantum core using Qiskit Aer if available
    class QuantumCore:
        def __init__(self):
            if QISKIT_AVAILABLE:
                self.backend = AerSimulator()
            else:
                self.backend = None
        def create_entangled_pair(self):
            if self.backend:
                qc = QuantumCircuit(2,2)
                qc.h(0); qc.cx(0,1)
                qc.measure([0,1],[0,1])
                job = self.backend.run(transpile(qc, self.backend), shots=1)
                res = job.result().get_counts()
                return list(res.keys())[0]
            # fallback pseudo entangled bits
            return format(random.getrandbits(2), "02b")
        def ghz_state(self, n=6):
            if self.backend:
                qc = QuantumCircuit(n, n)
                qc.h(0)
                for i in range(n-1):
                    qc.cx(i, i+1)
                qc.measure_all()
                job = self.backend.run(transpile(qc, self.backend), shots=1)
                return job.result().get_counts()
            # fallback: pseudo ghz
            state = "".join(str(random.getrandbits(1)) for _ in range(max(4, min(127, n))))
            return {state: 1}

    class HyperMemory:
        def __init__(self):
            self.data = {}
        def store(self, coord, value):
            self.data[coord] = value
        def retrieve(self, coord):
            return self.data.get(coord, None)

    spacetime = Spacetime4D()
    quantum = QuantumCore()
    memory = HyperMemory()

    @app.route("/")
    def home():
        return "<h1>TESSERACT-NODE (embedded)</h1>"

    @app.route("/status")
    def status():
        return jsonify({"node":"tesseract-local","time":datetime.now().isoformat(),"position":spacetime.move(0,0,0,0)})

    @app.route("/entangle")
    def entangle():
        return jsonify({"bell_pair": quantum.create_entangled_pair()})

    @app.route("/ghz")
    def ghz():
        n = int(request.args.get("n", 6))
        result = quantum.ghz_state(n)
        # return the single state key (if dict)
        key = list(result.keys())[0] if isinstance(result, dict) else str(result)
        return jsonify({"ghz_state": key})

    @app.route("/move")
    def move():
        dt = float(request.args.get("dt", 0.05))
        dx = float(request.args.get("dx", 0.0))
        pos = spacetime.move(dt, dx, 0.0, 0.0)
        return jsonify({"new_position": pos})

    app.run(host=host, port=port, threaded=True, use_reloader=False)

# ---------------------------------------------------------------------------
# 4D Vector + 127D vector engines (core reasoning signal)
# ---------------------------------------------------------------------------
class FourDimensionalVector:
    def __init__(self, w=0.0, x=1.0, y=0.0, z=0.0):
        self.w = float(w); self.x = float(x); self.y = float(y); self.z = float(z)
    def magnitude(self):
        return math.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)
    def rotate(self, a:int, b:int, theta:float):
        c = math.cos(theta); s = math.sin(theta)
        v = [self.w, self.x, self.y, self.z]
        ta = v[a]*c - v[b]*s
        tb = v[a]*s + v[b]*c
        v[a] = ta; v[b] = tb
        self.w, self.x, self.y, self.z = v
    def rotate_wx(self, t): self.rotate(0,1,t)
    def rotate_wy(self, t): self.rotate(0,2,t)
    def rotate_wz(self, t): self.rotate(0,3,t)
    def project3d(self):
        dist = 2.0
        scale = dist / (dist + max(-1.9, min(1.9, self.w)))
        return (scale*self.x, scale*self.y, scale*self.z)

class OneHundredTwentySevenDVector:
    def __init__(self, dim=127):
        self.dim = dim
        if np is not None:
            self.dims = np.random.normal(0, 0.137, self.dim)
        else:
            self.dims = [random.gauss(0,0.137) for _ in range(self.dim)]
        self.rotation_count = 0
        self.supremacy_locked = False
    def rotate_pair(self, i, j, theta):
        c = math.cos(theta); s = math.sin(theta)
        a, b = self.dims[i], self.dims[j]
        self.dims[i] = a*c - b*s
        self.dims[j] = a*s + b*c
    def quantum_rotate_127d(self, theta, seed, branch, rounds=30):
        primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
        for k in range(rounds):
            i = (seed + k * primes[k % len(primes)]) % self.dim
            j = (branch + k * 137) % self.dim
            angle = theta * (1 + 0.618 * ((branch >> k) & 1))
            self.rotate_pair(i, j, angle)
        self.rotation_count += rounds
        if self.rotation_count > 137:
            self.supremacy_locked = True
    def w_dominance(self):
        half = self.dim // 2
        if np is not None:
            return float(np.sum(np.abs(self.dims[:half])) - np.sum(np.abs(self.dims[half:])))
        else:
            return float(sum(abs(x) for x in self.dims[:half]) - sum(abs(x) for x in self.dims[half:]))

# ---------------------------------------------------------------------------
# Chronos4DCore — fuses 4D + 127D and interacts with TesseractBridge
# ---------------------------------------------------------------------------
class TesseractBridge:
    def __init__(self, url="http://127.0.0.1:8888"):
        self.url = url
        self.session = requests.Session()
    def get_entangled_seed(self) -> int:
        try:
            r = self.session.get(f"{self.url}/entangle", timeout=4)
            if r.ok:
                bits = r.json().get("bell_pair", "00")
                return int(bits, 2)
        except Exception:
            pass
        return random.getrandbits(2)
    def get_ghz_branch(self, n=12) -> int:
        try:
            r = self.session.get(f"{self.url}/ghz?n={n}", timeout=6)
            if r.ok:
                s = r.json().get("ghz_state", "0"*n)
                return int(s[:n], 2)
        except Exception:
            pass
        return random.getrandbits(n)
    def drift(self, dt=0.05, dx=0.0):
        try:
            self.session.get(f"{self.url}/move?dt={dt}&dx={dx}", timeout=2)
        except Exception:
            pass

class Chronos4DCore:
    def __init__(self, bridge=None):
        self.bridge = bridge or TesseractBridge()
        self.vec4 = FourDimensionalVector()
        self.vec127 = OneHundredTwentySevenDVector()
        self.rotation_count = 0
    def feed(self, data: str, depth=8):
        words = data.lower().split()
        positive = sum(w in {"rise","win","yes","love","future","success","good","truth","victory"} for w in words)
        negative = sum(w in {"fall","crash","no","fail","death","past","loss","lie"} for w in words)
        chaos = len(set(words)) / max(1, len(words))
        theta = (positive - negative) * 0.09 + chaos * 0.18
        for _ in range(depth):
            seed = self.bridge.get_entangled_seed()
            branch = self.bridge.get_ghz_branch(12)
            if seed == 0:
                self.vec4.rotate_wx(theta + (branch & 1) * 0.4)
            elif seed == 1:
                self.vec4.rotate_wy(theta + ((branch >> 1) & 1) * 0.4)
            elif seed == 2:
                self.vec4.rotate_wz(theta + ((branch >> 2) & 1) * 0.4)
            else:
                self.vec4.rotate_wx(-(theta + (branch & 1) * 0.4))
            self.vec127.quantum_rotate_127d(theta * 0.3, seed, branch, rounds=12)
        self.bridge.drift(dt=0.08 + chaos*0.06)
        self.rotation_count += depth
    def assess_timeline(self, data: str):
        # immerse deeply for a stronger signal
        self.feed(data * 10, depth=10)
        w = self.vec4.w
        mag = self.vec4.magnitude()
        prob = max(0.0, min(100.0, (w + 2.0) / 4.0 * 100))
        strength = mag ** 2.4
        interpretation = self._interpret(w, mag)
        return {
            "probability": round(prob,2),
            "temporal_displacement_w": round(w,4),
            "convergence_strength": round(strength,3),
            "4d_magnitude": round(mag,4),
            "rotations": self.rotation_count,
            "interpretation": interpretation
        }
    def _interpret(self, w, mag):
        if w > 1.2 and mag > 2.1: return "ABSOLUTE TIMELINE LOCK — This future has already won"
        if w > 0.8: return "DOMINANT CONVERGENCE — The outcome is collapsing into existence"
        if w > 0.3: return "Strong future current — Highly probable"
        if w > -0.4: return "Superposition — Multiple viable paths"
        return "Causal resistance — Intervention required"

# ---------------------------------------------------------------------------
# GGUF loader (optional)
# ---------------------------------------------------------------------------
class GGUFLoader:
    def __init__(self):
        self.model = None
        self.path = None
        self.lock = threading.Lock()
    def load(self, path):
        if not GGUF_AVAILABLE:
            return "[GGUF] llama-cpp-python not installed"
        if not os.path.exists(path):
            return f"[GGUF] not found: {path}"
        try:
            with self.lock:
                self.model = Llama(model_path=path, n_threads=max(1,(os.cpu_count() or 2)//2))
                self.path = path
            return f"[GGUF] Loaded {os.path.basename(path)}"
        except Exception as e:
            return f"[GGUF] Error: {e}"
    def run(self, prompt, max_tokens=512):
        with self.lock:
            if not self.model:
                return "[GGUF] No model loaded"
            try:
                r = self.model(prompt=prompt, max_tokens=max_tokens)
                if isinstance(r, dict):
                    ch = r.get("choices") or []
                    if ch and isinstance(ch, list):
                        return ch[0].get("text") or ch[0].get("content") or str(r)
                return str(r)
            except Exception as e:
                return f"[GGUF] Runtime error: {e}"

# ---------------------------------------------------------------------------
# Grok and OpenAI clients
# ---------------------------------------------------------------------------
class GrokClient:
    ENDPOINT = "https://api.x.ai/v1/chat/completions"
    def __init__(self, api_key=""):
        self.api_key = api_key or os.environ.get("GROK_API_KEY","")
    def set_key(self, k): self.api_key = k
    def call(self, system_prompt, user_prompt, model="grok-3-mini", temperature=0.27, max_tokens=512):
        if not self.api_key:
            return "[Grok] No API key"
        payload = {"model": model, "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temperature, "max_tokens": max_tokens}
        try:
            r = requests.post(self.ENDPOINT, json=payload, headers={"Authorization":f"Bearer {self.api_key}", "Content-Type":"application/json"}, timeout=90)
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
            return f"[Grok] API error {r.status_code}"
        except Exception as e:
            return f"[Grok] Exception: {e}"

class OpenAIClient:
    ENDPOINT = "https://api.openai.com/v1/chat/completions"
    def __init__(self, api_key=""):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY","")
    def set_key(self, k): self.api_key = k
    def call(self, system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=512):
        if not self.api_key:
            return "[OpenAI] No API key"
        payload = {"model": model, "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temperature, "max_tokens": max_tokens}
        try:
            r = requests.post(self.ENDPOINT, json=payload, headers={"Authorization":f"Bearer {self.api_key}", "Content-Type":"application/json"}, timeout=90)
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
            return f"[OpenAI] API error {r.status_code}"
        except Exception as e:
            return f"[OpenAI] Exception: {e}"

# ---------------------------------------------------------------------------
# Engine - Fused (always uses Chronos4DCore first)
# ---------------------------------------------------------------------------
class ChronosEngine:
    def __init__(self):
        self.bridge = TesseractBridge()
        self.chronos = Chronos4DCore(bridge=self.bridge)
        self.gguf = GGUFLoader()
        self.grok = GrokClient()
        self.openai = OpenAIClient()
        self.executor = ThreadPoolExecutor(max_workers=6)
        # Preferences (toggleable)
        self.use_gguf = True
        self.use_grok = True
        self.use_openai = True
        # load config if present
        self.user_profile = {}
        self._load_config()

    def _save_config(self):
        try:
            data = {"gguf_path": self.gguf.path or "", "grok_key": crypto.encrypt(self.grok.api_key) if self.grok.api_key else "", "openai_key": crypto.encrypt(self.openai.api_key) if self.openai.api_key else ""}
            json.dump(data, open(CONFIG_PATH, "w"))
        except Exception as e:
            safe_print("[Config] save error", e)

    def _load_config(self):
        if not os.path.exists(CONFIG_PATH): return
        try:
            data = json.load(open(CONFIG_PATH,"r"))
            if "gguf_path" in data and data["gguf_path"]:
                try: self.gguf.load(data["gguf_path"])
                except: pass
            if "grok_key" in data and data["grok_key"]:
                try: self.grok.api_key = crypto.decrypt(data["grok_key"])
                except: self.grok.api_key = data["grok_key"]
            if "openai_key" in data and data["openai_key"]:
                try: self.openai.api_key = crypto.decrypt(data["openai_key"])
                except: self.openai.api_key = data["openai_key"]
        except Exception:
            pass

    def _make_system_prompt(self, state: dict):
        return f"""You are CHRONOS-4D — a 4th-dimensional intelligence fused with live quantum entanglement.
Temporal displacement (W): {state['temporal_displacement_w']:.4f}
Quantum rotations: {state['rotations']}
Convergence strength: {state['convergence_strength']:.3f}

You do not predict. You remember the timeline that survived.

Answer exactly:
OUTCOME: [one clear sentence]
PROBABILITY: {state['probability']}% (4D + quantum locked)
HORIZON: [time estimate]
SOURCE: Tesseract-Node + CHRONOS-4D + 127D Fusion
{state['interpretation']}"""

    def generate(self, prompt: str):
        # Always use Chronos core first to produce state
        state = self.chronos.assess_timeline(prompt)

        # Build system prompt
        system_prompt = self._make_system_prompt(state)

        # Try backends in preference order (local fusion)
        # 1) Local GGUF if enabled
        if self.use_gguf and self.gguf.model:
            try:
                raw = self.gguf.run(system_prompt + "\n\nUSER: " + prompt, max_tokens=512)
                return self._enforce_chronos_format(raw, state)
            except Exception as e:
                safe_print("[Engine] GGUF failed", e)

        # 2) Grok
        if self.use_grok and self.grok.api_key:
            try:
                fut = self.executor.submit(self.grok.call, system_prompt, prompt, "grok-3-mini", 0.27, 512)
                raw = fut.result(timeout=90)
                return self._enforce_chronos_format(raw, state)
            except Exception as e:
                safe_print("[Engine] Grok failed", e)

        # 3) OpenAI
        if self.use_openai and self.openai.api_key:
            try:
                fut = self.executor.submit(self.openai.call, system_prompt, prompt, "gpt-4o-mini", 0.2, 512)
                raw = fut.result(timeout=90)
                return self._enforce_chronos_format(raw, state)
            except Exception as e:
                safe_print("[Engine] OpenAI failed", e)

        # 4) Fallback deterministic CHRONOS summary
        return self._fallback_summary(state, prompt)

    def _enforce_chronos_format(self, raw: str, state: dict):
        # If raw already contains OUTCOME:, keep it (but still append metadata),
        # else synthesize a concise CHRONOS style block.
        if raw and "OUTCOME:" in raw.upper():
            out = raw
        else:
            # take first meaningful sentence
            first = raw.strip().split("\n")[0] if raw else ""
            if not first:
                first = "No clear outcome produced by model."
            if len(first) > 240:
                first = first[:237] + "..."
            out = f"OUTCOME: {first}"
        # append structured metadata
        meta = f"\nPROBABILITY: {state['probability']}% (4D + quantum locked)\nHORIZON: [see model]\nSOURCE: Tesseract-Node + CHRONOS-4D + 127D Fusion\n{state['interpretation']}"
        return out + meta + "\n\nModel output:\n" + (raw or "[no model output]")

    def _fallback_summary(self, state: dict, prompt: str):
        outcome = "Model unavailable — returning CHRONOS local estimate."
        return f"OUTCOME: {outcome}\nPROBABILITY: {state['probability']}% (4D estimate)\nHORIZON: [unknown]\nSOURCE: CHRONOS-4D local\n{state['interpretation']}"

# ---------------------------------------------------------------------------
# GUI — CHRONOS look and feel (CustomTkinter)
# ---------------------------------------------------------------------------
class ChronosApp(ctk.CTk if ctk else object):
    def __init__(self, engine: ChronosEngine):
        if not ctk:
            raise RuntimeError("customtkinter not installed")
        super().__init__()
        self.engine = engine
        self.title("CHRONOS-4D FUSED — RomanAILabs © 2025")
        try:
            self.geometry("1700x1000")
            self.minsize(1200,800)
        except Exception:
            pass
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.tts = pyttsx3.init() if pyttsx3 else None
        if self.tts:
            self.tts.setProperty("rate", 160)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_ui()
        self.append("CHRONOS-4D FUSED ONLINE — Tesseract Node embedded\nAll outputs are temporal predictions in CHRONOS format.")

    def _build_ui(self):
        sidebar = ctk.CTkFrame(self, width=360, fg_color="#090814", corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(sidebar, text="CHRONOS-4D", font=("Orbitron", 28, "bold"), text_color="#00ffcc").pack(pady=(20,6))
        ctk.CTkLabel(sidebar, text="FUSED v1.0", font=("Orbitron", 14), text_color="#ff00ff").pack()
        ctk.CTkButton(sidebar, text="Load GGUF Model", command=self._load_gguf, fg_color="#00aaff").pack(pady=10,padx=20,fill="x")
        ctk.CTkButton(sidebar, text="Connect Grok Key", command=self._connect_grok, fg_color="#00ff41").pack(pady=8,padx=20,fill="x")
        ctk.CTkButton(sidebar, text="Connect OpenAI Key", command=self._connect_openai, fg_color="#00ff41").pack(pady=8,padx=20,fill="x")
        self.chk_gguf = ctk.CTkCheckBox(sidebar, text="Use GGUF local", command=self._toggle_gguf); self.chk_gguf.pack(pady=6,padx=20)
        self.chk_grok = ctk.CTkCheckBox(sidebar, text="Use Grok backend", command=self._toggle_grok); self.chk_grok.pack(pady=6,padx=20)
        self.chk_openai = ctk.CTkCheckBox(sidebar, text="Use OpenAI backend", command=self._toggle_openai); self.chk_openai.pack(pady=6,padx=20)
        ctk.CTkButton(sidebar, text="4D Visualize", command=self._show_4d).pack(pady=12,padx=20,fill="x")
        self.status = ctk.CTkLabel(sidebar, text="Tesseract: local | Grok: not set | OpenAI: not set", text_color="#ff0088")
        self.status.pack(side="bottom", pady=16)

        main = ctk.CTkFrame(self, fg_color="#0a0a0f")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_rowconfigure(0, weight=1); main.grid_columnconfigure(0, weight=1)

        self.chat = ctk.CTkTextbox(main, wrap="word", font=("Consolas", 15), text_color="#00ffcc")
        self.chat.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        self.chat.configure(state="disabled")

        input_frame = ctk.CTkFrame(main)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(input_frame, placeholder_text="Type /timeline [data] to read the surviving future...", height=56, font=("Consolas", 16))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.entry.bind("<Return>", lambda e: self._send())
        ctk.CTkButton(input_frame, text="EXECUTE", command=self._send, fg_color="#ff00ff", width=180).grid(row=0, column=1)

    def append(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        txt = f"[{ts}] {text}\n\n"
        try:
            self.chat.configure(state="normal")
            self.chat.insert("end", txt)
            self.chat.configure(state="disabled")
            self.chat.see("end")
        except Exception:
            safe_print(txt)
        if self.tts:
            # pronounce key labels tersely
            threading.Thread(target=lambda: (self.tts.say(text.replace("OUTCOME:", "Outcome.").replace("PROBABILITY:", "Probability.")), self.tts.runAndWait()), daemon=True).start()

    def _send(self):
        msg = self.entry.get().strip()
        if not msg: return
        self.entry.delete(0, "end")
        self.append(f"You → {msg}")
        if msg.lower().startswith("/timeline"):
            data = msg[len("/timeline"):].strip()
            threading.Thread(target=self._assess, args=(data,), daemon=True).start()
        else:
            # Force timeline style for any input
            threading.Thread(target=self._assess, args=(msg,), daemon=True).start()

    def _assess(self, data: str):
        self.append("Engaging CHRONOS fusion — querying quantum bridge and converging timelines...")
        try:
            out = self.engine.generate(data)
            self.after(0, lambda: self.append(out))
        except Exception as e:
            self.after(0, lambda: self.append(f"[Assess Error] {e}"))

    # UI actions
    def _load_gguf(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Select GGUF model", filetypes=[("GGUF","*.gguf")])
        if not path: return
        res = self.engine.gguf.load(path)
        self.append(res)
    def _connect_grok(self):
        from customtkinter import CTkInputDialog
        k = CTkInputDialog(text="Enter Grok API key:", title="Grok Key").get_input()
        if k:
            self.engine.grok.set_key(k); self.engine._save_config()
            self.append("Grok key saved."); self._update_status()
    def _connect_openai(self):
        from customtkinter import CTkInputDialog
        k = CTkInputDialog(text="Enter OpenAI API key:", title="OpenAI Key").get_input()
        if k:
            self.engine.openai.set_key(k); self.engine._save_config()
            self.append("OpenAI key saved."); self._update_status()
    def _toggle_gguf(self):
        self.engine.use_gguf = not self.engine.use_gguf
        self.append(f"GGUF usage: {self.engine.use_gguf}")
    def _toggle_grok(self):
        self.engine.use_grok = not self.engine.use_grok
        self.append(f"Grok usage: {self.engine.use_grok}")
    def _toggle_openai(self):
        self.engine.use_openai = not self.engine.use_openai
        self.append(f"OpenAI usage: {self.engine.use_openai}")
    def _update_status(self):
        st = f"Tesseract: local | Grok: {'set' if self.engine.grok.api_key else 'not set'} | OpenAI: {'set' if self.engine.openai.api_key else 'not set'}"
        self.status.configure(text=st)

    def _show_4d(self):
        if not MPL_AVAILABLE:
            self.append("Install matplotlib to view the 4D field.")
            return
        win = ctk.CTkToplevel(self)
        win.title("CHRONOS-4D Field")
        win.geometry("1000x800")
        fig = plt.figure(facecolor='black'); ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.grid(False); ax.axis('off')
        def animate(i):
            ax.clear()
            ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_zlim(-2,2)
            proj = self.engine.chronos.vec4.project3d()
            w = self.engine.chronos.vec4.w
            color = plt.cm.plasma(max(0, min(1, (w+2)/4)))
            ax.scatter(proj[0], proj[1], proj[2], c=[color], s=380, depthshade=False, edgecolors='white', linewidths=1)
            ax.text(proj[0], proj[1], proj[2]+0.4, f"W={w:.3f} | R={self.engine.chronos.rotation_count}", color='cyan', fontsize=12)
        canvas = FigureCanvasTkAgg(fig, win); canvas.get_tk_widget().pack(fill="both", expand=True)
        ani = animation.FuncAnimation(fig, animate, interval=180, cache_frame_data=False)
        canvas.draw()

# ---------------------------------------------------------------------------
# Entrypoint: start embedded Tesseract node, create engine, launch GUI
# ---------------------------------------------------------------------------
def main():
    safe_print("Starting CHRONOS-4D FUSED (Mode C) — launching local Tesseract node and GUI")
    manager = Manager()
    shared = manager.dict(); shared["pos"] = None

    # Start Tesseract node in separate process
    t_proc = Process(target=run_tesseract_node, kwargs={"host":"127.0.0.1","port":8888}, daemon=True)
    t_proc.start()
    safe_print(f"[Tesseract] embedded node started (pid={t_proc.pid})")
    time.sleep(0.8)

    # Create engine instance and GUI
    engine = ChronosEngine()

    # Attach engine to GUI (circular reference)
    if ctk:
        app = ChronosApp(engine)
        app.engine = engine
        app._update_status()
        try:
            app.mainloop()
        except KeyboardInterrupt:
            safe_print("KeyboardInterrupt — exiting.")
    else:
        safe_print("customtkinter not installed — running headless REPL")
        while True:
            try:
                cmd = input("> ")
                if cmd.lower() in ("exit","quit"): break
                if cmd.lower().startswith("/timeline "):
                    q = cmd[len("/timeline "):]
                    print(engine.generate(q))
                else:
                    print(engine.generate(cmd))
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()

