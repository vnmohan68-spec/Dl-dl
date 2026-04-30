import streamlit as st
import numpy as np
import json
from PIL import Image, ImageFilter, ImageStat
import os
import base64
import colorsys
from io import BytesIO
import streamlit.components.v1 as components

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH   = "plant_disease_model.keras"
CLASSES_PATH = "plant_classes.json"

# ══════════════════════════════════════════════════════════════════════════════
#  STRICT PREDICTION CONSTANTS  — only edit this block
# ══════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESH = 0.92   # top-1 must beat 92% (very strict)
CONFIDENCE_GAP    = 0.25   # top1 − top2 gap must exceed 25%
TEMP_SCALE        = 0.5    # softmax temperature (lower = sharper)
GREEN_RATIO_MIN   = 0.20   # green-dominant pixel fraction (RGB check)
LEAF_HSV_MIN      = 0.20   # plant-green pixel fraction (HSV hue range check)
BLUR_THRESHOLD    = 80.0   # Laplacian variance; below = blurry
BRIGHTNESS_MIN    = 35     # mean brightness; below = too dark
BRIGHTNESS_MAX    = 230    # mean brightness; above = overexposed

# Only these classes are valid — anything else = rejected
VALID_PLANTS = ["pepper", "potato", "tomato"]
# ══════════════════════════════════════════════════════════════════════════════

# ─── LOAD RESOURCES ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    if not os.path.exists(CLASSES_PATH):
        return [], 224
    with open(CLASSES_PATH) as f:
        data = json.load(f)
    return data["classes"], data["img_size"]

def format_label(label: str) -> str:
    return label.replace("_", " ").title()

def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ══════════════════════════════════════════════════════════════════════════════
#  SMART PREDICTION GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def temperature_softmax(logits: np.ndarray) -> np.ndarray:
    """Lower temperature = sharper, more decisive probabilities."""
    scaled = logits / TEMP_SCALE
    exp    = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()

def check_brightness(pil_image: Image.Image) -> bool:
    """Reject too dark or overexposed images."""
    mean = ImageStat.Stat(pil_image.convert("L")).mean[0]
    return BRIGHTNESS_MIN <= mean <= BRIGHTNESS_MAX

def check_blur(pil_image: Image.Image) -> bool:
    """Reject blurry images using Laplacian variance."""
    gray = pil_image.convert("L").resize((128, 128))
    lap  = gray.filter(ImageFilter.Kernel(
               size=3, kernel=[-1,-1,-1,-1,8,-1,-1,-1,-1],
               scale=1, offset=0))
    return float(np.array(lap, dtype=np.float32).var()) >= BLUR_THRESHOLD

def check_green_rgb(pil_image: Image.Image) -> bool:
    """RGB check: green channel must dominate in enough pixels."""
    arr     = np.array(pil_image.convert("RGB").resize((64, 64)), dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask    = (G > R * 1.05) & (G > B * 1.05)
    return float(mask.sum() / mask.size) >= GREEN_RATIO_MIN

def check_leaf_hsv(pil_image: Image.Image) -> bool:
    """
    HSV check: pixels must fall in plant-green hue range.
    Hue 60°–180° (0.17–0.50) = yellow-green to cyan-green.
    This rejects brown bark, blue sky, skin tones, concrete etc.
    Even other green plants (coconut, banana) with very different
    hue profiles or low saturation are caught here.
    """
    arr         = np.array(pil_image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    leaf_pixels = 0
    total       = arr.shape[0] * arr.shape[1]
    for row in arr:
        for r, g, b in row:
            h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            if 0.17 <= h <= 0.50 and s > 0.15 and v > 0.10:
                leaf_pixels += 1
    return (leaf_pixels / total) >= LEAF_HSV_MIN

def check_valid_plant_class(label: str) -> bool:
    """
    Hard class whitelist — prediction MUST belong to
    pepper / potato / tomato. Any other class = rejected.
    """
    label_lower = label.lower()
    return any(plant in label_lower for plant in VALID_PLANTS)

def smart_predict(pil_image: Image.Image, model, class_names: list) -> dict:
    """
    6-layer guard system + class whitelist.
    Returns: {status, reason, label, confidence, top_probs}
    """
    def reject(reason):
        return {"status": "rejected", "reason": reason,
                "label": "", "confidence": 0.0, "top_probs": []}

    # Guard 1 — Brightness
    if not check_brightness(pil_image):
        return reject("Image is too dark or overexposed. Use a well-lit photo.")

    # Guard 2 — Blur
    if not check_blur(pil_image):
        return reject("Image is too blurry. Upload a clear, focused photo.")

    # Guard 3 — RGB green check
    if not check_green_rgb(pil_image):
        return reject(
            "No plant leaf detected. Upload only Tomato, Potato "
            "or Bell Pepper leaf images.")

    # Guard 4 — HSV hue check (catches skin, wood, concrete, non-leaf greens)
    if not check_leaf_hsv(pil_image):
        return reject(
            "Image does not appear to be a plant leaf. Upload only "
            "Tomato, Potato or Bell Pepper leaf images.")

    # Model inference
    img_arr = np.array(
        pil_image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    raw   = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
    probs = temperature_softmax(raw)

    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_label = class_names[top_idx]

    # Guard 5 — Confidence threshold
    if top_conf < CONFIDENCE_THRESH:
        return reject(
            f"Model confidence too low ({top_conf*100:.1f}%). "
            "Upload a clear, close-up Tomato / Potato / Bell Pepper leaf.")

    # Guard 6 — Confidence gap (prevents Tomato vs Pepper confusion)
    sorted_p = np.sort(probs)[::-1]
    gap      = float(sorted_p[0] - sorted_p[1])
    if gap < CONFIDENCE_GAP:
        return reject(
            "Model is uncertain between plant classes. "
            "Upload a clearer, closer leaf photo.")

    # Guard 7 — Class whitelist (hard block on non-target plants)
    if not check_valid_plant_class(top_label):
        return reject(
            "This plant is not in the supported list. "
            "Only Tomato, Potato and Bell Pepper are supported.")

    # All guards passed ✓
    top3_idx = np.argsort(probs)[::-1][:3]
    return {
        "status":     "ok",
        "reason":     "",
        "label":      top_label,
        "confidence": top_conf,
        "top_probs":  [(class_names[i], float(probs[i])) for i in top3_idx],
    }

# ══════════════════════════════════════════════════════════════════════════════

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;700;900&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #020810 !important;
    font-family: 'Exo 2', sans-serif !important;
    overflow-x: hidden;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto;
}

/* ── Animated mesh background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 15% 20%, rgba(0,255,120,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 50% 70% at 85% 75%, rgba(0,180,255,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 80% 40% at 50% 50%, rgba(80,0,255,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 30% 30% at 70% 20%, rgba(0,255,200,0.06) 0%, transparent 60%);
    animation: meshPulse 10s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

/* ── Grid lines ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,120,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,120,0.04) 1px, transparent 1px);
    background-size: 70px 70px;
    animation: gridDrift 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes meshPulse {
    0%   { opacity: .7; transform: scale(1); }
    50%  { opacity: 1; transform: scale(1.03); }
    100% { opacity: .8; transform: scale(1.01); }
}
@keyframes gridDrift {
    0%   { background-position: 0 0; }
    100% { background-position: 70px 70px; }
}

/* ── Upload widget ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(0,255,120,0.03) !important;
    border: 2px dashed rgba(0,255,120,0.35) !important;
    border-radius: 20px !important;
    transition: all 0.4s ease !important;
    animation: dropPulse 3s ease-in-out infinite !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(0,255,120,0.07) !important;
    border-color: #00ff80 !important;
    box-shadow: 0 0 50px rgba(0,255,120,0.2), inset 0 0 50px rgba(0,255,120,0.03) !important;
}
@keyframes dropPulse {
    0%,100% { box-shadow: 0 0 20px rgba(0,255,120,0.05); }
    50%      { box-shadow: 0 0 40px rgba(0,255,120,0.15); }
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: rgba(0,255,120,0.75) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(0,255,120,0.25) !important;
    box-shadow: 0 0 50px rgba(0,0,0,0.7), 0 0 25px rgba(0,255,120,0.08) !important;
    transition: all 0.4s ease !important;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 70px rgba(0,0,0,0.8), 0 0 40px rgba(0,255,120,0.2) !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ff80, #00cfff, #7c3aed) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 12px rgba(0,255,120,0.6) !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00ff80 !important; }

/* ── Markdown text ── */
.stMarkdown p { color: rgba(200,220,210,0.8) !important; line-height: 1.7; }
.stMarkdown h3 {
    color: #00ff80 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: .08em;
}

/* ── Alert overrides ── */
[data-testid="stAlert"] {
    background: rgba(0,255,120,0.04) !important;
    border: 1px solid rgba(0,255,120,0.25) !important;
    border-radius: 14px !important;
    color: rgba(180,255,200,0.9) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00ff80, #00cfff);
    border-radius: 3px;
}

/* ── Custom HTML sections ── */
.phyto-section {
    position: relative;
    z-index: 2;
    animation: fadeUp .8s cubic-bezier(.22,1,.36,1) both;
}
@keyframes fadeUp {
    from { opacity:0; transform: translateY(28px); }
    to   { opacity:1; transform: translateY(0); }
}
.phyto-section:nth-child(2) { animation-delay:.15s; }
.phyto-section:nth-child(3) { animation-delay:.3s;  }

.glass {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(0,255,120,0.14);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 60px rgba(0,0,0,.5), inset 0 0 60px rgba(0,0,0,.2);
    transition: border-color .4s, box-shadow .4s, transform .4s;
}
.glass:hover {
    border-color: rgba(0,255,120,.3);
    box-shadow: 0 4px 80px rgba(0,0,0,.6), 0 0 40px rgba(0,255,120,.08);
    transform: translateY(-3px);
}
/* scan line */
.glass::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,120,.6), transparent);
    animation: scan 4s linear infinite;
}
@keyframes scan { to { left: 200%; } }

.tag {
    display: inline-block;
    padding: .25rem 1rem;
    border-radius: 999px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .72rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    border: 1px solid;
}
.tag-green { color: #00ff80; border-color: rgba(0,255,120,.5); background: rgba(0,255,120,.08); }
.tag-red   { color: #ff4d6d; border-color: rgba(255,77,109,.5); background: rgba(255,77,109,.08); }
.tag-blue  { color: #00cfff; border-color: rgba(0,207,255,.5); background: rgba(0,207,255,.08); }

.big-num {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 4rem;
    line-height: 1;
    background: linear-gradient(135deg, #00ff80 0%, #00cfff 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0,255,120,.4));
}

.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem;
    letter-spacing: .22em;
    color: rgba(150,200,170,.6);
    text-transform: uppercase;
    margin-top: .35rem;
}

.bar-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .73rem;
    color: rgba(180,210,195,.7);
    letter-spacing: .05em;
}

.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,120,.4), rgba(0,207,255,.4), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* 3D tilt card */
.tilt-card {
    transform-style: preserve-3d;
    transition: transform .15s ease;
}
</style>
""", unsafe_allow_html=True)

# ─── 3D THREE.JS HERO ──────────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background: transparent; overflow: hidden; }
  canvas { display:block; }
  #overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    pointer-events: none;
  }
  .title {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: clamp(2.4rem, 6vw, 5rem);
    letter-spacing: .06em;
    background: linear-gradient(135deg, #00ff80 0%, #00cfff 45%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(0,255,120,.5));
    animation: titlePulse 3s ease-in-out infinite alternate;
  }
  .sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: clamp(.65rem, 1.5vw, .9rem);
    letter-spacing: .35em;
    color: rgba(0,255,120,.55);
    text-transform: uppercase;
    margin-top: .6rem;
    animation: subFade 4s ease-in-out infinite alternate;
  }
  .pills {
    display: flex; gap: .8rem; margin-top: 1.2rem; flex-wrap: wrap; justify-content: center;
  }
  .pill {
    font-family: 'Share Tech Mono', monospace;
    font-size: .62rem;
    letter-spacing: .15em;
    padding: .3rem .9rem;
    border-radius: 999px;
    border: 1px solid rgba(0,255,120,.3);
    color: rgba(0,255,120,.7);
    background: rgba(0,255,120,.06);
    text-transform: uppercase;
    animation: pillFloat 3s ease-in-out infinite alternate;
  }
  .pill:nth-child(2) { animation-delay:.4s; border-color:rgba(0,207,255,.3); color:rgba(0,207,255,.7); background:rgba(0,207,255,.06); }
  .pill:nth-child(3) { animation-delay:.8s; border-color:rgba(168,85,247,.3); color:rgba(168,85,247,.7); background:rgba(168,85,247,.06); }
  @keyframes titlePulse { 0%{filter:drop-shadow(0 0 20px rgba(0,255,120,.4));} 100%{filter:drop-shadow(0 0 50px rgba(0,207,255,.7));} }
  @keyframes subFade    { 0%{opacity:.4;} 100%{opacity:.8;} }
  @keyframes pillFloat  { 0%{transform:translateY(0);} 100%{transform:translateY(-5px);} }
  @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@900&family=Share+Tech+Mono&display=swap');
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="overlay">
  <div class="title">PHYTO SCAN AI</div>
  <div class="sub">Neural Plant Disease Detection · v3.0</div>
  <div class="pills">
    <span class="pill">🌶 Bell Pepper</span>
    <span class="pill">🥔 Potato</span>
    <span class="pill">🍅 Tomato</span>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), alpha: true, antialias: true });
renderer.setSize(W, H);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
camera.position.z = 30;

const COUNT = 600;
const geo   = new THREE.BufferGeometry();
const pos   = new Float32Array(COUNT * 3);
const col   = new Float32Array(COUNT * 3);
const vel   = new Float32Array(COUNT * 3);
const palettes = [
  [0, 1, 0.47],
  [0, 0.81, 1],
  [0.66, 0.33, 0.98],
];
for (let i = 0; i < COUNT; i++) {
  pos[i*3]   = (Math.random() - .5) * 120;
  pos[i*3+1] = (Math.random() - .5) * 80;
  pos[i*3+2] = (Math.random() - .5) * 60;
  vel[i*3]   = (Math.random() - .5) * .02;
  vel[i*3+1] = (Math.random() - .5) * .02;
  vel[i*3+2] = (Math.random() - .5) * .01;
  const c = palettes[Math.floor(Math.random() * palettes.length)];
  col[i*3] = c[0]; col[i*3+1] = c[1]; col[i*3+2] = c[2];
}
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));

const mat = new THREE.PointsMaterial({ size: .35, vertexColors: true, transparent: true, opacity: .75 });
const pts = new THREE.Points(geo, mat);
scene.add(pts);

function makeRing(radius, y, color, opacity) {
  const rGeo = new THREE.RingGeometry(radius, radius + .12, 80);
  const rMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(rGeo, rMat);
  mesh.position.y = y;
  mesh.rotation.x = Math.PI / 2;
  return mesh;
}
const rings = [];
const ringColors = [0x00ff80, 0x00cfff, 0xa855f7, 0x00ff80, 0x00cfff];
for (let i = 0; i < 5; i++) {
  const r = makeRing(6 + i * 2.5, (i - 2) * 6, ringColors[i], .25);
  scene.add(r);
  rings.push(r);
}

function makeHex(size, pos, color) {
  const g = new THREE.CylinderGeometry(size, size, .05, 6);
  const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .15, wireframe: true });
  const h = new THREE.Mesh(g, m);
  h.position.set(...pos);
  return h;
}
const hexes = [
  makeHex(4, [-18, 5, -10], 0x00ff80),
  makeHex(3, [20, -6, -8],  0x00cfff),
  makeHex(5, [8, 10, -15],  0xa855f7),
];
hexes.forEach(h => scene.add(h));

let t = 0;
function animate() {
  requestAnimationFrame(animate);
  t += .008;
  const p = geo.attributes.position.array;
  for (let i = 0; i < COUNT; i++) {
    p[i*3]   += vel[i*3];
    p[i*3+1] += vel[i*3+1] + Math.sin(t + i) * .003;
    p[i*3+2] += vel[i*3+2];
    if (Math.abs(p[i*3])   > 60) vel[i*3]   *= -1;
    if (Math.abs(p[i*3+1]) > 40) vel[i*3+1] *= -1;
    if (Math.abs(p[i*3+2]) > 30) vel[i*3+2] *= -1;
  }
  geo.attributes.position.needsUpdate = true;
  pts.rotation.y = t * .05;
  rings.forEach((r, i) => {
    r.rotation.z = t * (.3 + i * .05);
    r.position.y = (i - 2) * 6 + Math.sin(t + i) * 1.5;
    r.material.opacity = .15 + Math.sin(t * 2 + i) * .1;
  });
  hexes.forEach((h, i) => {
    h.rotation.y = t * (.2 + i * .1);
    h.rotation.x = t * .1;
    h.position.y += Math.sin(t + i * 2) * .02;
  });
  camera.position.x = Math.sin(t * .2) * 3;
  camera.position.y = Math.cos(t * .15) * 2;
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
""", height=340)

# ─── STATS ROW ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="phyto-section" style="margin-top:-12px;">
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1.2rem;">
  <div class="glass" style="text-align:center; padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">11</div>
    <div class="stat-label">Disease Classes</div>
  </div>
  <div class="glass" style="text-align:center; padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">224</div>
    <div class="stat-label">Input Resolution</div>
  </div>
  <div class="glass" style="text-align:center; padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">3</div>
    <div class="stat-label">Plant Species</div>
  </div>
  <div class="glass" style="text-align:center; padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">AI</div>
    <div class="stat-label">Powered Engine</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
model = load_model()
class_names, IMG_SIZE = load_classes()

# ─── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="phyto-section glass" style="margin-bottom:0;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;
                  color:#00ff80;letter-spacing:.08em;margin-bottom:.3rem;">
        📡 SCAN LEAF SPECIMEN
      </div>
      <p style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
                color:rgba(0,255,120,.5);letter-spacing:.15em;margin-bottom:1.5rem;">
        UPLOAD IMAGE → AI ANALYSIS → DIAGNOSIS
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        w, h = image.size
        st.markdown(f"""
        <div style="display:flex;gap:.8rem;margin-top:.8rem;flex-wrap:wrap;">
          <span class="tag tag-blue">📐 {w}×{h}px</span>
          <span class="tag tag-green">🖼 {image.mode}</span>
          <span class="tag tag-blue">{uploaded.name}</span>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if uploaded is None:
        components.html("""
        <!DOCTYPE html><html><head>
        <style>
          *{margin:0;padding:0;box-sizing:border-box;}
          body{background:transparent;overflow:hidden;display:flex;align-items:center;justify-content:center;height:100%;}
          canvas{position:absolute;inset:0;}
          .msg{position:relative;z-index:2;text-align:center;font-family:'Share Tech Mono',monospace;}
          .icon{font-size:4rem;animation:bounce 2s ease-in-out infinite;}
          .line1{color:rgba(0,255,120,.7);font-size:.85rem;letter-spacing:.2em;margin-top:1rem;}
          .line2{color:rgba(0,255,120,.35);font-size:.65rem;letter-spacing:.15em;margin-top:.5rem;}
          @keyframes bounce{0%,100%{transform:translateY(0) scale(1);}50%{transform:translateY(-12px) scale(1.05);}}
          @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        </style></head>
        <body>
        <canvas id="c"></canvas>
        <div class="msg">
          <div class="icon">🌿</div>
          <div class="line1">AWAITING SPECIMEN</div>
          <div class="line2">Upload a leaf image to begin neural scan</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
          const c = document.getElementById('c');
          const W = window.innerWidth, H = window.innerHeight;
          const renderer = new THREE.WebGLRenderer({canvas:c,alpha:true,antialias:true});
          renderer.setSize(W,H);
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(60,W/H,.1,200);
          camera.position.z = 18;
          const geo = new THREE.TorusGeometry(6, .08, 8, 80);
          const mat = new THREE.MeshBasicMaterial({color:0x00ff80,transparent:true,opacity:.5});
          const torus = new THREE.Mesh(geo, mat);
          scene.add(torus);
          const geo2 = new THREE.TorusGeometry(8, .05, 8, 80);
          const mat2 = new THREE.MeshBasicMaterial({color:0x00cfff,transparent:true,opacity:.3});
          const torus2 = new THREE.Mesh(geo2, mat2);
          scene.add(torus2);
          const dGeo = new THREE.BufferGeometry();
          const dPos = new Float32Array(200*3);
          for(let i=0;i<200;i++){
            const angle = Math.random()*Math.PI*2;
            const r = 5+Math.random()*6;
            dPos[i*3] = Math.cos(angle)*r;
            dPos[i*3+1] = (Math.random()-.5)*12;
            dPos[i*3+2] = Math.sin(angle)*r;
          }
          dGeo.setAttribute('position',new THREE.BufferAttribute(dPos,3));
          const dMat = new THREE.PointsMaterial({color:0x00ff80,size:.15,transparent:true,opacity:.6});
          scene.add(new THREE.Points(dGeo,dMat));
          let t=0;
          function loop(){
            requestAnimationFrame(loop); t+=.012;
            torus.rotation.x = t*.4; torus.rotation.y = t*.3;
            torus2.rotation.x = -t*.25; torus2.rotation.z = t*.35;
            camera.position.x = Math.sin(t*.3)*3;
            camera.position.y = Math.cos(t*.2)*2;
            camera.lookAt(0,0,0);
            renderer.render(scene,camera);
          }
          loop();
        </script>
        </body></html>
        """, height=440)

    else:
        image = Image.open(uploaded)

        if model is None:
            st.markdown("""
            <div class="glass" style="text-align:center;padding:2rem;">
              <div style="font-size:2rem;margin-bottom:1rem;">⚠️</div>
              <div style="font-family:'Rajdhani',sans-serif;color:#ff4d6d;font-size:1.2rem;letter-spacing:.08em;">
                MODEL NOT FOUND
              </div>
              <p style="color:rgba(255,100,100,.6);font-family:'Share Tech Mono',monospace;font-size:.75rem;margin-top:.8rem;">
                Place plant_disease_model.keras in the same folder
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔬 Neural analysis in progress…"):
                result = smart_predict(image, model, class_names)

            # ── REJECTED ───────────────────────────────────────────────────────
            if result["status"] == "rejected":
                st.markdown(f"""
                <div class="phyto-section glass" style="border-color:#f59e0b44;
                     box-shadow:0 0 60px rgba(245,158,11,.2),0 4px 60px rgba(0,0,0,.5);
                     text-align:center;padding:2.5rem;">
                  <div style="font-size:3rem;margin-bottom:1rem;">🚫</div>
                  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.5rem;
                              color:#f59e0b;letter-spacing:.06em;margin-bottom:.8rem;">
                    UNKNOWN / NOT A PLANT LEAF
                  </div>
                  <div class="neon-divider" style="background:linear-gradient(90deg,transparent,rgba(245,158,11,.4),transparent);"></div>
                  <p style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
                            color:rgba(245,200,80,.7);letter-spacing:.08em;line-height:1.8;margin-top:1rem;">
                    {result["reason"]}<br><br>
                    ⚠️ This app <strong style="color:#f59e0b;">only accepts</strong> clear,
                    close-up leaf photos of:<br>
                    <strong style="color:#f59e0b;">🌶 Bell Pepper &nbsp;|&nbsp; 🥔 Potato &nbsp;|&nbsp; 🍅 Tomato</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)
                st.stop()

            # ── CONFIDENT RESULT ───────────────────────────────────────────────
            top_label  = result["label"]
            top_conf   = result["confidence"] * 100
            top_probs  = result["top_probs"]
            is_healthy = "healthy" in top_label.lower()

            badge_icon   = "✅" if is_healthy else "🦠"
            status_color = "#00ff80" if is_healthy else "#ff4d6d"
            glow_color   = "rgba(0,255,120,.3)" if is_healthy else "rgba(255,77,109,.3)"

            st.markdown(f"""
            <div class="phyto-section glass" style="border-color:{status_color}44;
                 box-shadow:0 0 60px {glow_color},0 4px 60px rgba(0,0,0,.5);">

              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;
                              letter-spacing:.2em;color:rgba(150,200,170,.5);margin-bottom:.5rem;">
                    DIAGNOSIS RESULT
                  </div>
                  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.6rem;
                              color:{status_color};letter-spacing:.05em;line-height:1.1;">
                    {badge_icon} {format_label(top_label)}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div class="big-num">{top_conf:.1f}<span style="font-size:1.4rem;opacity:.6;">%</span></div>
                  <div class="stat-label">Confidence</div>
                </div>
              </div>

              <div class="neon-divider"></div>

              <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;
                          color:rgba(150,200,170,.5);letter-spacing:.15em;margin-bottom:1rem;">
                TOP PREDICTIONS BREAKDOWN
              </div>
            """, unsafe_allow_html=True)

            for name, prob in top_probs:
                conf  = prob * 100
                is_h  = "healthy" in name
                bar_c = "#00ff80" if is_h else "#ff4d6d" if conf > 50 else "#00cfff"
                st.markdown(f"""
                <div style="margin-bottom:.9rem;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.35rem;">
                    <span class="bar-label">{format_label(name)}</span>
                    <span style="font-family:'Share Tech Mono',monospace;font-size:.72rem;color:{bar_c};">{conf:.1f}%</span>
                  </div>
                  <div style="height:6px;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden;">
                    <div style="width:{conf}%;height:100%;border-radius:999px;
                         background:linear-gradient(90deg,{bar_c},{bar_c}88);
                         box-shadow:0 0 10px {bar_c}88;
                         transition:width 1s ease;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if is_healthy:
                msg       = "No pathogen detected. Plant exhibits nominal health indicators."
                msg_color = "rgba(0,255,120,.8)"
                msg_icon  = "🟢"
            else:
                msg       = "Pathogen signature identified. Recommend immediate agronomic intervention."
                msg_color = "rgba(255,100,100,.8)"
                msg_icon  = "🔴"

            st.markdown(f"""
              <div class="neon-divider"></div>
              <div style="display:flex;gap:.8rem;align-items:flex-start;">
                <span style="font-size:1.1rem;">{msg_icon}</span>
                <p style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
                          color:{msg_color};letter-spacing:.04em;line-height:1.6;">
                  {msg}
                </p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            arc_color = "0x00ff80" if is_healthy else "0xff4d6d"
            arc_hex   = "#00ff80" if is_healthy else "#ff4d6d"
            components.html(f"""
            <!DOCTYPE html><html><head>
            <style>
              *{{margin:0;padding:0;box-sizing:border-box;}}
              body{{background:transparent;overflow:hidden;display:flex;
                    align-items:center;justify-content:center;}}
              canvas{{display:block;}}
              .center{{position:absolute;text-align:center;pointer-events:none;}}
              .pct{{font-family:'Exo 2',sans-serif;font-weight:900;font-size:1.8rem;
                    color:{arc_hex};filter:drop-shadow(0 0 12px {arc_hex});}}
              .lbl{{font-family:'Share Tech Mono',monospace;font-size:.55rem;
                    letter-spacing:.2em;color:rgba(150,200,170,.5);}}
              @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@900&family=Share+Tech+Mono&display=swap');
            </style></head>
            <body>
            <canvas id="c"></canvas>
            <div class="center">
              <div class="pct">{top_conf:.0f}%</div>
              <div class="lbl">CONFIDENCE</div>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
              const W=300,H=160;
              const renderer=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),alpha:true,antialias:true}});
              renderer.setSize(W,H);
              const scene=new THREE.Scene();
              const camera=new THREE.PerspectiveCamera(50,W/H,.1,100);
              camera.position.z=7;
              const conf={top_conf/100};
              const fullGeo=new THREE.TorusGeometry(2.5,.12,8,100,Math.PI*2);
              const fullMat=new THREE.MeshBasicMaterial({{color:0x1a2a1a,transparent:true,opacity:.3}});
              scene.add(new THREE.Mesh(fullGeo,fullMat));
              const arcGeo=new THREE.TorusGeometry(2.5,.18,8,100,Math.PI*2*conf);
              const arcMat=new THREE.MeshBasicMaterial({{color:{arc_color}}});
              const arc=new THREE.Mesh(arcGeo,arcMat);
              arc.rotation.z=Math.PI/2;
              scene.add(arc);
              const pGeo=new THREE.BufferGeometry();
              const pPos=new Float32Array(60*3);
              for(let i=0;i<60;i++){{
                const a=Math.random()*Math.PI*2, r=2.2+Math.random()*.8;
                pPos[i*3]=Math.cos(a)*r; pPos[i*3+1]=Math.sin(a)*r; pPos[i*3+2]=(Math.random()-.5)*1;
              }}
              pGeo.setAttribute('position',new THREE.BufferAttribute(pPos,3));
              const pMat=new THREE.PointsMaterial({{color:{arc_color},size:.06,transparent:true,opacity:.7}});
              scene.add(new THREE.Points(pGeo,pMat));
              let t=0;
              function loop(){{
                requestAnimationFrame(loop); t+=.015;
                arc.rotation.z=Math.PI/2+t*.3;
                renderer.render(scene,camera);
              }}
              loop();
            </script>
            </body></html>
            """, height=170)

# ─── BOTTOM CLASSES GRID ────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="phyto-section glass">
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.2rem;
              color:#00cfff;letter-spacing:.1em;margin-bottom:1.2rem;">
    🗂 DETECTABLE DISEASE CATALOGUE
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.7rem;">
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🌶 Pepper — Bacterial Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(0,255,120,.2);border-radius:10px;
                background:rgba(0,255,120,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(150,255,150,.7);letter-spacing:.06em;">
      🌶 Pepper — Healthy
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🥔 Potato — Early Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🥔 Potato — Late Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Bacterial Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🍅 Tomato — Early Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Late Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(168,85,247,.2);border-radius:10px;
                background:rgba(168,85,247,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(200,150,255,.7);letter-spacing:.06em;">
      🍅 Tomato — Leaf Mold
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🍅 Tomato — Septoria Leaf Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Target Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(168,85,247,.2);border-radius:10px;
                background:rgba(168,85,247,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(200,150,255,.7);letter-spacing:.06em;">
      🍅 Tomato — Mosaic Virus
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:2rem;
     font-family:'Share Tech Mono',monospace;font-size:.62rem;
     letter-spacing:.2em;color:rgba(0,255,120,.25);">
  PHYTO SCAN AI · NEURAL PATHOGEN DETECTION · POWERED BY KERAS + TENSORFLOW
</div>
""", unsafe_allow_html=True)
