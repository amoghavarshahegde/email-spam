"""
app.py – SpamShield AI · Multi-page Premium App
OWNER: app/ team

Pages:  Analyse | History | About
Run with:
    streamlit run app/app.py
from the repo root.
"""

import os, sys, re, joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("preprocessing"))

try:
    from cleaning import clean_text
except ImportError:
    st.error("Cannot import cleaning.py – run from the repo root.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:      #080812;
  --bg2:     #0e0e20;
  --violet:  #7c3aed;
  --violet2: #9f5ff5;
  --pink:    #ec4899;
  --cyan:    #06b6d4;
  --lime:    #22c55e;
  --amber:   #f59e0b;
  --red:     #ef4444;
  --white:   #ffffff;
  --muted:   rgba(255,255,255,0.5);
}

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: #fff !important;
}
.main .block-container {
    background: transparent !important;
    padding: 1.2rem 2rem 4rem !important;
    max-width: 1000px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09091a 0%, #110a2a 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.35) !important;
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stSlider div[role="slider"] {
    background: var(--violet2) !important;
    box-shadow: 0 0 14px rgba(124,58,237,0.9) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(124,58,237,0.25) !important; }

/* ── NAV BUTTONS ── */
.nav-btn {
    display:flex; align-items:center; gap:10px;
    padding:12px 18px; border-radius:12px;
    margin-bottom:6px; cursor:pointer;
    border:1px solid transparent;
    font-size:0.92rem; font-weight:600;
    color:rgba(255,255,255,0.6);
    transition:all 0.25s;
    background:transparent;
}
.nav-btn:hover, .nav-btn.active {
    background:rgba(124,58,237,0.2);
    border-color:rgba(124,58,237,0.5);
    color:#fff;
    box-shadow: 0 0 20px rgba(124,58,237,0.2);
}
.nav-btn.active { color:#a78bfa; }

/* ── ALL BUTTONS ── */
.stButton > button {
    font-family:'Outfit',sans-serif !important;
    font-weight:600 !important;
    border-radius:12px !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    background:rgba(255,255,255,0.06) !important;
    color:#fff !important;
    transition:all 0.25s !important;
    padding:10px 20px !important;
}
.stButton > button:hover {
    background:rgba(124,58,237,0.25) !important;
    border-color:rgba(124,58,237,0.6) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(124,58,237,0.3) !important;
    color:#fff !important;
}
button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#ec4899) !important;
    border:none !important; color:#fff !important;
    font-size:1.05rem !important; font-weight:700 !important;
    padding:14px !important; border-radius:14px !important;
    box-shadow:0 8px 32px rgba(124,58,237,0.5) !important;
    letter-spacing:0.5px;
}
button[kind="primary"]:hover {
    background:linear-gradient(135deg,#9f5ff5,#f472b6) !important;
    transform:translateY(-3px) !important;
    box-shadow:0 14px 40px rgba(124,58,237,0.65) !important;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background:rgba(255,255,255,0.05) !important;
    border:1.5px solid rgba(124,58,237,0.35) !important;
    border-radius:14px !important;
    color:#fff !important;
    font-family:'Outfit',sans-serif !important;
    font-size:0.97rem !important;
    padding:16px !important;
}
.stTextArea textarea::placeholder { color:rgba(255,255,255,0.28) !important; }
.stTextArea textarea:focus {
    border-color:rgba(124,58,237,0.85) !important;
    box-shadow:0 0 0 3px rgba(124,58,237,0.2), 0 0 25px rgba(124,58,237,0.12) !important;
}

/* ── SECTION LABEL ── */
.slabel {
    font-size:0.72rem; font-weight:700; letter-spacing:2px;
    text-transform:uppercase; color:rgba(255,255,255,0.35);
    margin-bottom:10px; margin-top:4px;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0e0820 0%, #160b35 40%, #0a0818 100%);
    border:1px solid rgba(124,58,237,0.3);
    border-radius:24px; padding:52px 48px 44px;
    margin-bottom:28px; position:relative; overflow:hidden;
    transition: box-shadow 0.3s, transform 0.3s;
}
.hero:hover {
    box-shadow: 0 0 60px rgba(124,58,237,0.25), 0 0 120px rgba(236,72,153,0.1) !important;
    transform: translateY(-2px);
}
.hero::before {
    content:''; position:absolute; top:-100px; right:-80px;
    width:350px; height:350px;
    background:radial-gradient(circle,rgba(124,58,237,0.2),transparent 65%);
    border-radius:50%; pointer-events:none;
}
.hero::after {
    content:''; position:absolute; bottom:-80px; left:-60px;
    width:280px; height:280px;
    background:radial-gradient(circle,rgba(236,72,153,0.15),transparent 65%);
    border-radius:50%; pointer-events:none;
}
.hero-eye {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(124,58,237,0.18);
    border:1px solid rgba(124,58,237,0.5);
    color:#c084fc; font-size:0.74rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    padding:5px 14px; border-radius:999px; margin-bottom:18px;
}
.hero-h1 {
    font-size:3.2rem; font-weight:900; line-height:1.1;
    color:#fff; margin:0 0 10px; letter-spacing:-1px;
}
.grad { background:linear-gradient(90deg,#7c3aed,#ec4899,#06b6d4);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub {
    font-size:1rem; color:rgba(255,255,255,0.55);
    max-width:520px; line-height:1.75; margin:0;
}
.hero-badges { display:flex; gap:10px; margin-top:22px; flex-wrap:wrap; }
.hbadge {
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:999px; padding:6px 16px;
    font-size:0.78rem; font-weight:600; color:rgba(255,255,255,0.65);
    display:inline-flex; align-items:center; gap:6px;
}

/* ── STAT CARDS ── */
.stats { display:flex; gap:14px; margin-bottom:28px; }
.scard {
    flex:1; border-radius:18px; padding:22px 18px;
    text-align:center; position:relative; overflow:hidden;
    cursor:default;
    transition: transform 0.3s, box-shadow 0.3s !important;
}
.scard:hover {
    transform: translateY(-6px) scale(1.03) !important;
}
.scard-1 { background:linear-gradient(135deg,rgba(124,58,237,0.22),rgba(124,58,237,0.06)); border:1px solid rgba(124,58,237,0.3); }
.scard-1:hover { box-shadow:0 0 40px rgba(124,58,237,0.4), 0 20px 40px rgba(124,58,237,0.15) !important; border-color:rgba(124,58,237,0.7) !important; }
.scard-2 { background:linear-gradient(135deg,rgba(236,72,153,0.22),rgba(236,72,153,0.06)); border:1px solid rgba(236,72,153,0.3); }
.scard-2:hover { box-shadow:0 0 40px rgba(236,72,153,0.4), 0 20px 40px rgba(236,72,153,0.15) !important; border-color:rgba(236,72,153,0.7) !important; }
.scard-3 { background:linear-gradient(135deg,rgba(6,182,212,0.22),rgba(6,182,212,0.06));  border:1px solid rgba(6,182,212,0.3); }
.scard-3:hover { box-shadow:0 0 40px rgba(6,182,212,0.4), 0 20px 40px rgba(6,182,212,0.15) !important; border-color:rgba(6,182,212,0.7) !important; }
.scard-4 { background:linear-gradient(135deg,rgba(34,197,94,0.22),rgba(34,197,94,0.06));   border:1px solid rgba(34,197,94,0.3); }
.scard-4:hover { box-shadow:0 0 40px rgba(34,197,94,0.4), 0 20px 40px rgba(34,197,94,0.15) !important; border-color:rgba(34,197,94,0.7) !important; }
.snum { font-size:2.1rem; font-weight:900; color:#fff; line-height:1; }
.snum-1{color:#a78bfa;} .snum-2{color:#f9a8d4;} .snum-3{color:#67e8f9;} .snum-4{color:#86efac;}
.slbl2 { font-size:0.72rem; font-weight:600; color:rgba(255,255,255,0.4);
         text-transform:uppercase; letter-spacing:1px; margin-top:5px; }

/* ── VERDICT ── */
.verdict-spam {
    background:linear-gradient(135deg,#1a0505,#3a0a0a);
    border:1.5px solid rgba(239,68,68,0.5);
    border-left:6px solid #ef4444;
    border-radius:20px; padding:32px 36px;
    margin:20px 0 16px; position:relative; overflow:hidden;
    box-shadow:0 0 60px rgba(239,68,68,0.15);
    transition: box-shadow 0.3s, transform 0.3s;
}
.verdict-spam:hover { transform:translateY(-3px); box-shadow:0 0 80px rgba(239,68,68,0.3) !important; }
.verdict-spam::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:180px; height:180px;
    background:radial-gradient(circle,rgba(239,68,68,0.2),transparent 70%);
    border-radius:50%;
}
.verdict-ham {
    background:linear-gradient(135deg,#041a0a,#083318);
    border:1.5px solid rgba(34,197,94,0.45);
    border-left:6px solid #22c55e;
    border-radius:20px; padding:32px 36px;
    margin:20px 0 16px; position:relative; overflow:hidden;
    box-shadow:0 0 60px rgba(34,197,94,0.1);
    transition: box-shadow 0.3s, transform 0.3s;
}
.verdict-ham:hover { transform:translateY(-3px); box-shadow:0 0 80px rgba(34,197,94,0.25) !important; }
.verdict-ham::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:180px; height:180px;
    background:radial-gradient(circle,rgba(34,197,94,0.15),transparent 70%);
    border-radius:50%;
}
.verd-icon { font-size:3rem; line-height:1; margin-bottom:8px; }
.verd-title { font-size:2.2rem; font-weight:900; letter-spacing:0.5px; color:#fff; margin:0; }
.vt-spam { color:#ff6b6b; } .vt-ham { color:#4ade80; }
.verd-sub { font-size:0.95rem; color:rgba(255,255,255,0.55); margin-top:6px; }

/* ── CONF BLOCK ── */
.conf-block {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px; padding:20px 22px;
    transition: transform 0.25s, box-shadow 0.25s;
}
.conf-block:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(0,0,0,0.3); }
.conf-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.conf-name { font-size:0.78rem; font-weight:700; text-transform:uppercase;
             letter-spacing:1px; color:rgba(255,255,255,0.45); }
.conf-pct { font-size:1.6rem; font-weight:900; color:#fff; }
.cps { color:#ff6b6b; } .cph { color:#4ade80; }

/* ── WHY SECTION ── */
.why-card {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px; padding:24px 26px; margin-top:20px;
    transition: box-shadow 0.3s, transform 0.3s;
}
.why-card:hover { transform:translateY(-2px); box-shadow:0 8px 40px rgba(0,0,0,0.4); }
.why-title {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:2px; color:#a78bfa; margin-bottom:16px;
    display:flex; align-items:center; gap:8px;
}
.trigger-word {
    display:inline-block; margin:4px;
    padding:5px 12px; border-radius:999px;
    font-size:0.82rem; font-weight:700; font-family:'JetBrains Mono',monospace;
}
.tw-spam {
    background:rgba(239,68,68,0.18);
    border:1px solid rgba(239,68,68,0.45);
    color:#ff6b6b;
}
.tw-ham {
    background:rgba(34,197,94,0.15);
    border:1px solid rgba(34,197,94,0.4);
    color:#4ade80;
}
.feat-row {
    display:flex; align-items:center; gap:14px;
    margin-bottom:12px;
}
.feat-label {
    font-size:0.8rem; color:rgba(255,255,255,0.55);
    min-width:160px; font-weight:500;
}
.feat-bar-bg {
    flex:1; height:8px; background:rgba(255,255,255,0.07);
    border-radius:999px; overflow:hidden;
}
.feat-bar-fill { height:100%; border-radius:999px; }
.feat-val {
    font-size:0.8rem; font-weight:700; color:#fff; min-width:40px; text-align:right;
}
.reason-box {
    background:rgba(124,58,237,0.08);
    border:1px solid rgba(124,58,237,0.25);
    border-radius:12px; padding:16px 18px; margin-top:16px;
    font-size:0.9rem; line-height:1.75; color:rgba(255,255,255,0.75);
}
.reason-box b { color:#a78bfa; }
.reason-box .rx { color:#ff6b6b; font-weight:700; }
.reason-box .rg { color:#4ade80; font-weight:700; }

/* ── HISTORY TABLE ── */
.hist-row {
    display:flex; align-items:center; gap:14px;
    padding:14px 18px; border-radius:14px; margin-bottom:8px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.07);
    transition:all 0.25s;
}
.hist-row:hover {
    background:rgba(124,58,237,0.1);
    border-color:rgba(124,58,237,0.35);
    transform:translateX(4px);
}
.hist-time { font-size:0.72rem; color:rgba(255,255,255,0.3); min-width:80px; }
.hist-badge-spam {
    background:rgba(239,68,68,0.2); border:1px solid rgba(239,68,68,0.5);
    color:#ff6b6b; font-size:0.72rem; font-weight:700;
    padding:3px 10px; border-radius:999px; min-width:60px; text-align:center;
}
.hist-badge-ham {
    background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.4);
    color:#4ade80; font-size:0.72rem; font-weight:700;
    padding:3px 10px; border-radius:999px; min-width:60px; text-align:center;
}
.hist-msg { font-size:0.85rem; color:rgba(255,255,255,0.65); flex:1;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.hist-pct { font-size:0.85rem; font-weight:800; min-width:52px; text-align:right; }

/* ── ABOUT ── */
.about-card {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px; padding:28px;
    margin-bottom:16px;
    transition:all 0.25s;
}
.about-card:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(0,0,0,0.4); }
.about-h { font-size:1.1rem; font-weight:800; color:#a78bfa; margin-bottom:12px; }
.pipe-step {
    display:flex; align-items:flex-start; gap:16px;
    padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);
}
.pipe-step:last-child { border-bottom:none; }
.pipe-num {
    width:34px; height:34px; border-radius:10px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:0.9rem;
}
.pipe-info h4 { margin:0 0 4px; font-size:0.92rem; color:#fff; font-weight:700; }
.pipe-info p { margin:0; font-size:0.82rem; color:rgba(255,255,255,0.5); line-height:1.6; }

/* ── CLEANED TEXT ── */
.cleaned-box {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:18px 20px; margin-top:16px;
}
.cleaned-label {
    font-size:0.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:rgba(124,58,237,0.75); margin-bottom:8px;
}
.cleaned-text {
    font-family:'JetBrains Mono',monospace;
    font-size:0.8rem; color:rgba(255,255,255,0.45); line-height:1.8; word-break:break-word;
}

/* ── FANCY DIVIDER ── */
.fdiv {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(124,58,237,0.5),rgba(236,72,153,0.4),transparent);
    margin:28px 0; border:none;
}

/* ── FOOTER ── */
.footer {
    text-align:center; padding:28px 0 4px;
    font-size:0.76rem; color:rgba(255,255,255,0.18);
}
.footer span { color:#a78bfa; font-weight:700; }

/* ── CURSOR: keep the native arrow, overlays are handled by JS ── */

/* ── HOVER GLOW on interactive blocks (pure CSS fallback) ── */
.scard, .conf-block, .verdict-spam, .verdict-ham, .hist-row, .about-card, .why-card {
    will-change: transform;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CURSOR + BURST JS  (injected into parent document via iframe)
# ═══════════════════════════════════════════════════════════════════════════════
components.html("""
<script>
(function(){
  var pdoc = window.parent.document;

  // ── Glow orb (follows cursor) ──────────────────────────────────────────────
  var glow = pdoc.createElement('div');
  glow.style.cssText = [
    'position:fixed','width:380px','height:380px','border-radius:50%',
    'background:radial-gradient(circle,rgba(124,58,237,0.13),transparent 68%)',
    'pointer-events:none','transform:translate(-50%,-50%)',
    'transition:width .4s ease,height .4s ease,background .4s ease,opacity .3s',
    'z-index:99998','mix-blend-mode:screen','opacity:1'
  ].join(';');
  pdoc.body.appendChild(glow);

  // ── Sharp dot ──────────────────────────────────────────────────────────────
  var dot = pdoc.createElement('div');
  dot.style.cssText = [
    'position:fixed','width:12px','height:12px','border-radius:50%',
    'background:linear-gradient(135deg,#7c3aed,#ec4899)',
    'pointer-events:none','transform:translate(-50%,-50%)',
    'transition:width .15s,height .15s,transform .05s',
    'z-index:99999',
    'box-shadow:0 0 14px rgba(124,58,237,1),0 0 28px rgba(236,72,153,0.5)'
  ].join(';');
  pdoc.body.appendChild(dot);

  // ── Ring (follows slightly behind) ────────────────────────────────────────
  var ring = pdoc.createElement('div');
  ring.style.cssText = [
    'position:fixed','width:36px','height:36px','border-radius:50%',
    'border:2px solid rgba(124,58,237,0.6)',
    'pointer-events:none','transform:translate(-50%,-50%)',
    'z-index:99997',
    'transition:width .2s,height .2s,border-color .2s',
    'backdrop-filter:blur(0px)'
  ].join(';');
  pdoc.body.appendChild(ring);

  // ── Add keyframes for burst ────────────────────────────────────────────────
  var style = pdoc.createElement('style');
  style.textContent = `
    @keyframes burstRing {
      0%   { width:0;    height:0;    opacity:.9; }
      100% { width:220px; height:220px; opacity:0; }
    }
    @keyframes burstDot {
      0%   { opacity:1; transform:translate(-50%,-50%) scale(1); }
      100% { opacity:0; transform:translate(-50%,-50%) scale(2.5); }
    }
  `;
  pdoc.head.appendChild(style);

  var mx=0, my=0, rx=0, ry=0;

  // Track mouse
  pdoc.addEventListener('mousemove', function(e){
    mx = e.clientX; my = e.clientY;
    glow.style.left  = mx+'px'; glow.style.top  = my+'px';
    dot.style.left   = mx+'px'; dot.style.top   = my+'px';
  });

  // Ring lags behind
  (function animRing(){
    rx += (mx - rx) * 0.12;
    ry += (my - ry) * 0.12;
    ring.style.left = rx+'px'; ring.style.top = ry+'px';
    requestAnimationFrame(animRing);
  })();

  // Hover detection on hovered elements → burst glow
  function attachHover(selector, color1, color2){
    setInterval(function(){
      pdoc.querySelectorAll(selector).forEach(function(el){
        if(el._hoverAttached) return;
        el._hoverAttached = true;
        el.addEventListener('mouseenter', function(){
          glow.style.width   = '600px';
          glow.style.height  = '600px';
          glow.style.background = 'radial-gradient(circle,'+color1+',transparent 65%)';
          dot.style.width  = '18px'; dot.style.height = '18px';
          ring.style.width = '56px'; ring.style.height= '56px';
          ring.style.borderColor = color2;
        });
        el.addEventListener('mouseleave', function(){
          glow.style.width   = '380px';
          glow.style.height  = '380px';
          glow.style.background = 'radial-gradient(circle,rgba(124,58,237,0.13),transparent 68%)';
          dot.style.width  = '12px'; dot.style.height = '12px';
          ring.style.width = '36px'; ring.style.height= '36px';
          ring.style.borderColor = 'rgba(124,58,237,0.6)';
        });
      });
    }, 600);
  }

  attachHover('.scard',       'rgba(124,58,237,0.22)',  'rgba(124,58,237,0.8)');
  attachHover('.conf-block',  'rgba(6,182,212,0.18)',   'rgba(6,182,212,0.7)');
  attachHover('.verdict-spam','rgba(239,68,68,0.2)',    'rgba(239,68,68,0.7)');
  attachHover('.verdict-ham', 'rgba(34,197,94,0.18)',   'rgba(34,197,94,0.7)');
  attachHover('.hist-row',    'rgba(236,72,153,0.15)',  'rgba(236,72,153,0.6)');
  attachHover('.about-card',  'rgba(6,182,212,0.15)',   'rgba(6,182,212,0.6)');
  attachHover('.why-card',    'rgba(245,158,11,0.15)',  'rgba(245,158,11,0.6)');
  attachHover('button',       'rgba(124,58,237,0.2)',   'rgba(124,58,237,0.7)');
  attachHover('.hero',        'rgba(124,58,237,0.18)',  'rgba(124,58,237,0.6)');

  // Click BURST
  pdoc.addEventListener('click', function(e){
    // Burst ring
    var b = pdoc.createElement('div');
    b.style.cssText = [
      'position:fixed',
      'left:'+e.clientX+'px','top:'+e.clientY+'px',
      'width:0','height:0','border-radius:50%',
      'border:3px solid rgba(124,58,237,0.8)',
      'pointer-events:none','transform:translate(-50%,-50%)',
      'z-index:99996','animation:burstRing .7s cubic-bezier(.2,.8,.4,1) forwards'
    ].join(';');
    pdoc.body.appendChild(b);

    // Burst dot
    var bd = pdoc.createElement('div');
    bd.style.cssText = [
      'position:fixed',
      'left:'+e.clientX+'px','top:'+e.clientY+'px',
      'width:14px','height:14px','border-radius:50%',
      'background:radial-gradient(circle,#ec4899,#7c3aed)',
      'pointer-events:none','transform:translate(-50%,-50%)',
      'z-index:99996','animation:burstDot .5s ease-out forwards'
    ].join(';');
    pdoc.body.appendChild(bd);

    // Sparkles
    for(var i=0;i<8;i++){
      (function(i){
        var sp = pdoc.createElement('div');
        var angle = (i/8)*360;
        var dist  = 50+Math.random()*60;
        var size  = 4+Math.random()*5;
        var colors=['#7c3aed','#ec4899','#06b6d4','#f59e0b','#22c55e'];
        var col = colors[Math.floor(Math.random()*colors.length)];
        sp.style.cssText = [
          'position:fixed','border-radius:50%',
          'width:'+size+'px','height:'+size+'px',
          'left:'+e.clientX+'px','top:'+e.clientY+'px',
          'background:'+col,
          'pointer-events:none','z-index:99995',
          'transform:translate(-50%,-50%)',
          'transition:all .6s cubic-bezier(.2,.8,.4,1)',
          'opacity:1','box-shadow:0 0 8px '+col
        ].join(';');
        pdoc.body.appendChild(sp);
        setTimeout(function(){
          var rad = angle*Math.PI/180;
          sp.style.left   = (e.clientX + Math.cos(rad)*dist)+'px';
          sp.style.top    = (e.clientY + Math.sin(rad)*dist)+'px';
          sp.style.opacity = '0';
          sp.style.transform = 'translate(-50%,-50%) scale(0)';
        }, 10);
        setTimeout(function(){ sp.remove(); }, 700);
      })(i);
    }

    setTimeout(function(){ b.remove(); bd.remove(); }, 800);
  });

  // Cursor hide on leave
  pdoc.addEventListener('mouseleave', function(){ glow.style.opacity='0'; dot.style.opacity='0'; ring.style.opacity='0'; });
  pdoc.addEventListener('mouseenter', function(){ glow.style.opacity='1'; dot.style.opacity='1'; ring.style.opacity='1'; });
})();
</script>
""", height=0)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if 'history'  not in st.session_state: st.session_state.history  = []
if 'page'     not in st.session_state: st.session_state.page     = 'analyse'
if 'inp'      not in st.session_state: st.session_state.inp      = ''

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join("models", "LogisticRegression.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPLANATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
SPAM_KEYWORDS = [
    'free','win','winner','prize','claim','urgent','call now','limited',
    'congratulations','congratulation','selected','reward','cash','offer',
    'click','deal','discount','cheap','guaranteed','act now','apply now',
    'approved','credit','loan','bonus','gift','earn','income','money',
    'save','subscribe','trial','risk free','no cost','won','awarded',
    'expires','verify','account','bank','password','ssn','social security',
]

def explain(original_text, cleaned_text, model, spam_prob):
    """Return explanation dict for why a message is spam or ham."""
    result = {}

    # ── 1. Handcrafted features ───────────────────────────────────────────────
    result['len']        = len(original_text)
    result['excl']       = original_text.count('!')
    result['links']      = len(re.findall(r'http|www|\.com|\.net|\.org|click here', original_text, re.I))
    result['urls_clean'] = cleaned_text.count('url')
    total = max(len(original_text), 1)
    upper = sum(1 for c in original_text if c.isupper())
    digits= sum(1 for c in original_text if c.isdigit())
    result['upper_ratio'] = upper / total
    result['digit_ratio'] = digits / total

    # ── 2. Spam keyword hits ──────────────────────────────────────────────────
    low = original_text.lower()
    hits = [kw for kw in SPAM_KEYWORDS if kw in low]
    result['kw_hits'] = hits
    result['kw_count'] = len(hits)

    # ── 3. TF-IDF top words from model (real coefficients) ───────────────────
    try:
        vec   = model.named_steps['tfidf']
        clf   = model.named_steps['clf']
        coefs = clf.coef_[0]
        fnames= vec.get_feature_names_out()
        X     = vec.transform([cleaned_text])
        tfidf_vals = X.toarray()[0]
        scores     = tfidf_vals * coefs            # positive→spam, negative→ham
        nonzero    = np.nonzero(tfidf_vals)[0]
        word_scores= [(fnames[i], scores[i], coefs[i]) for i in nonzero]
        spam_ws = sorted([w for w in word_scores if w[1]>0], key=lambda x:-x[1])[:12]
        ham_ws  = sorted([w for w in word_scores if w[1]<0], key=lambda x:x[1])[:8]
        result['spam_words'] = spam_ws
        result['ham_words']  = ham_ws
    except Exception:
        result['spam_words'] = []
        result['ham_words']  = []

    # ── 4. Plain-English reason ───────────────────────────────────────────────
    reasons_spam, reasons_ham = [], []
    if result['kw_count'] >= 3:
        reasons_spam.append(f"<span class='rx'>{result['kw_count']} spam trigger words</span> detected ({', '.join(hits[:4])}{'…' if len(hits)>4 else ''})")
    elif result['kw_count'] > 0:
        reasons_spam.append(f"Contains spam keyword(s): <span class='rx'>{', '.join(hits)}</span>")
    if result['excl'] >= 2:
        reasons_spam.append(f"<span class='rx'>{result['excl']} exclamation marks</span> — common in spam")
    if result['upper_ratio'] > 0.3:
        reasons_spam.append(f"<span class='rx'>{result['upper_ratio']*100:.0f}% uppercase letters</span> — shouting style typical of spam")
    if result['links'] >= 1:
        reasons_spam.append(f"<span class='rx'>{result['links']} URL/link</span> reference(s) found")
    if result['digit_ratio'] > 0.1:
        reasons_spam.append(f"<span class='rx'>{result['digit_ratio']*100:.0f}% digits</span> — phone numbers / money amounts common in spam")
    if result['spam_words']:
        top = ', '.join([f"<span class='rx'>{w[0]}</span>" for w in result['spam_words'][:5]])
        reasons_spam.append(f"TF-IDF model flagged high-weight spam tokens: {top}")

    if result['kw_count'] == 0:
        reasons_ham.append("<span class='rg'>No spam trigger words</span> detected")
    if result['excl'] == 0:
        reasons_ham.append("<span class='rg'>No exclamation marks</span> — calm, natural tone")
    if result['upper_ratio'] < 0.1:
        reasons_ham.append(f"<span class='rg'>Low uppercase ratio ({result['upper_ratio']*100:.0f}%)</span> — normal writing style")
    if result['links'] == 0:
        reasons_ham.append("<span class='rg'>No URLs or links</span> present")
    if result['ham_words']:
        top = ', '.join([f"<span class='rg'>{w[0]}</span>" for w in result['ham_words'][:4]])
        reasons_ham.append(f"TF-IDF model found ham-leaning tokens: {top}")

    result['reasons_spam'] = reasons_spam
    result['reasons_ham']  = reasons_ham
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  —  NAV + SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 20px;'>
        <div style='font-size:3.8rem; filter:drop-shadow(0 0 20px rgba(124,58,237,1));'>🛡️</div>
        <div style='font-size:1.5rem; font-weight:900;
                    background:linear-gradient(90deg,#7c3aed,#ec4899);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px; margin-top:8px;'>SpamShield AI</div>
        <div style='font-size:0.7rem; color:rgba(255,255,255,0.28);
                    text-transform:uppercase; letter-spacing:2px; margin-top:4px;'>
            v2.0 · ML Powered
        </div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.5),transparent);margin:0 0 22px;'></div>
    <div style='font-size:0.68rem; font-weight:700; text-transform:uppercase;
                letter-spacing:2px; color:rgba(255,255,255,0.3); margin-bottom:10px; padding:0 4px;'>
        Navigation
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("🔍", "Analyse",  "analyse"),
        ("📜", "History",  "history"),
        ("ℹ️",  "About",    "about"),
    ]
    for icon, label, key in pages:
        active = "active" if st.session_state.page == key else ""
        badge = ""
        if key == "history" and len(st.session_state.history) > 0:
            badge = f"<span style='background:rgba(124,58,237,0.5);border-radius:999px;padding:1px 8px;font-size:0.68rem;margin-left:auto;'>{len(st.session_state.history)}</span>"
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("""
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.3),transparent);margin:20px 0;'></div>
    <div style='font-size:0.68rem; font-weight:700; text-transform:uppercase;
                letter-spacing:2px; color:rgba(255,255,255,0.3); margin-bottom:14px; padding:0 4px;'>
        ⚙️ &nbsp; Settings
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Spam Threshold", 0.10, 0.95, 0.50, 0.05,
        help="Messages with spam probability ≥ this value are flagged SPAM.",
        label_visibility="collapsed",
    )
    pct   = int((threshold-0.10)/0.85*100)
    tcolor= "#ef4444" if threshold<0.35 else ("#f59e0b" if threshold<0.65 else "#22c55e")
    tlbl  = "Aggressive" if threshold<0.35 else ("Balanced" if threshold<0.65 else "Lenient")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                border-radius:10px;padding:12px 16px;margin:6px 0 20px;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
            <span style='font-size:0.72rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:1px;'>
                {tlbl}
            </span>
            <span style='font-size:1.1rem;font-weight:900;color:{tcolor};'>{threshold:.2f}</span>
        </div>
        <div style='height:5px;background:rgba(255,255,255,0.07);border-radius:999px;'>
            <div style='width:{pct}%;height:100%;background:{tcolor};border-radius:999px;
                        box-shadow:0 0 8px {tcolor}88;'></div>
        </div>
    </div>
    <div style='font-size:0.8rem;line-height:1.9;color:rgba(255,255,255,0.45);padding:0 2px;'>
        <b style='color:#a78bfa;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;'>⚡ Pipeline</b><br>
        <span style='color:#06b6d4;'>①</span> Strip HTML &amp; symbols<br>
        <span style='color:#06b6d4;'>②</span> Mask URLs / emails / numbers<br>
        <span style='color:#06b6d4;'>③</span> TF-IDF vectorisation<br>
        <span style='color:#06b6d4;'>④</span> Logistic Regression<br>
        <span style='color:#06b6d4;'>⑤</span> Threshold decision
    </div>
    <div style='height:1px;background:rgba(124,58,237,0.2);margin:18px 0;'></div>
    <div style='text-align:center;font-size:0.7rem;color:rgba(255,255,255,0.18);'>
        Test F1 &nbsp;0.9312 &nbsp;·&nbsp; AUC &nbsp;0.989<br>5,157 training messages
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'analyse':

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eye">🤖 &nbsp; Machine Learning · NLP · Real-time</div>
        <div class="hero-h1">Detect Spam<br><span class="grad">Instantly.</span></div>
        <p class="hero-sub">
            Paste any SMS or email message and our AI analyses it in milliseconds —
            and explains <b>exactly why</b> it's spam or safe.
        </p>
        <div class="hero-badges">
            <span class="hbadge">⚡ &lt;50ms inference</span>
            <span class="hbadge">🎯 98.3% accuracy</span>
            <span class="hbadge">🔍 Full explanation</span>
            <span class="hbadge">🔒 Runs locally</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stats">
        <div class="scard scard-1"><div class="snum snum-1">98.3%</div><div class="slbl2">Accuracy</div></div>
        <div class="scard scard-2"><div class="snum snum-2">0.989</div><div class="slbl2">AUC Score</div></div>
        <div class="scard scard-3"><div class="snum snum-3">5,157</div><div class="slbl2">Training Msgs</div></div>
        <div class="scard scard-4"><div class="snum snum-4">0.931</div><div class="slbl2">F1 Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(f"⚠️ Model not found at `{MODEL_PATH}`. Run `python models/train_classical.py` first.")
        st.stop()

    # Example buttons
    SPAM_EX = ("WINNER!! As a valued network customer you have been selected to receive "
               "a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.")
    HAM_EX  = "Hey, are we still on for lunch tomorrow? Let me know if you need to reschedule!"

    st.markdown('<div class="slabel">✨ &nbsp; Try an example</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🚨  Load Spam Example",   use_container_width=True): st.session_state.inp = SPAM_EX
    if c2.button("💬  Load Normal Example", use_container_width=True): st.session_state.inp = HAM_EX

    # Input
    st.markdown('<div class="slabel" style="margin-top:20px;">📝 &nbsp; Your message</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "msg", value=st.session_state.inp, height=160,
        placeholder="Paste any SMS or email here…",
        label_visibility="collapsed",
    )
    go = st.button("🔍  Analyse Message", type="primary", use_container_width=True)

    # ── Result ──────────────────────────────────────────────────────────────
    if go:
        text = user_input.strip()
        if not text:
            st.warning("⚠️ Please enter a message.")
        else:
            with st.spinner("🧠 Analysing…"):
                cleaned   = clean_text(text)
                probs     = model.predict_proba([cleaned])[0]
                spam_prob = float(probs[1])
                ham_prob  = float(probs[0])
                is_spam   = spam_prob >= threshold
                exp       = explain(text, cleaned, model, spam_prob)

            # Save to history
            st.session_state.history.insert(0, {
                'time':     datetime.now().strftime('%H:%M · %d %b'),
                'msg':      text[:90] + ('…' if len(text)>90 else ''),
                'full':     text,
                'verdict':  'SPAM' if is_spam else 'SAFE',
                'pct':      spam_prob*100,
                'thresh':   threshold,
                'exp':      exp,
            })

            st.markdown('<div class="fdiv"></div>', unsafe_allow_html=True)

            # Verdict banner
            if is_spam:
                st.markdown(f"""
                <div class="verdict-spam">
                    <div class="verd-icon">🚨</div>
                    <div class="verd-title"><span class="vt-spam">SPAM DETECTED</span></div>
                    <div class="verd-sub">This message shows strong spam signals. Do not click any links.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-ham">
                    <div class="verd-icon">✅</div>
                    <div class="verd-title"><span class="vt-ham">SAFE — Not Spam</span></div>
                    <div class="verd-sub">No spam patterns detected. This message appears legitimate.</div>
                </div>""", unsafe_allow_html=True)

            # Confidence bars
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"""
                <div class="conf-block">
                    <div class="conf-top">
                        <span class="conf-name">🔴 Spam probability</span>
                        <span class="conf-pct cps">{spam_prob*100:.1f}%</span>
                    </div>
                    <div style='height:10px;background:rgba(255,255,255,0.07);border-radius:999px;'>
                        <div style='width:{spam_prob*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#ef4444,#ff6b6b);border-radius:999px;
                            box-shadow:0 0 10px rgba(239,68,68,0.6);'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div class="conf-block">
                    <div class="conf-top">
                        <span class="conf-name">🟢 Safe probability</span>
                        <span class="conf-pct cph">{ham_prob*100:.1f}%</span>
                    </div>
                    <div style='height:10px;background:rgba(255,255,255,0.07);border-radius:999px;'>
                        <div style='width:{ham_prob*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#22c55e,#4ade80);border-radius:999px;
                            box-shadow:0 0 10px rgba(34,197,94,0.6);'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── WHY section ──────────────────────────────────────────────────
            st.markdown("""
            <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:2px;color:rgba(255,255,255,0.3);margin:28px 0 10px;'>
                🔬 &nbsp; Deep Analysis — Why this verdict?
            </div>""", unsafe_allow_html=True)

            # ── Trigger words
            spam_wds = exp.get('spam_words', [])
            ham_wds  = exp.get('ham_words',  [])
            kw_hits  = exp.get('kw_hits', [])

            tw_html = ""
            if spam_wds:
                for w,s,c in spam_wds[:10]:
                    tw_html += f'<span class="trigger-word tw-spam" title="spam weight {c:.3f}">🔴 {w}</span>'
            if ham_wds:
                for w,s,c in ham_wds[:6]:
                    tw_html += f'<span class="trigger-word tw-ham" title="ham weight {c:.3f}">🟢 {w}</span>'
            if not tw_html:
                tw_html = '<span style="color:rgba(255,255,255,0.3);font-size:0.85rem;">No significant tokens found</span>'

            st.markdown(f"""
            <div class="why-card">
                <div class="why-title">🏷️ &nbsp; Key Tokens Influencing the Decision</div>
                <div style='margin-bottom:14px; line-height:2.2;'>{tw_html}</div>
                <div style='font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px;'>
                    Red = pushes toward SPAM &nbsp;·&nbsp; Green = pushes toward SAFE
                    &nbsp;·&nbsp; Hover for exact coefficient weight
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Feature breakdown bars (via components.html so they always render)
            def _bar_row(label, value, maxv, color, display):
                pct = round(min(value / maxv, 1.0) * 100, 2) if maxv > 0 else 0
                return (
                    '<div style="display:flex;align-items:center;gap:14px;margin-bottom:13px;">'
                    f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.6);min-width:168px;font-weight:500;">{label}</div>'
                    '<div style="flex:1;height:9px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden;">'
                    f'<div style="width:{pct}%;height:100%;background:{color};border-radius:999px;'
                    f'box-shadow:0 0 8px {color}99;transition:width 0.6s ease;"></div>'
                    '</div>'
                    f'<div style="font-size:0.82rem;font-weight:800;color:#fff;min-width:52px;text-align:right;">{display}</div>'
                    '</div>'
                )

            feat_rows = (
                _bar_row("📏 Message length",    exp['len'],           500,  "#a78bfa", f"{exp['len']} chars")
              + _bar_row("❗ Exclamation marks",  exp['excl'],          10,   "#ef4444", str(exp['excl']))
              + _bar_row("🔗 URL / link refs",    exp['links'],         5,    "#f59e0b", str(exp['links']))
              + _bar_row("🔠 Uppercase ratio",    exp['upper_ratio'],   1,    "#ec4899", f"{exp['upper_ratio']*100:.0f}%")
              + _bar_row("🔢 Digit ratio",        exp['digit_ratio'],   0.5,  "#06b6d4", f"{exp['digit_ratio']*100:.0f}%")
              + _bar_row("⚠️ Spam keywords hit",  exp['kw_count'],      10,   "#ef4444", str(exp['kw_count']))
            )

            components.html(
                '<div style="background:#111128;border:1px solid rgba(255,255,255,0.08);'
                'border-radius:18px;padding:22px 24px;font-family:Outfit,sans-serif;">'
                '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:2px;color:#a78bfa;margin-bottom:16px;">📊 &nbsp; Feature Score Breakdown</div>'
                + feat_rows +
                '</div>',
                height=260,
            )

            # ── Spam keywords found
            if kw_hits:
                kw_html = " ".join([f'<span class="trigger-word tw-spam">⚠ {k}</span>' for k in kw_hits])
                st.markdown(f"""
                <div class="why-card" style="margin-top:14px;">
                    <div class="why-title">🔑 &nbsp; Spam Trigger Words Found in Message</div>
                    <div style='line-height:2.4;'>{kw_html}</div>
                </div>""", unsafe_allow_html=True)

            # ── Plain English verdict
            reasons_to_show = exp['reasons_spam'] if is_spam else exp['reasons_ham']
            verdict_color   = "rgba(239,68,68,0.1)"  if is_spam else "rgba(34,197,94,0.08)"
            verdict_border  = "rgba(239,68,68,0.3)"  if is_spam else "rgba(34,197,94,0.25)"
            verdict_icon    = "🚨" if is_spam else "✅"

            if reasons_to_show:
                bullets = "".join([f"<li style='margin-bottom:8px;'>{r}</li>" for r in reasons_to_show])
                st.markdown(f"""
                <div class="why-card" style="margin-top:14px;
                     background:{verdict_color}; border-color:{verdict_border};">
                    <div class="why-title">{verdict_icon} &nbsp; Plain-English Explanation</div>
                    <ul style='margin:0; padding-left:20px; font-size:0.9rem;
                               color:rgba(255,255,255,0.75); line-height:1.8;'>
                        {bullets}
                    </ul>
                </div>""", unsafe_allow_html=True)

            # ── Cleaned text
            st.markdown(f"""
            <div class="cleaned-box">
                <div class="cleaned-label">🔎 Cleaned text fed to model</div>
                <div class="cleaned-text">{cleaned[:450]}{'…' if len(cleaned)>450 else ''}</div>
            </div>
            <div style='text-align:center;margin-top:14px;font-size:0.8rem;color:rgba(255,255,255,0.28);'>
                Threshold: <b style='color:#a78bfa;'>{threshold:.2f}</b> —
                adjust with the sidebar slider
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'history':
    st.markdown("""
    <div class="hero" style="padding:36px 44px 32px;">
        <div class="hero-eye">📜 &nbsp; Session Memory</div>
        <div class="hero-h1" style="font-size:2.4rem;">Analysis <span class="grad">History</span></div>
        <p class="hero-sub">Every message you analysed in this session, with verdicts and spam scores.</p>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:rgba(255,255,255,0.3);'>
            <div style='font-size:3rem;'>📭</div>
            <div style='margin-top:12px;font-size:1rem;'>No history yet — go to Analyse and check some messages!</div>
        </div>""", unsafe_allow_html=True)
    else:
        hcol1, hcol2 = st.columns([3,1])
        with hcol2:
            if st.button("🗑️  Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with hcol1:
            st.markdown(f"""
            <div style='font-size:0.82rem;color:rgba(255,255,255,0.4);padding-top:14px;'>
                {len(hist)} analysis result{'s' if len(hist)!=1 else ''} &nbsp;·&nbsp;
                <span style='color:#ff6b6b;'>
                    {sum(1 for h in hist if h['verdict']=='SPAM')} spam
                </span> &nbsp;/&nbsp;
                <span style='color:#4ade80;'>
                    {sum(1 for h in hist if h['verdict']=='SAFE')} safe
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        for i, h in enumerate(hist):
            is_spam = h['verdict'] == 'SPAM'
            badge   = f'<span class="hist-badge-spam">🚨 SPAM</span>' if is_spam \
                      else f'<span class="hist-badge-ham">✅ SAFE</span>'
            pct_col = "#ff6b6b" if is_spam else "#4ade80"
            st.markdown(f"""
            <div class="hist-row">
                <span class="hist-time">{h['time']}</span>
                {badge}
                <span class="hist-msg">{h['msg']}</span>
                <span class="hist-pct" style="color:{pct_col};">
                    {h['pct']:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)

            # Expandable detail
            with st.expander(f"See details #{i+1}", expanded=False):
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);border-radius:12px;
                            padding:16px;font-size:0.88rem;color:rgba(255,255,255,0.7);
                            line-height:1.75;'>
                    <b style='color:#a78bfa;'>Full message:</b><br>{h['full']}
                </div>""", unsafe_allow_html=True)

                exp = h.get('exp', {})
                kw = exp.get('kw_hits', [])
                sw = exp.get('spam_words', [])

                if kw:
                    kw_html = " ".join([f'<span class="trigger-word tw-spam">⚠ {k}</span>' for k in kw])
                    st.markdown(f"**Spam keywords found:** {kw_html}", unsafe_allow_html=True)
                if sw:
                    sw_html = " ".join([f'<span class="trigger-word tw-spam">🔴 {w[0]}</span>' for w in sw[:8]])
                    st.markdown(f"**Top spam tokens:** {sw_html}", unsafe_allow_html=True)

                reasons = exp.get('reasons_spam' if is_spam else 'reasons_ham', [])
                if reasons:
                    st.markdown("**Why this verdict:**")
                    for r in reasons:
                        st.markdown(f"• {r}", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'about':
    st.markdown("""
    <div class="hero" style="padding:36px 44px 32px;">
        <div class="hero-eye">ℹ️ &nbsp; Documentation</div>
        <div class="hero-h1" style="font-size:2.4rem;">How <span class="grad">SpamShield</span> Works</div>
        <p class="hero-sub">A full breakdown of the ML pipeline, dataset, and architecture.</p>
    </div>""", unsafe_allow_html=True)

    # Pipeline steps
    steps = [
        ("#7c3aed","1","📥 Data Preparation",
         "Raw CSV loaded with latin-1 encoding. Columns renamed to <code>text</code>/<code>label</code>. "
         "Spam→1, Ham→0. Duplicates and empty rows dropped. 5,157 clean messages retained."),
        ("#ec4899","2","🧹 Text Cleaning",
         "HTML tags stripped. URLs replaced with <code>url</code>, emails with <code>email</code>, "
         "numbers with <code>num</code>. Symbols removed. Lowercase. Extra spaces collapsed."),
        ("#06b6d4","3","✂️ Stratified Splitting",
         "70% train / 15% validation / 15% test. <code>random_state=42</code>. "
         "Spam ratio preserved exactly across all splits (12.4%). No data leakage."),
        ("#f59e0b","4","📐 TF-IDF Vectorisation",
         "Unigrams + bigrams. <code>min_df=2</code>, <code>max_features=50,000</code>. "
         "Fitted on <b>train only</b> — never touches val or test. 8,681 features used."),
        ("#22c55e","5","⚙️ Model Training",
         "Three sklearn Pipelines: Naive Bayes, Logistic Regression (GridSearchCV C∈{0.1,1,10}), "
         "LinearSVC — all with <code>class_weight='balanced'</code> for the 87/13 imbalance."),
        ("#a78bfa","6","📊 Evaluation",
         "All models evaluated on the held-out test set. Confusion matrices, ROC curves, "
         "and misclassified examples saved. Best F1: LinearSVC 0.936 / Best AUC: 0.990."),
    ]

    for color, num, title, desc in steps:
        st.markdown(f"""
        <div class="about-card">
            <div class="pipe-step" style="border-bottom:none;padding:0;">
                <div class="pipe-num" style="background:{color}22;color:{color};border:1px solid {color}44;">
                    {num}
                </div>
                <div class="pipe-info">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Model comparison table
    st.markdown("""
    <div class="about-card" style="margin-top:4px;">
        <div class="about-h">📈 Model Comparison — Test Set</div>
        <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
            <thead>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
                    <th style='text-align:left;padding:10px 12px;color:rgba(255,255,255,0.4);font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;'>Model</th>
                    <th style='padding:10px 12px;color:rgba(255,255,255,0.4);font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;'>Accuracy</th>
                    <th style='padding:10px 12px;color:rgba(255,255,255,0.4);font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;'>F1</th>
                    <th style='padding:10px 12px;color:rgba(255,255,255,0.4);font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;'>AUC</th>
                    <th style='padding:10px 12px;color:rgba(255,255,255,0.4);font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;'>FP</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
                    <td style='padding:12px;color:#a78bfa;font-weight:700;'>LinearSVC ⭐</td>
                    <td style='padding:12px;text-align:center;'>98.45%</td>
                    <td style='padding:12px;text-align:center;color:#4ade80;font-weight:700;'>0.936</td>
                    <td style='padding:12px;text-align:center;color:#4ade80;font-weight:700;'>0.990</td>
                    <td style='padding:12px;text-align:center;'>4</td>
                </tr>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
                    <td style='padding:12px;color:#f9a8d4;font-weight:700;'>Logistic Regression</td>
                    <td style='padding:12px;text-align:center;'>98.32%</td>
                    <td style='padding:12px;text-align:center;'>0.931</td>
                    <td style='padding:12px;text-align:center;'>0.989</td>
                    <td style='padding:12px;text-align:center;'>5</td>
                </tr>
                <tr>
                    <td style='padding:12px;color:#67e8f9;font-weight:700;'>Naive Bayes</td>
                    <td style='padding:12px;text-align:center;'>98.45%</td>
                    <td style='padding:12px;text-align:center;'>0.934</td>
                    <td style='padding:12px;text-align:center;'>0.984</td>
                    <td style='padding:12px;text-align:center;color:#4ade80;font-weight:700;'>1</td>
                </tr>
            </tbody>
        </table>
        <div style='font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:14px;'>
            FP = False Positives (real messages incorrectly flagged as spam — the most critical error)
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>SpamShield AI v2.0</span> &nbsp;·&nbsp;
    scikit-learn + Streamlit &nbsp;·&nbsp;
    UCI SMS Spam Collection &nbsp;·&nbsp;
    Built with 💜 for students 🚀
</div>""", unsafe_allow_html=True)
