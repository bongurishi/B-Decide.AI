"""
Intro (Splash) screen for B-Decide AI Streamlit apps.

Streamlit doesn't have traditional page routing. We implement a "first-run"
intro view using st.session_state and st.rerun().
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def render_intro(
    *,
    title: str = "B-Decide.AI",
    subheading: str = "B = MyBlood, MyBrand, MyLegacy",
    tagline: str = "From Data → Decisions → Impact",
    started_state_key: str = "bdecide_started",
) -> bool:
    """
    Render an animated full-screen intro until the user clicks Get Started.

    Returns:
        True if user already clicked Get Started in this session, else False.
        When False, this function calls st.stop() to prevent further rendering.
    """

    if st.session_state.get(started_state_key, False):
        return True

    # Hide sidebar/menu/footer for a clean full-screen intro.
    st.markdown(
        """
<style>
  /* Hide sidebar and default UI chrome on intro */
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="stHeader"] { display: none !important; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Full-screen animated gradient background */
  .stApp {
    background: radial-gradient(1200px 600px at 20% 20%, rgba(30,136,229,0.22), transparent 60%),
                radial-gradient(900px 450px at 80% 30%, rgba(0,200,150,0.16), transparent 55%),
                radial-gradient(900px 450px at 50% 80%, rgba(255,200,0,0.12), transparent 60%),
                linear-gradient(120deg, #06121f, #0a1f3a, #06121f);
    background-size: 180% 180%;
    animation: bgMove 10s ease-in-out infinite;
  }
  @keyframes bgMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* Subtle particle glow overlay (no JS needed) */
  .bdecide-particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
      radial-gradient(2px 2px at 10% 20%, rgba(255,255,255,0.22), transparent 60%),
      radial-gradient(2px 2px at 20% 70%, rgba(127,209,255,0.28), transparent 60%),
      radial-gradient(2px 2px at 30% 40%, rgba(255,255,255,0.18), transparent 60%),
      radial-gradient(2px 2px at 40% 90%, rgba(0,200,150,0.22), transparent 60%),
      radial-gradient(2px 2px at 55% 30%, rgba(255,255,255,0.16), transparent 60%),
      radial-gradient(2px 2px at 65% 60%, rgba(127,209,255,0.22), transparent 60%),
      radial-gradient(2px 2px at 78% 22%, rgba(255,255,255,0.18), transparent 60%),
      radial-gradient(2px 2px at 88% 75%, rgba(0,200,150,0.18), transparent 60%),
      radial-gradient(2px 2px at 92% 48%, rgba(255,255,255,0.14), transparent 60%);
    opacity: 0.9;
    filter: blur(0.2px);
    animation: particlesDrift 12s ease-in-out infinite;
  }
  @keyframes particlesDrift {
    0%   { transform: translate3d(0,0,0); opacity: 0.75; }
    50%  { transform: translate3d(-10px, -12px, 0); opacity: 1; }
    100% { transform: translate3d(0,0,0); opacity: 0.8; }
  }

  /* Subtle animated tech grid overlay */
  .bdecide-grid {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 64px 64px;
    mask-image: radial-gradient(circle at 50% 45%, rgba(0,0,0,1), rgba(0,0,0,0) 68%);
    opacity: 0.55;
    animation: gridDrift 14s linear infinite;
  }
  @keyframes gridDrift {
    0%   { transform: translate3d(0,0,0); }
    100% { transform: translate3d(-64px, -64px, 0); }
  }

  /* Soft scanline glow */
  .bdecide-scanline {
    position: fixed;
    left: -20%;
    right: -20%;
    top: -20%;
    height: 140px;
    pointer-events: none;
    z-index: 0;
    background: linear-gradient(90deg, transparent, rgba(127,209,255,0.12), transparent);
    filter: blur(10px);
    animation: scanMove 5.6s ease-in-out infinite;
    opacity: 0.65;
  }
  @keyframes scanMove {
    0%   { transform: translateY(-10vh) skewY(-8deg); }
    55%  { transform: translateY(54vh) skewY(-8deg); }
    100% { transform: translateY(110vh) skewY(-8deg); }
  }

  /* Animated floating "software" glyphs */
  .bdecide-float-layer {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
  }
  .bdecide-float {
    position: absolute;
    color: rgba(255,255,255,0.10);
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    animation: floatUp linear infinite;
    text-shadow: 0 0 18px rgba(30,136,229,0.20);
    user-select: none;
  }
  @keyframes floatUp {
    0%   { transform: translateY(120vh) rotate(0deg); opacity: 0; }
    10%  { opacity: 0.7; }
    100% { transform: translateY(-30vh) rotate(360deg); opacity: 0; }
  }

  /* Center card */
  .bdecide-splash {
    position: relative;
    z-index: 1;
    min-height: 92vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
  }
  .bdecide-card {
    width: min(920px, 95vw);
    border-radius: 0px;
    padding: 0.25rem 0.25rem 0.25rem;
    background: transparent;
    border: none;
    box-shadow: none;
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
    text-align: center;
  }

  /* Logo (shining + blinking + shine sweep) */
  .bdecide-logo-wrap {
    position: relative;
    display: inline-block;
  }
  .bdecide-logo {
    font-size: clamp(4.2rem, 9.6vw, 7.4rem);
    font-weight: 900;
    letter-spacing: 0.5px;
    margin: 0.2rem 0 0.4rem;
    /* Gradient text */
    background: linear-gradient(90deg, #eaf4ff 0%, #7fd1ff 35%, #00C896 75%, #eaf4ff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-shadow:
      0 0 10px rgba(30,136,229,0.55),
      0 0 22px rgba(30,136,229,0.35),
      0 0 46px rgba(0,200,150,0.20);
    animation: glowPulse 1.8s ease-in-out infinite, blinkSoft 6s ease-in-out infinite;
  }
  .bdecide-logo::after {
    content: "";
    position: absolute;
    inset: -14px -20px;
    background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.22) 45%, transparent 65%);
    transform: translateX(-120%) rotate(2deg);
    filter: blur(2px);
    animation: shineSweep 3.4s ease-in-out infinite;
    mix-blend-mode: screen;
    pointer-events: none;
  }
  @keyframes shineSweep {
    0%   { transform: translateX(-140%) rotate(2deg); opacity: 0.0; }
    20%  { opacity: 0.45; }
    55%  { opacity: 0.2; }
    100% { transform: translateX(140%) rotate(2deg); opacity: 0.0; }
  }
  @keyframes glowPulse {
    0%, 100% { filter: drop-shadow(0 0 0 rgba(30,136,229,0)); transform: translateY(0); }
    50%      { filter: drop-shadow(0 0 18px rgba(30,136,229,0.55)); transform: translateY(-1px); }
  }
  @keyframes blinkSoft {
    0%, 7%, 100% { opacity: 1; }
    8% { opacity: 0.65; }
    9% { opacity: 1; }
    10% { opacity: 0.78; }
    11% { opacity: 1; }
  }

  .bdecide-subheading {
    font-size: clamp(1.05rem, 2.1vw, 1.35rem);
    font-weight: 700;
    color: rgba(255,255,255,0.85);
    margin: 0.2rem 0 0.6rem;
  }

  .bdecide-tagline {
    font-size: clamp(1.15rem, 2.6vw, 1.7rem);
    font-weight: 800;
    color: #7fd1ff;
    margin: 0.4rem 0 1.2rem;
    text-shadow: 0 0 14px rgba(30,136,229,0.25);
    display: inline-block;
    position: relative;
  }
  /* Tagline reveal animation */
  .bdecide-tagline::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -8px;
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, rgba(127,209,255,0), rgba(127,209,255,0.85), rgba(0,200,150,0.85), rgba(127,209,255,0));
    transform: scaleX(0);
    transform-origin: center;
    animation: underlineGrow 1.2s ease forwards;
    animation-delay: 0.25s;
    opacity: 0.85;
  }
  @keyframes underlineGrow {
    to { transform: scaleX(1); }
  }

  .bdecide-desc {
    font-size: 1.02rem;
    color: rgba(255,255,255,0.78);
    line-height: 1.6;
    margin: 0.2rem auto 1.2rem;
    max-width: 58ch;
  }

  /* Premium feature chips */
  .bdecide-chips {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 1.1rem auto 1.35rem;
    width: min(920px, 96vw);
  }
  @media (max-width: 840px) {
    .bdecide-chips { grid-template-columns: 1fr; }
  }
  .bdecide-chip {
    padding: 14px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 40px rgba(0,0,0,0.22);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    text-align: left;
    transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
  }
  .bdecide-chip:hover {
    transform: translateY(-2px);
    border-color: rgba(127,209,255,0.32);
    background: rgba(255,255,255,0.075);
  }
  .bdecide-chip-title {
    font-weight: 800;
    color: rgba(255,255,255,0.92);
    margin-bottom: 6px;
    letter-spacing: 0.2px;
  }
  .bdecide-chip-desc {
    color: rgba(255,255,255,0.72);
    font-size: 0.95rem;
    line-height: 1.5;
  }

  .bdecide-cta {
    margin: 0.2rem auto 0.8rem;
    color: rgba(255,255,255,0.80);
    font-weight: 650;
    letter-spacing: 0.2px;
  }

  /* Style Streamlit button */
  .stButton > button {
    width: 100%;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    background: linear-gradient(90deg, #1E88E5, #00C896) !important;
    color: white !important;
    box-shadow: 0 12px 30px rgba(30,136,229,0.22) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 40px rgba(30,136,229,0.28) !important;
  }
  /* Pulsing glow around the CTA button container */
  .bdecide-btn-wrap {
    position: relative;
  }
  .bdecide-btn-wrap::before {
    content: "";
    position: absolute;
    inset: -10px;
    border-radius: 18px;
    background: radial-gradient(circle at 50% 50%, rgba(127,209,255,0.28), transparent 60%);
    filter: blur(10px);
    animation: btnPulse 1.9s ease-in-out infinite;
    z-index: -1;
  }
  @keyframes btnPulse {
    0%, 100% { opacity: 0.55; transform: scale(0.98); }
    50%      { opacity: 1; transform: scale(1.02); }
  }
</style>
        """,
        unsafe_allow_html=True,
    )

    # Floating glyph layer (positions + durations are hard-coded for stability).
    floats = [
        ("</>", "6%", "10.8s", "1.35rem"),
        ("{ }", "14%", "13.5s", "1.8rem"),
        ("AI", "22%", "11.7s", "2.0rem"),
        ("API", "31%", "15.2s", "1.55rem"),
        ("ML", "41%", "12.4s", "1.9rem"),
        ("DATA", "52%", "14.0s", "1.45rem"),
        ("PIPELINE", "62%", "16.4s", "1.15rem"),
        ("CLOUD", "70%", "12.8s", "1.2rem"),
        ("SECURE", "78%", "15.6s", "1.15rem"),
        ("λ", "86%", "16.2s", "2.1rem"),
        ("Δ", "92%", "12.9s", "1.7rem"),
        ("MODEL", "96%", "14.9s", "1.1rem"),
    ]
    floats_html = "\n".join(
        f'<div class="bdecide-float" style="left:{left}; animation-duration:{dur}; font-size:{size};">{txt}</div>'
        for (txt, left, dur, size) in floats
    )

    st.markdown(
        f"""
<div class="bdecide-particles"></div>
<div class="bdecide-grid"></div>
<div class="bdecide-scanline"></div>

<div class="bdecide-float-layer">
  {floats_html}
</div>

<div class="bdecide-splash">
<div class="bdecide-card">
<div class="bdecide-logo-wrap"><div class="bdecide-logo">{title}</div></div>
<div class="bdecide-subheading">{subheading}</div>
<div class="bdecide-tagline">{tagline}</div>
<div class="bdecide-desc">A premium Decision Intelligence platform that turns customer data into churn predictions, smart retention actions, and clear explanations — built for real business impact.</div>
<div class="bdecide-chips">
  <div class="bdecide-chip">
    <div class="bdecide-chip-title">Predict</div>
    <div class="bdecide-chip-desc">High-accuracy churn probability powered by ML.</div>
  </div>
  <div class="bdecide-chip">
    <div class="bdecide-chip-title">Recommend</div>
    <div class="bdecide-chip-desc">Dynamic, rule-driven actions with fuzzy logic control.</div>
  </div>
  <div class="bdecide-chip">
    <div class="bdecide-chip-title">Explain</div>
    <div class="bdecide-chip-desc">Human-friendly explanations you can share with teams.</div>
  </div>
</div>
<div class="bdecide-cta">Ready to turn retention into revenue? Start your analysis now.</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Centered Get Started button at the end.
    col_left, col_mid, col_right = st.columns([2, 3, 2])
    with col_mid:
        st.markdown('<div class="bdecide-btn-wrap">', unsafe_allow_html=True)
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state[started_state_key] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Stop Streamlit from rendering the rest of the app until started.
    st.stop()


