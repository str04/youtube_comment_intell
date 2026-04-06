import streamlit as st

st.set_page_config(
    page_title="YT Comment Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --surface2:  #1c2030;
    --border:    #2a2f42;
    --accent:    #6c63ff;
    --accent2:   #ff6584;
    --green:     #1fd99a;
    --yellow:    #f5c542;
    --red:       #ff5b5b;
    --text:      #e8eaf0;
    --muted:     #7a7f95;
    --radius:    12px;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hide auto-generated page nav links at top of sidebar */
[data-testid="stSidebarNav"] { display: none !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Inputs */
input, textarea, select,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity .2s ease !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--muted) !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expanders */
details, summary {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: var(--radius) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Selectbox */
div[data-baseweb="select"] * {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Alerts */
.stAlert { border-radius: var(--radius) !important; }

/* Custom card */
.yt-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.yt-card-accent {
    border-left: 3px solid var(--accent);
}
.tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: var(--muted);
    margin: 2px;
}
.tag-positive { border-color: #1fd99a44; color: var(--green); background: #1fd99a11; }
.tag-negative { border-color: #ff5b5b44; color: var(--red);   background: #ff5b5b11; }
.tag-neutral  { border-color: #7a7f9544; color: var(--muted); }
.tag-sarcasm  { border-color: #f5c54244; color: var(--yellow); background: #f5c54211; }
.segment-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #fff;
}
.mono { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem'>
        <div style='font-size:1.5rem;font-weight:700;letter-spacing:-0.5px'>
            🎯 YT Intel
        </div>
        <div style='font-size:0.78rem;color:var(--muted);margin-top:4px'>
            Comment Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("app.py",                       label="🏠  Home",          )
    st.page_link("pages/1_analyze.py",            label="🔍  Analyze Video"  )
    st.page_link("pages/2_compare.py",            label="⚔️  Compare Videos" )
    st.page_link("pages/3_about.py",              label="ℹ️  About"          )

    st.divider()
    st.markdown("<div style='color:var(--muted);font-size:0.75rem'>v1.0 · Built with ❤️</div>",
                unsafe_allow_html=True)

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:3rem 0 2rem;text-align:center'>
    <div style='font-size:3.5rem;font-weight:800;letter-spacing:-2px;line-height:1.1;
                background:linear-gradient(135deg,#6c63ff,#ff6584);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        YouTube Comment<br>Intelligence
    </div>
    <div style='color:var(--muted);font-size:1.1rem;margin-top:1rem;max-width:600px;margin-inline:auto'>
        Deep NLP analysis of YouTube comments — sentiment, emotions, topics,
        audience segments, toxicity, sarcasm & AI-powered creator insights.
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
cols = st.columns(3)
features = [
    ("🧠", "NLP Pipeline",   "Sentiment · Emotion · Sarcasm · Toxicity · Intent · Topic Modeling"),
    ("👥", "Audience Intel", "KMeans segmentation into Super Fans · Critics · Curious Minds · Casual"),
    ("🤖", "AI Insights",    "Groq LLM generates creator reports & content gap recommendations"),
    ("📊", "Rich Visuals",   "Gauges · Treemaps · Emotion radar · Word clouds · Timeline charts"),
    ("⚔️", "Video Compare",  "Side-by-side NLP breakdown of any two YouTube videos"),
    ("🌐", "Hinglish Ready", "Multilingual BERT handles Roman Hindi / code-switched comments"),
]
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div class='yt-card yt-card-accent' style='min-height:120px'>
            <div style='font-size:1.6rem'>{icon}</div>
            <div style='font-weight:600;margin:6px 0 4px'>{title}</div>
            <div style='font-size:0.82rem;color:var(--muted)'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("### 🚀 Quick Start")
    st.markdown("Paste any YouTube URL in the **Analyze Video** page and hit **Run Analysis**. Results are cached in MongoDB so repeat analyses are instant.")
with c2:
    if st.button("🔍  Analyze a Video", use_container_width=True):
        st.switch_page("pages/1_analyze.py")
    if st.button("⚔️  Compare Two Videos", use_container_width=True):
        st.switch_page("pages/2_compare.py")
