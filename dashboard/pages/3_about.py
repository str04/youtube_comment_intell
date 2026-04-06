import streamlit as st

st.set_page_config(page_title="About · YT Intel", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#0d0f14;--surface:#151820;--surface2:#1c2030;--border:#2a2f42;
      --accent:#6c63ff;--accent2:#ff6584;--green:#1fd99a;--yellow:#f5c542;
      --red:#ff5b5b;--text:#e8eaf0;--muted:#7a7f95;--radius:12px;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;
  background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebarNav"]{display:none!important;}
.block-container{padding:1.5rem 2rem!important;max-width:1000px;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
hr{border-color:var(--border)!important;}
.yt-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.4rem;margin-bottom:1rem;}
.mono{font-family:'JetBrains Mono',monospace;background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;padding:2px 8px;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:1.4rem;font-weight:700;padding:1rem 0 1.5rem'>🎯 YT Intel</div>",
                unsafe_allow_html=True)
    st.page_link("app.py",             label="🏠  Home")
    st.page_link("pages/1_analyze.py", label="🔍  Analyze Video")
    st.page_link("pages/2_compare.py", label="⚔️  Compare Videos")
    st.page_link("pages/3_about.py",   label="ℹ️  About")

st.markdown("## ℹ️ About This Project")
st.markdown("<div style='color:var(--muted);margin-bottom:2rem'>YouTube Comment Intelligence — a Data Science capstone project</div>",
            unsafe_allow_html=True)

# Architecture
st.markdown("### 🏗️ System Architecture")
st.code("""youtube-comment-intelligence/
├── ingestion/
│   ├── youtube_api.py            ← yt-dlp comment fetching
│   ├── comment_parser.py         ← clean · deduplicate · thread
│   └── mongo_store.py            ← MongoDB persistence
├── nlp/
│   ├── sentiment.py              ← cardiffnlp/twitter-roberta
│   ├── topic_model.py            ← BERTopic clustering
│   ├── toxicity.py               ← unitary/toxic-bert
│   ├── sarcasm.py                ← cardiffnlp/twitter-roberta-irony
│   ├── intent_classifier.py      ← rule-based (question/praise/complaint)
│   └── hinglish.py               ← XLM-RoBERTa multilingual
├── ml/
│   ├── audience_segmentation.py  ← KMeans (4 segments)
│   ├── quality_scorer.py         ← composite score
│   └── viral_predictor.py        ← GradientBoosting like-count predictor
├── ai/
│   ├── langchain_summary.py      ← Groq LLaMA3-70B creator report
│   └── content_gap_finder.py     ← Groq LLaMA3-70B gap analysis
└── dashboard/
    ├── app.py                    ← Home (this app)
    └── pages/
        ├── 1_analyze.py          ← Full analysis page
        ├── 2_compare.py          ← Side-by-side comparison
        └── 3_about.py            ← This page""", language="bash")

st.divider()

# Models
st.markdown("### 🤖 Models Used")
models = [
    ("Sentiment",    "cardiffnlp/twitter-roberta-base-sentiment-latest", "positive · negative · neutral scores"),
    ("Emotions",     "j-hartmann/emotion-english-distilroberta-base",    "joy · anger · sadness · surprise · fear · disgust"),
    ("Toxicity",     "unitary/toxic-bert",                               "binary toxic classifier"),
    ("Sarcasm",      "cardiffnlp/twitter-roberta-base-irony",            "irony / non-irony"),
    ("Multilingual", "cardiffnlp/twitter-xlm-roberta-base-sentiment",    "Hinglish / Hindi support"),
    ("Topic Model",  "BERTopic (multilingual)",                          "unsupervised topic discovery"),
    ("LLM Report",   "Groq · LLaMA3-70B",                               "creator summaries & gap analysis"),
]
for name, model_id, desc in models:
    st.markdown(f"""
    <div class='yt-card' style='display:flex;align-items:flex-start;gap:1rem;padding:.9rem 1.2rem'>
        <div style='min-width:110px;font-weight:600;color:var(--accent)'>{name}</div>
        <div>
            <span class='mono'>{model_id}</span>
            <div style='color:var(--muted);font-size:.82rem;margin-top:4px'>{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Pipeline flow
st.markdown("### ⚙️ Analysis Pipeline")
steps = [
    ("1", "Fetch",       "yt-dlp pulls up to 500 top comments + metadata from any public YouTube URL"),
    ("2", "Parse",       "Remove URLs, normalize whitespace, demojize, deduplicate, filter empty"),
    ("3", "Language",    "langdetect + Hinglish keyword markers route comments to correct model"),
    ("4", "NLP Batch",   "Sentiment · Emotion · Toxicity · Sarcasm run in batched inference (batch_size=32)"),
    ("5", "Intent",      "Rule-based classifier tags each comment: question / praise / complaint / suggestion"),
    ("6", "Topics",      "BERTopic clusters comment text into thematic groups with keyword labels"),
    ("7", "Segments",    "KMeans (k=4) clusters comments into audience personas on NLP feature vectors"),
    ("8", "Score",       "Quality scorer ranks comments by length · likes · sentiment · toxicity penalty"),
    ("9", "AI Report",   "Enriched analysis JSON sent to Groq LLaMA3 for creator summary + content gaps"),
    ("10", "Cache",      "Full analysis stored in MongoDB — repeat URL loads instantly from cache"),
]
for num, title, desc in steps:
    st.markdown(f"""
    <div style='display:flex;gap:1rem;margin-bottom:.8rem;align-items:flex-start'>
        <div style='min-width:32px;height:32px;background:var(--accent);border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:.8rem;flex-shrink:0'>{num}</div>
        <div>
            <div style='font-weight:600'>{title}</div>
            <div style='color:var(--muted);font-size:.85rem'>{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Setup instructions
st.markdown("### 🚀 Setup & Run")
st.code("""# 1. Clone & install
pip install -r requirements.txt

# 2. Create .env
GROQ_API_KEY=your_groq_key_here
MONGO_URI=mongodb://localhost:27017

# 3. Run the dashboard
streamlit run dashboard/app.py
""", language="bash")

st.markdown("""
<div class='yt-card' style='border-left:3px solid var(--yellow);margin-top:1rem'>
    <b>⚠️ Note:</b> The dashboard currently runs with <b>mock data</b>. 
    To wire in the real pipeline, replace the <code style='background:var(--surface2);
    padding:2px 6px;border-radius:4px'>run_mock_analysis()</code> function in 
    <code>pages/1_analyze.py</code> with actual calls to your 
    <code>ingestion/</code>, <code>nlp/</code>, <code>ml/</code>, and <code>ai/</code> modules.
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='color:var(--muted);font-size:.82rem;text-align:center'>"
            "Built as a Data Science project · Streamlit + Plotly + HuggingFace + BERTopic + Groq"
            "</div>", unsafe_allow_html=True)
