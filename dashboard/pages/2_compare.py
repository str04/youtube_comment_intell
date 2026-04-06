import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random, hashlib, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Compare · YT Intel", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
:root{--bg:#0d0f14;--surface:#151820;--surface2:#1c2030;--border:#2a2f42;
      --accent:#6c63ff;--accent2:#ff6584;--green:#1fd99a;--yellow:#f5c542;
      --red:#ff5b5b;--text:#e8eaf0;--muted:#7a7f95;--radius:12px;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;
  background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebarNav"]{display:none!important;}
.block-container{padding:1.5rem 2rem!important;max-width:1400px;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
input,div[data-baseweb="input"] input{background:var(--surface2)!important;
  border:1px solid var(--border)!important;color:var(--text)!important;border-radius:var(--radius)!important;
  font-family:'Space Grotesk',sans-serif!important;}
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;
  border-radius:var(--radius)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:600!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.85!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;padding:1rem!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}
button[data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif!important;color:var(--muted)!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
hr{border-color:var(--border)!important;}
.yt-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.2rem 1.4rem;margin-bottom:1rem;}
.win-badge{display:inline-block;background:#1fd99a22;color:#1fd99a;
  border:1px solid #1fd99a44;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:1.4rem;font-weight:700;padding:1rem 0 1.5rem'>🎯 YT Intel</div>",
                unsafe_allow_html=True)
    st.page_link("app.py",             label="🏠  Home")
    st.page_link("pages/1_analyze.py", label="🔍  Analyze Video")
    st.page_link("pages/2_compare.py", label="⚔️  Compare Videos")
    st.page_link("pages/3_about.py",   label="ℹ️  About")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#e8eaf0"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def mock_analysis(url, seed_extra=""):
    random.seed(hashlib.md5((url + seed_extra).encode()).hexdigest())
    sentiments = ["positive", "negative", "neutral"]
    emotions   = ["joy", "anger", "sadness", "surprise", "fear", "disgust", "neutral"]
    intents    = ["praise", "complaint", "question", "suggestion", "general"]

    n = random.randint(80, 200)
    comments = []
    for _ in range(n):
        sent = random.choices(sentiments, weights=[random.randint(40,70), random.randint(10,30), 25])[0]
        comments.append({
            "sentiment":        sent,
            "dominant_emotion": random.choice(emotions),
            "intent":           random.choice(intents),
            "is_toxic":         random.random() < 0.06,
            "is_sarcastic":     random.random() < 0.08,
            "like_count":       random.randint(0, 300),
            "positive_score":   random.uniform(0,1),
            "negative_score":   random.uniform(0,1),
        })

    sc = {s: sum(1 for c in comments if c["sentiment"] == s) for s in sentiments}
    ec = {e: sum(1 for c in comments if c["dominant_emotion"] == e) for e in emotions}
    ic = {i: sum(1 for c in comments if c["intent"] == i) for i in intents}
    pos = sc.get("positive", 0)
    approval = round(pos / len(comments) * 100, 1)

    titles = ["Amazing Tutorial You Need to Watch", "Controversial Take on Modern Tech",
              "Best Of 2024 — Year Review", "Deep Dive: Advanced Concepts Explained"]
    channels = ["TechInsider", "CreatorPro", "StudyWithMe", "DataNerd"]

    return {
        "title":    random.choice(titles),
        "channel":  random.choice(channels),
        "view_count": random.randint(10_000, 2_000_000),
        "like_count": random.randint(500, 50_000),
        "thumbnail": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    }, {
        "enriched_comments": comments,
        "approval_score":    approval,
        "toxic_count":       sum(1 for c in comments if c["is_toxic"]),
        "sarcasm_count":     sum(1 for c in comments if c["is_sarcastic"]),
        "sentiment_counts":  sc,
        "emotion_counts":    ec,
        "intent_counts":     ic,
        "comparison_summary": f"""Video A had an approval score of {approval}%. The audience showed strong emotional engagement with notably high joy and praise intent. Critics were present but in the minority. Recommended action: lean into the content style that generated the most praise comments.""",
    }


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## ⚔️ Compare Two Videos")
st.markdown("<div style='color:var(--muted);margin-bottom:1.5rem'>Side-by-side NLP breakdown of any two YouTube videos</div>",
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🅰️ Video 1")
    url1 = st.text_input("URL for Video 1", placeholder="https://youtube.com/watch?v=...",
                         key="url1", label_visibility="collapsed")
with c2:
    st.markdown("#### 🅱️ Video 2")
    url2 = st.text_input("URL for Video 2", placeholder="https://youtube.com/watch?v=...",
                         key="url2", label_visibility="collapsed")

_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    compare_btn = st.button("⚔️ Run Comparison", use_container_width=True)

st.divider()

if compare_btn and url1 and url2:
    with st.spinner("Running analysis on both videos…"):
        meta1, an1 = mock_analysis(url1, "A")
        meta2, an2 = mock_analysis(url2, "B")
    st.session_state["cmp"] = (meta1, an1, meta2, an2)

elif compare_btn:
    st.warning("Please enter URLs for both videos.")

if "cmp" in st.session_state:
    meta1, an1, meta2, an2 = st.session_state["cmp"]

    # ── Video headers ─────────────────────────────────────────────────────────
    h1, vs, h2 = st.columns([5, 1, 5])
    for col, meta, an, label, color in [
        (h1, meta1, an1, "🅰️", "#6c63ff"),
        (h2, meta2, an2, "🅱️", "#ff6584"),
    ]:
        with col:
            st.markdown(f"""
            <div class='yt-card' style='border-top:3px solid {color}'>
                <div style='font-size:.75rem;color:{color};font-weight:600;margin-bottom:6px'>{label}</div>
                <div style='font-weight:700;font-size:1rem'>{meta['title']}</div>
                <div style='color:var(--muted);font-size:.82rem;margin-top:4px'>📺 {meta['channel']}</div>
            </div>
            """, unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("👁️", f"{meta['view_count']:,}")
            m2.metric("💬", f"{len(an['enriched_comments'])}")
            m3.metric("✅", f"{an['approval_score']}%")
    with vs:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:2rem'>VS</div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Comparison tabs ───────────────────────────────────────────────────────
    tab_overview, tab_sentiment, tab_emotion, tab_intent, tab_ai = st.tabs([
        "📊 Overview", "😊 Sentiment", "😮 Emotions", "💬 Intent", "🤖 AI Summary"
    ])

    # Overview — approval + toxic + sarcasm
    with tab_overview:
        metrics = [
            ("✅ Approval Score",   an1["approval_score"],  an2["approval_score"],  "%"),
            ("☠️ Toxic Comments",   an1["toxic_count"],     an2["toxic_count"],     ""),
            ("🙄 Sarcastic",        an1.get("sarcasm_count",0), an2.get("sarcasm_count",0), ""),
            ("💬 Total Comments",   len(an1["enriched_comments"]), len(an2["enriched_comments"]), ""),
        ]
        for label, v1, v2, suffix in metrics:
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"<div style='padding:.6rem 0;color:var(--muted)'>{label}</div>",
                            unsafe_allow_html=True)
            with c2:
                winner = v1 > v2 if label != "☠️ Toxic Comments" else v1 < v2
                badge  = "<span class='win-badge'>winner</span>" if winner else ""
                st.markdown(f"""
                <div style='text-align:center;background:var(--surface);border:1px solid #6c63ff44;
                     border-radius:8px;padding:.5rem;font-weight:700;font-size:1.1rem'>
                    {v1}{suffix} {badge}
                </div>""", unsafe_allow_html=True)
            with c3:
                winner2 = v2 > v1 if label != "☠️ Toxic Comments" else v2 < v1
                badge2  = "<span class='win-badge'>winner</span>" if winner2 else ""
                st.markdown(f"""
                <div style='text-align:center;background:var(--surface);border:1px solid #ff658444;
                     border-radius:8px;padding:.5rem;font-weight:700;font-size:1.1rem'>
                    {v2}{suffix} {badge2}
                </div>""", unsafe_allow_html=True)
            st.markdown("<hr style='margin:0;border-color:var(--border)'>", unsafe_allow_html=True)

    # Sentiment comparison
    with tab_sentiment:
        c1, c2 = st.columns(2)
        for col, an, title, color in [(c1, an1, "🅰️ Video 1", "#6c63ff"), (c2, an2, "🅱️ Video 2", "#ff6584")]:
            with col:
                sc = an["sentiment_counts"]
                donut = go.Figure(go.Pie(
                    labels=list(sc.keys()), values=list(sc.values()), hole=0.55,
                    marker_colors=["#1fd99a","#ff5b5b","#7a7f95"],
                    textfont_color="#e8eaf0",
                ))
                donut.update_layout(title=title, **PLOTLY_LAYOUT,
                                    legend=dict(font=dict(color="#e8eaf0")))
                st.plotly_chart(donut, use_container_width=True)

        # Grouped bar
        sent_labels = ["positive", "negative", "neutral"]
        sc1 = an1["sentiment_counts"]
        sc2 = an2["sentiment_counts"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Video 1", x=sent_labels,
                             y=[sc1.get(s,0) for s in sent_labels],
                             marker_color="#6c63ff"))
        fig.add_trace(go.Bar(name="Video 2", x=sent_labels,
                             y=[sc2.get(s,0) for s in sent_labels],
                             marker_color="#ff6584"))
        fig.update_layout(title="Sentiment Side-by-Side", barmode="group", **PLOTLY_LAYOUT,
                          legend=dict(font=dict(color="#e8eaf0")))
        st.plotly_chart(fig, use_container_width=True)

    # Emotion comparison
    with tab_emotion:
        emotions = ["joy","anger","sadness","surprise","fear","disgust","neutral"]
        ec1 = an1["emotion_counts"]
        ec2 = an2["emotion_counts"]

        radar = go.Figure()
        for label, ec, color in [("Video 1", ec1, "#6c63ff"), ("Video 2", ec2, "#ff6584")]:
            vals = [ec.get(e, 0) for e in emotions]
            radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=emotions + [emotions[0]],
                fill="toself", name=label,
                fillcolor=f"{color}33", line=dict(color=color, width=2),
            ))
        radar.update_layout(title="Emotion Radar Overlay", **PLOTLY_LAYOUT,
            polar=dict(bgcolor="#1c2030",
                       radialaxis=dict(gridcolor="#2a2f42", color="#7a7f95"),
                       angularaxis=dict(gridcolor="#2a2f42", color="#e8eaf0")),
            legend=dict(font=dict(color="#e8eaf0")))
        st.plotly_chart(radar, use_container_width=True)

    # Intent comparison
    with tab_intent:
        intents = ["praise","complaint","question","suggestion","general"]
        ic1 = an1["intent_counts"]
        ic2 = an2["intent_counts"]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Video 1", x=intents,
                              y=[ic1.get(i,0) for i in intents], marker_color="#6c63ff"))
        fig2.add_trace(go.Bar(name="Video 2", x=intents,
                              y=[ic2.get(i,0) for i in intents], marker_color="#ff6584"))
        fig2.update_layout(title="Intent Comparison", barmode="group", **PLOTLY_LAYOUT,
                           legend=dict(font=dict(color="#e8eaf0")))
        st.plotly_chart(fig2, use_container_width=True)

        # Praise-to-complaint ratio
        c1, c2 = st.columns(2)
        for col, ic, an, label in [(c1, ic1, an1, "🅰️"), (c2, ic2, an2, "🅱️")]:
            ratio = round(ic.get("praise",1) / max(ic.get("complaint",1), 1), 2)
            with col:
                st.metric(f"{label} Praise / Complaint Ratio", ratio,
                          "higher = more praise" if ratio > 1 else "more complaints than praise")

    # AI Summary
    with tab_ai:
        st.markdown("#### 🤖 AI Comparison Summary")
        st.markdown(f"""
        <div class='yt-card' style='border-left:3px solid var(--accent);
             white-space:pre-line;font-size:.9rem;line-height:1.7'>
{an1.get("comparison_summary", "No summary available.")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Key Metrics Table")
        compare_table = pd.DataFrame({
            "Metric":   ["Approval Score","Total Comments","Toxic Comments","Sarcastic Comments",
                          "Top Sentiment","Top Emotion"],
            "Video A":  [f"{an1['approval_score']}%", len(an1["enriched_comments"]),
                          an1["toxic_count"], an1.get("sarcasm_count",0),
                          max(an1["sentiment_counts"], key=an1["sentiment_counts"].get),
                          max(an1["emotion_counts"],   key=an1["emotion_counts"].get)],
            "Video B":  [f"{an2['approval_score']}%", len(an2["enriched_comments"]),
                          an2["toxic_count"], an2.get("sarcasm_count",0),
                          max(an2["sentiment_counts"], key=an2["sentiment_counts"].get),
                          max(an2["emotion_counts"],   key=an2["emotion_counts"].get)],
        })
        st.dataframe(compare_table, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 0;color:var(--muted)'>
        <div style='font-size:3rem'>⚔️</div>
        <div style='font-size:1.1rem;margin-top:1rem'>Enter two YouTube URLs above and hit <b>Run Comparison</b></div>
    </div>
    """, unsafe_allow_html=True)