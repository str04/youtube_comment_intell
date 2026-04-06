import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os

# ── Path setup — bulletproof root detection ───────────────────────────────────
def find_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if all(os.path.isdir(os.path.join(current, d)) for d in ["nlp", "ml", "ai"]):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = find_project_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analyze · YT Intel", page_icon="🔍", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
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
.block-container{padding:1.5rem 2rem!important;max-width:1400px;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
input,textarea,div[data-baseweb="input"] input{background:var(--surface2)!important;
  border:1px solid var(--border)!important;color:var(--text)!important;border-radius:var(--radius)!important;
  font-family:'Space Grotesk',sans-serif!important;}
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;
  border-radius:var(--radius)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:600!important;padding:.6rem 1.4rem!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.85!important;}
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;padding:1rem!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}
button[data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif!important;color:var(--muted)!important;background:transparent!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
div[data-baseweb="select"] *{background:var(--surface2)!important;color:var(--text)!important;
  border-color:var(--border)!important;font-family:'Space Grotesk',sans-serif!important;}
hr{border-color:var(--border)!important;}
.yt-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.2rem 1.4rem;margin-bottom:1rem;}
.yt-card-accent{border-left:3px solid var(--accent);}
.tag{display:inline-block;background:var(--surface2);border:1px solid var(--border);
  border-radius:20px;padding:2px 10px;font-size:.75rem;color:var(--muted);margin:2px;}
.tag-positive{border-color:#1fd99a44;color:#1fd99a;background:#1fd99a11;}
.tag-negative{border-color:#ff5b5b44;color:#ff5b5b;background:#ff5b5b11;}
.tag-neutral{border-color:#7a7f9544;color:#7a7f95;}
.tag-sarcasm{border-color:#f5c54244;color:#f5c542;background:#f5c54211;}
.tag-toxic{border-color:#ff5b5b44;color:#ff5b5b;background:#ff5b5b22;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:1.4rem;font-weight:700;padding:1rem 0 1.5rem'>🎯 YT Intel</div>",
                unsafe_allow_html=True)
    st.page_link("app.py",             label="🏠  Home")
    st.page_link("pages/1_analyze.py", label="🔍  Analyze Video")
    st.page_link("pages/2_compare.py", label="⚔️  Compare Videos")
    st.page_link("pages/3_about.py",   label="ℹ️  About")
    st.divider()
    st.markdown("**⚙️ Settings**")
    max_comments = st.slider("Max comments to fetch", 50, 500, 200, 50)
    show_replies = st.checkbox("Include replies", value=False)
    use_cache    = st.checkbox("Use cached results", value=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#e8eaf0"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ── Real pipeline ─────────────────────────────────────────────────────────────
def run_analysis(url: str, max_c: int, use_cache: bool = True) -> tuple[dict, dict]:
    from ingestion.youtube_api    import fetch_all, extract_video_id
    from ingestion.comment_parser import parse_comments
    from nlp.sentiment            import analyze_sentiment, analyze_emotions
    from nlp.toxicity             import analyze_toxicity
    from nlp.sarcasm              import detect_sarcasm
    from nlp.intent_classifier    import classify_intents
    from nlp.topic_model          import run_topic_modeling
    from nlp.hinglish             import detect_languages, analyze_hinglish_sentiment
    from ml.audiance_segmentation import segment_audience, get_segment_summary
    from ml.quality_scorer        import score_all_comments
    from ml.viral_predictor       import predict_virality
    from ai_modules.langchain_summary  import generate_creator_summary
    from ai_modules.content_gap_finder import find_content_gaps
    from nlp.ensemble_classifier        import ensemble_classify
    from ingestion.mongo_sttore    import (
        save_video, save_comments, save_analysis,
        load_analysis, analysis_exists,
    )

    # Step 0 — cache check
    video_id = extract_video_id(url)
    if use_cache and video_id and analysis_exists(video_id):
        cached = load_analysis(video_id)
        if cached and "meta" in cached and "analysis" in cached:
            st.info("⚡ Loaded from cache (MongoDB). Uncheck 'Use cached results' to re-run.")
            return cached["meta"], cached["analysis"]

    # Step 1 — fetch (single yt-dlp call for both metadata + comments)
    with st.status("📡 Fetching video & comments in one call…", expanded=True) as status:
        meta, raw = fetch_all(url, max_comments=max_c)
        video_id  = meta["video_id"]

        if not raw:
            st.error("❌ No comments found. The video may have comments disabled.")
            st.stop()

        status.update(label="🧹 Parsing & cleaning comments…")
        comments = parse_comments(raw)

        if not comments:
            st.error("❌ All comments were filtered out. Try a different video.")
            st.stop()

        st.write(f"✅ Fetched **{len(comments)}** comments for: **{meta.get('title','')[:60]}**")

    # Step 2 — texts
    texts = [c.get("clean_text") or c.get("text", "") for c in comments]

    # Step 3 — language detection
    with st.status("🌐 Detecting languages…"):
        languages = detect_languages(texts)
        en_idx    = [i for i, l in enumerate(languages) if l not in ("hinglish", "hi", "ur")]
        multi_idx = [i for i, l in enumerate(languages) if l in ("hinglish", "hi", "ur")]
        st.write(f"  English: {len(en_idx)} · Hinglish/Hindi: {len(multi_idx)}")

    # Step 4 — sentiment
    sent_results = [{}] * len(comments)
    with st.status("😊 Running sentiment analysis…"):
        if en_idx:
            en_sents = analyze_sentiment([texts[i] for i in en_idx])
            for j, i in enumerate(en_idx):
                sent_results[i] = en_sents[j]
        if multi_idx:
            ml_sents = analyze_hinglish_sentiment([texts[i] for i in multi_idx])
            for j, i in enumerate(multi_idx):
                sent_results[i] = ml_sents[j]

    # Step 5 — emotions
    with st.status("😮 Running emotion detection…"):
        emo_results = analyze_emotions(texts, sentiments=sent_results)

    # Step 6 — toxicity + sarcasm
    with st.status("☠️ Checking toxicity & sarcasm…"):
        tox_results  = analyze_toxicity(texts)
        sarc_results = detect_sarcasm(texts)

    # Step 7 — intent
    with st.status("💬 Classifying intents…"):
        intents = classify_intents(texts)

    # Step 8 — merge
    for i, c in enumerate(comments):
        c.update(sent_results[i])
        c.update(emo_results[i])
        c.update(tox_results[i])
        c.update(sarc_results[i])
        c["intent"]   = intents[i]
        c["language"] = languages[i]
        # safe defaults
        c.setdefault("sentiment",        "neutral")
        c.setdefault("positive_score",   0.0)
        c.setdefault("negative_score",   0.0)
        c.setdefault("neutral_score",    1.0)
        c.setdefault("dominant_emotion", "neutral")
        c.setdefault("is_toxic",         False)
        c.setdefault("toxic_score",      0.0)
        c.setdefault("is_sarcastic",     False)
        c.setdefault("sarcasm_score",    0.0)
        c.setdefault("like_count",       0)

    # Step 9 — Ensemble emotion classification
    with st.status("🎯 Running ensemble emotion classifier (3 votes + Groq)…"):
        try:
            comments = ensemble_classify(
                comments,
                video_title = meta.get("title", "Unknown"),
            )
            st.write(f"✅ Ensemble classified {len(comments)} comments")
        except Exception as e:
            st.write(f"⚠️ Ensemble skipped: {e}")

    # Step 10 — topic modeling
    with st.status("🗂️ Running topic modeling (BERTopic)…"):
        topic_result = run_topic_modeling(texts)

    # Step 11 — ML
    with st.status("👥 Segmenting audience & scoring…"):
        comments = segment_audience(comments)
        comments = score_all_comments(comments)
        comments = predict_virality(comments)

    # Step 12 — build analysis dict
    ALL_SENTIMENTS = ["positive", "negative", "neutral"]
    ALL_EMOTIONS   = ["joy", "anger", "sadness", "surprise", "fear", "disgust", "neutral"]
    ALL_INTENTS    = ["praise", "complaint", "question", "suggestion", "general"]
    total    = len(comments) or 1
    approval = round(sum(1 for c in comments if c.get("sentiment") == "positive") / total * 100, 1)

    analysis = {
        "enriched_comments": comments,
        "approval_score":    approval,
        "toxic_count":       sum(1 for c in comments if c.get("is_toxic")),
        "sarcasm_count":     sum(1 for c in comments if c.get("is_sarcastic")),
        "sentiment_counts":  {s: sum(1 for c in comments if c.get("sentiment") == s)
                              for s in ALL_SENTIMENTS},
        "emotion_counts":    {e: sum(1 for c in comments if c.get("dominant_emotion") == e)
                              for e in ALL_EMOTIONS},
        "intent_counts":     {i: sum(1 for c in comments if c.get("intent") == i)
                              for i in ALL_INTENTS},
        "topics":            topic_result.get("topics", []),
        "segment_summary":   get_segment_summary(comments),
    }

    # Step 13 — Groq AI
    with st.status("🤖 Generating AI insights (Groq)…"):
        try:
            analysis["creator_summary"] = generate_creator_summary(analysis, meta)
        except Exception as e:
            analysis["creator_summary"] = f"⚠️ Groq error: {e}"
        try:
            analysis["content_gaps"] = find_content_gaps(analysis, meta)
        except Exception as e:
            analysis["content_gaps"] = f"⚠️ Groq error: {e}"

    # Step 14 — save to MongoDB
    try:
        save_video(meta)
        save_comments(video_id, comments)
        save_analysis(video_id, {"meta": meta, "analysis": analysis})
    except Exception as e:
        st.warning(f"⚠️ MongoDB save failed (results still shown): {e}")

    return meta, analysis


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🔍 Analyze Video")
st.markdown("<div style='color:var(--muted);margin-bottom:1.5rem'>Paste a YouTube URL and run the full NLP pipeline</div>",
            unsafe_allow_html=True)

# ── URL input ─────────────────────────────────────────────────────────────────
col_url, col_btn = st.columns([5, 1])
with col_url:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
                        label_visibility="collapsed")
with col_btn:
    run_btn = st.button("▶ Run Analysis", use_container_width=True)

st.divider()

# ── Trigger pipeline ──────────────────────────────────────────────────────────
if run_btn and url:
    try:
        meta, analysis = run_analysis(url, max_comments, use_cache)
        st.session_state["analysis"] = analysis
        st.session_state["meta"]     = meta
        st.success("✅ Analysis complete!")
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.stop()

elif run_btn and not url:
    st.warning("⚠️ Please enter a YouTube URL first.")

# ── Render results ────────────────────────────────────────────────────────────
if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    meta     = st.session_state["meta"]
    comments = analysis["enriched_comments"]
    df       = pd.DataFrame(comments)

    # ── Video header ──────────────────────────────────────────────────────────
    h1, h2 = st.columns([1, 3])
    with h1:
        if meta.get("thumbnail"):
            st.image(meta["thumbnail"], use_container_width=True)
    with h2:
        st.markdown(f"### {meta.get('title','Unknown Title')}")
        st.markdown(f"<span style='color:var(--muted)'>📺 {meta.get('channel','Unknown Channel')}</span>",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👁️ Views",    f"{meta.get('view_count', 0):,}")
        m2.metric("👍 Likes",    f"{meta.get('like_count', 0):,}")
        m3.metric("💬 Comments", f"{len(comments)}")
        m4.metric("✅ Approval", f"{analysis['approval_score']}%")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_sent, tab_emo, tab_topic, tab_audience, tab_comments, tab_ai = st.tabs([
        "📊 Sentiment", "😮 Emotions", "🗂️ Topics",
        "👥 Audience",  "💬 Comments", "🤖 AI Insights",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Sentiment
    # ══════════════════════════════════════════════════════════════════════════
    with tab_sent:
        sc    = analysis["sentiment_counts"]
        total = sum(sc.values()) or 1

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("✅ Positive", sc.get("positive",0), f"{sc.get('positive',0)/total*100:.1f}%")
        g2.metric("❌ Negative", sc.get("negative",0), f"{sc.get('negative',0)/total*100:.1f}%")
        g3.metric("➖ Neutral",  sc.get("neutral",0),  f"{sc.get('neutral',0)/total*100:.1f}%")
        g4.metric("☠️ Toxic",    analysis["toxic_count"],
                  f"{analysis['toxic_count']/total*100:.1f}%")

        c1, c2 = st.columns(2)

        with c1:
            approval = analysis["approval_score"]
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=approval,
                title={"text": "Approval Score", "font": {"size": 16}},
                gauge={
                    "axis":       {"range": [0, 100], "tickcolor": "#7a7f95"},
                    "bar":        {"color": "#6c63ff"},
                    "bgcolor":    "#1c2030",
                    "bordercolor":"#2a2f42",
                    "steps": [
                        {"range": [0,  40],  "color": "rgba(255,91,91,0.13)"},
                        {"range": [40, 65],  "color": "rgba(245,197,66,0.13)"},
                        {"range": [65, 100], "color": "rgba(31,217,154,0.13)"},
                    ],
                    "threshold": {"line": {"color": "#1fd99a", "width": 3}, "value": approval},
                },
                number={"suffix": "%", "font": {"size": 36, "color": "#e8eaf0"}},
            ))
            gauge.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(gauge, use_container_width=True)

        with c2:
            donut = go.Figure(go.Pie(
                labels=list(sc.keys()), values=list(sc.values()), hole=0.55,
                marker_colors=["#1fd99a", "#ff5b5b", "#7a7f95"],
                textfont_color="#e8eaf0",
            ))
            donut.update_layout(title="Sentiment Distribution", **PLOTLY_LAYOUT,
                                legend=dict(font=dict(color="#e8eaf0")))
            st.plotly_chart(donut, use_container_width=True)

        st.markdown("#### 🔬 Deeper Signals")
        d1, d2, d3 = st.columns(3)
        d1.metric("🙄 Sarcastic", analysis.get("sarcasm_count", 0),
                  f"{analysis.get('sarcasm_count',0)/total*100:.1f}%")
        d2.metric("☠️ Toxic",    analysis["toxic_count"],
                  f"{analysis['toxic_count']/total*100:.1f}%")
        d3.metric("🌐 Hinglish",
                  sum(1 for c in comments if c.get("language") in ("hinglish","hi")),
                  "multilingual comments")

        st.markdown("#### Sentiment × Intent")
        intent_sent = df.groupby(["intent", "sentiment"]).size().reset_index(name="count")
        if not intent_sent.empty:
            bar = px.bar(intent_sent, x="intent", y="count", color="sentiment",
                         color_discrete_map={"positive":"#1fd99a","negative":"#ff5b5b","neutral":"#7a7f95"},
                         barmode="group")
            bar.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Emotions
    # ══════════════════════════════════════════════════════════════════════════
    with tab_emo:
        ec = analysis["emotion_counts"]
        EMOTION_COLORS = {
            "joy":     "#1fd99a", "anger":   "#ff5b5b",
            "sadness": "#378ADD", "surprise":"#f5c542",
            "fear":    "#ff6584", "disgust": "#888780",
            "neutral": "#7a7f95",
        }
        cats = list(ec.keys())
        vals = list(ec.values())

        c1, c2 = st.columns(2)
        with c1:
            radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(108,99,255,.25)",
                line=dict(color="#6c63ff", width=2),
            ))
            radar.update_layout(title="Emotion Radar", **PLOTLY_LAYOUT,
                polar=dict(bgcolor="#1c2030",
                           radialaxis=dict(gridcolor="#2a2f42", color="#7a7f95"),
                           angularaxis=dict(gridcolor="#2a2f42", color="#e8eaf0")))
            st.plotly_chart(radar, use_container_width=True)

        with c2:
            emo_df = pd.DataFrame({"emotion": cats, "count": vals})
            bar2   = px.bar(emo_df, x="count", y="emotion", orientation="h",
                            color="emotion", color_discrete_map=EMOTION_COLORS)
            bar2.update_layout(title="Emotion Breakdown", showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(bar2, use_container_width=True)

        st.markdown("#### Top Comment per Emotion")
        for emo, color in EMOTION_COLORS.items():
            top = df[df["dominant_emotion"] == emo].sort_values("like_count", ascending=False)
            if len(top):
                row = top.iloc[0]
                st.markdown(f"""
                <div class='yt-card' style='border-left:3px solid {color}'>
                    <span style='color:{color};font-weight:600;font-size:.8rem'>{emo.upper()}</span>
                    <div style='margin-top:6px;font-size:.9rem'>{str(row['text'])[:200]}</div>
                    <div style='color:var(--muted);font-size:.75rem;margin-top:4px'>👍 {row['like_count']} likes</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Topics
    # ══════════════════════════════════════════════════════════════════════════
    with tab_topic:
        topics = analysis.get("topics", [])
        if topics:
            topic_df = pd.DataFrame(topics)
            c1, c2   = st.columns(2)
            with c1:
                treemap = px.treemap(topic_df, path=["label"], values="count",
                                     color="count",
                                     color_continuous_scale=["#1c2030","#6c63ff","#ff6584"])
                treemap.update_layout(title="Topic Treemap", **PLOTLY_LAYOUT)
                st.plotly_chart(treemap, use_container_width=True)
            with c2:
                bar3 = px.bar(topic_df.sort_values("count", ascending=True),
                              x="count", y="label", orientation="h",
                              color="count",
                              color_continuous_scale=["#6c63ff","#ff6584"])
                bar3.update_layout(title="Topic Sizes", showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(bar3, use_container_width=True)

            st.markdown("#### Topic Details")
            for t in topics:
                with st.expander(f"📌 {t['label']}  ·  {t['count']} comments"):
                    word_html = " ".join(f"<span class='tag'>{w}</span>" for w in t["words"])
                    st.markdown(word_html, unsafe_allow_html=True)
        else:
            st.info("Not enough comments for topic modeling (minimum 10 required).")

        st.markdown("#### 💬 Intent Distribution")
        ic = analysis["intent_counts"]
        intent_df = pd.DataFrame({"intent": list(ic.keys()), "count": list(ic.values())})
        intent_colors = {
            "praise":"#1fd99a", "complaint":"#ff5b5b",
            "question":"#378ADD", "suggestion":"#f5c542", "general":"#7a7f95",
        }
        bar4 = px.bar(intent_df, x="intent", y="count",
                      color="intent", color_discrete_map=intent_colors)
        bar4.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(bar4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Audience Segments
    # ══════════════════════════════════════════════════════════════════════════
    with tab_audience:
        seg_summary = analysis.get("segment_summary", {})
        SEG_ICONS   = {"Super Fans":"⭐","Critics":"🗣️","Curious Minds":"🧐","Casual Viewers":"😐"}

        s1, s2, s3, s4 = st.columns(4)
        for col, (name, info) in zip([s1, s2, s3, s4], seg_summary.items()):
            color = info["color"]
            col.markdown(f"""
            <div class='yt-card' style='border-top:3px solid {color};text-align:center'>
                <div style='font-size:1.8rem'>{SEG_ICONS.get(name,"👤")}</div>
                <div style='font-weight:700;font-size:1.1rem;margin:4px 0'>{name}</div>
                <div style='font-size:2rem;font-weight:800;color:{color}'>{info['count']}</div>
                <div style='font-size:.75rem;color:var(--muted)'>{info['description']}</div>
                <div style='margin-top:8px;font-size:.8rem'>Avg likes: <b>{info['avg_likes']}</b></div>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            seg_donut = go.Figure(go.Pie(
                labels=list(seg_summary.keys()),
                values=[v["count"] for v in seg_summary.values()],
                hole=0.55,
                marker_colors=[v["color"] for v in seg_summary.values()],
                textfont_color="#e8eaf0",
            ))
            seg_donut.update_layout(title="Audience Composition", **PLOTLY_LAYOUT,
                                    legend=dict(font=dict(color="#e8eaf0")))
            st.plotly_chart(seg_donut, use_container_width=True)

        with c2:
            avg_likes_df = pd.DataFrame({
                "segment":   list(seg_summary.keys()),
                "avg_likes": [v["avg_likes"] for v in seg_summary.values()],
            })
            bar5 = px.bar(avg_likes_df, x="segment", y="avg_likes",
                          color="segment",
                          color_discrete_map={k: v["color"] for k, v in seg_summary.items()})
            bar5.update_layout(title="Avg Likes per Segment", showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(bar5, use_container_width=True)

        st.markdown("#### Sample Comments by Segment")
        for name, info in seg_summary.items():
            color  = info["color"]
            seg_df = df[df["segment_label"] == name].sort_values("like_count", ascending=False)
            with st.expander(f"{SEG_ICONS.get(name,'👤')} {name}  ·  {info['count']} commenters"):
                for _, row in seg_df.head(5).iterrows():
                    badges = f"<span class='tag tag-{row['sentiment']}'>{row['sentiment']}</span>"
                    if row.get("is_sarcastic"):
                        badges += "<span class='tag tag-sarcasm'>sarcasm</span>"
                    if row.get("is_toxic"):
                        badges += "<span class='tag tag-negative'>toxic</span>"
                    st.markdown(f"""
                    <div style='padding:8px 0;border-bottom:1px solid var(--border)'>
                        <div style='font-size:.88rem;margin-bottom:4px'>{str(row['text'])[:200]}</div>
                        <div>{badges}
                            <span style='color:var(--muted);font-size:.75rem;margin-left:6px'>
                                👍 {row['like_count']}
                            </span>
                        </div>
                    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Comments Browser
    # ══════════════════════════════════════════════════════════════════════════
    with tab_comments:
        st.markdown("#### 💬 Comment Browser")

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_sent   = st.selectbox("Sentiment", ["All","positive","negative","neutral"])
        with fc2:
            f_intent = st.selectbox("Intent",    ["All","praise","complaint","question","suggestion","general"])
        with fc3:
            f_seg    = st.selectbox("Segment",   ["All"] + list(analysis.get("segment_summary",{}).keys()))
        with fc4:
            f_sort   = st.selectbox("Sort by",   ["like_count","quality_score","toxic_score","sarcasm_score"])

        fdf = df.copy()
        if f_sent   != "All": fdf = fdf[fdf["sentiment"]     == f_sent]
        if f_intent != "All": fdf = fdf[fdf["intent"]        == f_intent]
        if f_seg    != "All": fdf = fdf[fdf["segment_label"] == f_seg]

        if f_sort in fdf.columns:
            fdf = fdf.sort_values(f_sort, ascending=False)

        st.markdown(f"<div style='color:var(--muted);font-size:.82rem;margin-bottom:.8rem'>"
                    f"Showing {len(fdf)} comments</div>", unsafe_allow_html=True)

        for _, row in fdf.head(30).iterrows():
            emo_label = row.get("dominant_emotion", "")
            badges    = (f"<span class='tag tag-{row.get('sentiment','neutral')}'>"
                         f"{row.get('sentiment','neutral')}</span>"
                         f"<span class='tag'>{emo_label}</span>"
                         f"<span class='tag'>{row.get('intent','')}</span>")
            if row.get("is_sarcastic"):
                badges += "<span class='tag tag-sarcasm'>🙄 sarcasm</span>"
            if row.get("is_toxic"):
                badges += "<span class='tag tag-negative'>☠️ toxic</span>"
            lang = row.get("language", "en")
            if lang not in ("en", "unknown"):
                badges += f"<span class='tag'>{lang}</span>"
            seg_color = row.get("segment_color", "#888")
            quality   = row.get("quality_score", 0)
            st.markdown(f"""
            <div class='yt-card'>
                <div style='font-size:.9rem;line-height:1.5'>{str(row.get('text',''))[:300]}</div>
                <div style='display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap'>
                    {badges}
                    <span style='margin-left:auto;color:var(--muted);font-size:.75rem'>
                        👍 {row.get('like_count',0)} ·
                        quality: <span style='color:#6c63ff'>{quality:.2f}</span>
                    </span>
                    <span class='tag'
                          style='background:{seg_color}22;color:{seg_color};border-color:{seg_color}44'>
                        {row.get('segment_label','')}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        # CSV download — only include columns that exist
        export_cols = [c for c in
                       ["text","sentiment","dominant_emotion","intent","is_toxic",
                        "is_sarcastic","like_count","quality_score","segment_label","language"]
                       if c in fdf.columns]
        csv = fdf[export_cols].to_csv(index=False)
        st.download_button("⬇️ Download filtered comments as CSV", csv,
                           "comments.csv", "text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — AI Insights
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ai:
        ai1, ai2 = st.columns(2)

        with ai1:
            st.markdown("#### 🤖 Creator Report")
            st.markdown(f"""
            <div class='yt-card yt-card-accent'
                 style='white-space:pre-line;font-size:.88rem;line-height:1.7'>
{analysis.get('creator_summary','No summary generated.')}
            </div>""", unsafe_allow_html=True)

        with ai2:
            st.markdown("#### 💡 Content Gap Finder")
            st.markdown(f"""
            <div class='yt-card' style='border-left:3px solid var(--yellow);
                 white-space:pre-line;font-size:.88rem;line-height:1.7'>
{analysis.get('content_gaps','No gaps found.')}
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 Full Stats Snapshot")
        snap1, snap2, snap3 = st.columns(3)
        with snap1:
            st.markdown("**Sentiment**")
            for k, v in analysis["sentiment_counts"].items():
                st.markdown(f"<div style='display:flex;justify-content:space-between'>"
                            f"<span>{k}</span><b>{v}</b></div>", unsafe_allow_html=True)
        with snap2:
            st.markdown("**Top Emotions**")
            for k, v in sorted(analysis["emotion_counts"].items(), key=lambda x: x[1], reverse=True)[:5]:
                st.markdown(f"<div style='display:flex;justify-content:space-between'>"
                            f"<span>{k}</span><b>{v}</b></div>", unsafe_allow_html=True)
        with snap3:
            st.markdown("**Intents**")
            for k, v in analysis["intent_counts"].items():
                st.markdown(f"<div style='display:flex;justify-content:space-between'>"
                            f"<span>{k}</span><b>{v}</b></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 0;color:var(--muted)'>
        <div style='font-size:3rem'>🎬</div>
        <div style='font-size:1.1rem;margin-top:1rem'>
            Enter a YouTube URL above and hit <b>▶ Run Analysis</b>
        </div>
        <div style='font-size:.85rem;margin-top:.5rem'>
            The pipeline will fetch real comments and run the full NLP stack.
        </div>
    </div>""", unsafe_allow_html=True)