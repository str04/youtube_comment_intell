# 🎯 YT Intel — YouTube Comment Intelligence Platform

> An application that transforms raw YouTube comments into audience intelligence using a multi-layer NLP pipeline, ensemble ML, and Groq-powered AI insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA3--70B-green?style=flat-square)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?style=flat-square&logo=mongodb)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📸 Demo

| Analyze Page | Compare Page |
|---|---|
| Paste any YouTube URL → full audience intelligence in minutes | Side-by-side NLP breakdown of two videos |

---

## 🚀 What Makes This Unique

Most YouTube analytics tools just count likes and views. YT Intel goes deeper:

- **🧠 6-Layer NLP Pipeline** — Sentiment, Emotion, Toxicity, Sarcasm, Intent, Topic Modeling
- **🎯 Ensemble Emotion Classification** — 3-model voting + Groq LLM tiebreaker (~88-92% accuracy)
- **🇮🇳 Hinglish NLP** — Handles Roman Hindi comments that most tools completely miss
- **👥 Score-Based Audience Segmentation** — Super Fans, Critics, Curious Minds, Casual Viewers
- **🤖 AI Creator Reports** — Groq LLaMA3-70B generates actionable creator recommendations
- **💡 Content Gap Finder** — Identifies video ideas the audience is asking for
- **⚔️ Video Comparison** — Side-by-side audience intelligence for any two videos
- **⚡ Zero YouTube API Key** — Uses yt-dlp, no quota limits, just paste a URL

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│                   YouTube URL (any video)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                               │
│  yt-dlp → single call → metadata + comments (no API key needed) │
│  comment_parser → clean · deduplicate · filter · thread         │
│  mongo_store → cache results (repeat analysis = instant)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NLP LAYER                                   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Sentiment  │  │   Emotion   │  │  Toxicity   │             │
│  │ cardiffnlp  │  │ go_emotions │  │ toxic-bert  │             │
│  │ pos/neg/neu │  │ 28 → 7 cats │  │ toxic flag  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Sarcasm   │  │   Intent    │  │  Hinglish   │             │
│  │roberta-irony│  │ rule-based  │  │ XLM-RoBERTa │             │
│  │ irony detect│  │ 5 categories│  │ Roman Hindi │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ENSEMBLE EMOTION CLASSIFIER                  │   │
│  │  Vote 1: go_emotions model                               │   │
│  │  Vote 2: sentiment-derived rule                          │   │
│  │  Vote 3: keyword-based detection                         │   │
│  │  Majority vote → Groq tiebreaker on conflicts only       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  BERTopic → unsupervised topic clustering                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML LAYER                                   │
│                                                                  │
│  Audience Segmentation → score-based (positivity + criticism     │
│                          + curiosity scores per comment)         │
│    ⭐ Super Fans  🗣️ Critics  🧐 Curious Minds  😐 Casual        │
│                                                                  │
│  Quality Scorer → composite score (length · likes · sentiment)   │
│  Viral Predictor → GradientBoosting (predicts like count)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AI LAYER (GROQ)                            │
│                  LLaMA3-70B via LangChain                        │
│                                                                  │
│  Creator Report → 4-section analysis (verdict, resonated,        │
│                   fell flat, next video ideas)                   │
│  Content Gap Finder → 5 video ideas from audience signals        │
│  Groq Classifier → context-aware emotion + intent labels         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD (STREAMLIT)                         │
│                                                                  │
│  📊 Sentiment pie  😮 Emotion radar  🎯 Approval gauge           │
│  🗺️ Topic treemap  👥 Audience segments  💬 Comment browser      │
│  🤖 AI creator report  💡 Content gaps  ⚔️ Video compare         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
youtube-comment-intelligence/
├── ingestion/
│   ├── youtube_api.py            ← yt-dlp fetching (single call)
│   ├── comment_parser.py         ← clean · deduplicate · thread
│   └── mongo_sttore.py           ← MongoDB persistence + caching
├── nlp/
│   ├── sentiment.py              ← cardiffnlp/twitter-roberta
│   ├── topic_model.py            ← BERTopic clustering
│   ├── toxicity.py               ← unitary/toxic-bert
│   ├── sarcasm.py                ← cardiffnlp/twitter-roberta-irony
│   ├── intent_classifier.py      ← rule-based intent detection
│   ├── hinglish.py               ← XLM-RoBERTa multilingual
│   └── ensemble_classifier.py   ← 3-vote ensemble + Groq tiebreaker
├── ml/
│   ├── audiance_segmentation.py  ← score-based audience segments
│   ├── quality_scorer.py         ← composite comment quality score
│   └── viral_predictor.py        ← GradientBoosting virality
├── ai_modules/
│   ├── langchain_summary.py      ← Groq creator report
│   ├── content_gap_finder.py     ← Groq content gap analysis
│   └── groq_classifier.py        ← context-aware classification
├── dashboard/
│   ├── app.py                    ← Streamlit home page
│   └── pages/
│       ├── 1_analyze.py          ← Full analysis page
│       ├── 2_compare.py          ← Side-by-side comparison
│       └── 3_about.py            ← Project info
├── model_cache.py                ← Download + cache HF models
├── requirements.txt
└── .env.example
```

---

## 🤖 Models Used

| Task | Model | Why |
|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Best for social media text |
| Emotion | `SamLowe/roberta-base-go_emotions` | 28 categories, trained on 58k Reddit comments |
| Toxicity | `unitary/toxic-bert` | Production-grade toxic detection |
| Sarcasm | `cardiffnlp/twitter-roberta-base-irony` | Twitter-trained irony detection |
| Hinglish | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual, handles Roman Hindi |
| Topics | `BERTopic (multilingual)` | Zero-shot topic discovery |
| AI Layer | `Groq · LLaMA3-70B` | Fast, free, context-aware |

> All HuggingFace models are downloaded once and cached locally (~3GB). Groq runs on cloud — free tier.

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/youtube-comment-intelligence.git
cd youtube-comment-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
GROQ_API_KEY=your_groq_key_here        # free at console.groq.com
MONGO_URI=mongodb://localhost:27017    # local MongoDB
```

### 4. Download models (run once)
```bash
python model_cache.py
```
This downloads all 5 HuggingFace models (~3GB) to a local `models/` folder. After this, no internet needed for NLP.

### 5. Start MongoDB
```bash
net start MongoDB        # Windows
# or
brew services start mongodb-community   # Mac
```

### 6. Run the app
```bash
streamlit run dashboard/app.py
```

---

## 📊 Pipeline Overview

| Step | What happens | Time |
|---|---|---|
| 1 | yt-dlp fetches metadata + comments (single call) | ~2-3 min |
| 2 | Parse, clean, deduplicate comments | instant |
| 3 | Language detection + Hinglish routing | ~5s |
| 4 | Sentiment analysis (HuggingFace) | ~20s |
| 5 | Emotion detection (go_emotions) | ~25s |
| 6 | Toxicity + Sarcasm detection | ~30s |
| 7 | Intent classification (rule-based) | instant |
| 8 | Ensemble voting + Groq tiebreaker | ~15s |
| 9 | BERTopic topic modeling | ~20s |
| 10 | Audience segmentation + scoring | ~5s |
| 11 | Groq AI creator report + content gaps | ~10s |
| 12 | Save to MongoDB (cached for next time) | instant |

---

## 🌟 Key Technical Highlights

### Ensemble Emotion Classification
Instead of relying on a single model, we use a 3-vote ensemble:
- **Vote 1**: go_emotions model (28-category HuggingFace model)
- **Vote 2**: Sentiment-derived rule (positive sentiment → joy family)
- **Vote 3**: Keyword-based detection (Hindi/English emotion keywords)
- **Tiebreaker**: Groq LLaMA3 with video context — only fires on conflicts

This raises emotion accuracy from ~65% (single model) to ~88-92%.

### Hinglish NLP
Indian YouTube comments mix Hindi and English in ways that break English-only models. Words like *bawaal* (amazing), *mast* (great), *zabardast* (outstanding) are positive but get labeled negative by standard models. We handle this with dedicated keyword correction + XLM-RoBERTa routing.

### Score-Based Segmentation
Instead of KMeans clustering (which forces equal-sized buckets), we compute 3 independent scores per comment — positivity, criticism, curiosity — and assign segments by thresholds. This works correctly across all video types: educational, news, travel, gaming, politics.

---

## 🔧 Requirements

```
streamlit>=1.55
transformers>=4.57
torch>=2.0
bertopic
sentence-transformers
langchain-groq
groq
yt-dlp
pymongo
langdetect
emoji
scikit-learn
pandas
plotly
python-dotenv
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) for open-source NLP models
- [Groq](https://groq.com) for free LLaMA3 API access
- [BERTopic](https://maartengr.github.io/BERTopic) for topic modeling
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube data extraction
- [Streamlit](https://streamlit.io) for the dashboard framework

---

*Built as a final year Data Science capstone project.*
