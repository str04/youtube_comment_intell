import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
    )


def generate_creator_summary(analysis: dict, metadata: dict) -> str:
    llm    = get_llm()
    prompt = PromptTemplate(
        input_variables=["title", "channel", "approval", "sentiments",
                         "emotions", "topics", "intents", "toxic_count", "total"],
        template="""
You are an expert YouTube audience analyst. Given comment analysis data, write a concise creator report.

Video: "{title}" by {channel}
Comments analyzed: {total} | Approval score: {approval}/100
Sentiment: {sentiments} | Emotions: {emotions}
Comment intents: {intents} | Topics: {topics} | Toxic comments: {toxic_count}

Write exactly 4 sections:
1. AUDIENCE VERDICT (2 sentences — did they like it overall?)
2. WHAT RESONATED (2-3 bullets — what the audience loved)
3. WHAT FELL FLAT (2-3 bullets — concerns or criticism)
4. YOUR NEXT VIDEO (2-3 concrete content ideas based on what the audience is asking for)

Be specific and actionable. Plain language only.
""",
    )
    chain = prompt | llm
    result = chain.invoke({
        "title":       metadata.get("title", "Unknown"),
        "channel":     metadata.get("channel", "Unknown"),
        "approval":    analysis.get("approval_score", 0),
        "sentiments":  str(analysis.get("sentiment_counts", {})),
        "emotions":    str(analysis.get("emotion_counts", {})),
        "topics":      ", ".join([t["label"] for t in analysis.get("topics", [])[:6]]),
        "intents":     str(analysis.get("intent_counts", {})),
        "toxic_count": analysis.get("toxic_count", 0),
        "total":       len(analysis.get("enriched_comments", [])),
    })
    return result.content.strip()


def generate_comparison_summary(an1: dict, an2: dict, meta1: dict, meta2: dict) -> str:
    llm    = get_llm()
    prompt = PromptTemplate(
        input_variables=["t1", "t2", "a1", "a2", "s1", "s2", "e1", "e2", "tx1", "tx2"],
        template="""
Compare two YouTube videos based on their comment analysis.

Video 1: "{t1}" — Approval: {a1}/100 | Sentiments: {s1} | Emotions: {e1} | Toxic: {tx1}
Video 2: "{t2}" — Approval: {a2}/100 | Sentiments: {s2} | Emotions: {e2} | Toxic: {tx2}

Write a 5-6 sentence comparison covering:
1. Which video performed better with its audience and why
2. The key emotional difference between the two audiences
3. One specific recommendation for each video
""",
    )
    chain  = prompt | llm
    result = chain.invoke({
        "t1":  meta1.get("title", "V1"),
        "t2":  meta2.get("title", "V2"),
        "a1":  an1.get("approval_score", 0),
        "a2":  an2.get("approval_score", 0),
        "s1":  str(an1.get("sentiment_counts", {})),
        "s2":  str(an2.get("sentiment_counts", {})),
        "e1":  str(an1.get("emotion_counts", {})),
        "e2":  str(an2.get("emotion_counts", {})),
        "tx1": an1.get("toxic_count", 0),
        "tx2": an2.get("toxic_count", 0),
    })
    return result.content.strip()