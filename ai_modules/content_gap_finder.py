import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
    )


def find_content_gaps(analysis: dict, metadata: dict) -> str:
    llm = get_llm()

    questions   = [c["text"] for c in analysis.get("enriched_comments", [])
                   if c.get("intent") == "question"][:30]
    suggestions = [c["text"] for c in analysis.get("enriched_comments", [])
                   if c.get("intent") == "suggestion"][:20]

    prompt = PromptTemplate(
        input_variables=["title", "topics", "questions", "suggestions"],
        template="""
You are a YouTube content strategist.

Video: "{title}"
Topics audience discussed: {topics}

Questions asked in comments:
{questions}

Suggestions from the audience:
{suggestions}

Identify 5 specific content gap opportunities — topics this creator hasn't covered that their audience clearly wants.

Format each as:
- VIDEO IDEA: [specific title]
  WHY: [one sentence explaining the audience demand signal]

Be specific and creative. Base ideas only on the actual comments.
""",
    )
    chain  = prompt | llm
    result = chain.invoke({
        "title":       metadata.get("title", "Unknown"),
        "topics":      ", ".join([t["label"] for t in analysis.get("topics", [])[:8]]),
        "questions":   "\n".join(f"- {q}" for q in questions)   or "None found",
        "suggestions": "\n".join(f"- {s}" for s in suggestions) or "None found",
    })
    return result.content.strip()