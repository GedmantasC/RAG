import os
import json
import httpx
import toml
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


# Load API key
secrets = toml.load(".key/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

MODEL = "gpt-4o-mini"
client = OpenAI()

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say 'connection OK' in exactly two words."}],
)
print("API check:", response.choices[0].message.content)

# Define tools for the career consultant
@tool
def search_jobs(role: str, location: str) -> str:
    """Search for job openings matching a specific role and location."""
    return json.dumps(
        {
            "jobs": [
                {"title": f"Senior {role}", "company": "TechCorp", "location": location},
                {"title": role, "company": "DataInc", "location": location},
            ]
        }
    )

@tool
def compare_salaries(role: str, location: str) -> str:
    """Compare average salaries for a given role in a specific location."""
    return json.dumps(
        {
            "role": role,
            "location": location,
            "average_salary": 78500,
            "currency": "EUR",
            "range": {"min": 65000, "max": 92000},
        }
    )

@tool
def analyze_resume(resume_text: str) -> str:
    """Analyze a resume and provide improvement suggestions."""
    return json.dumps(
        {
            "score": 7,
            "suggestions": [
                "Add more quantifiable achievements",
                "Include relevant certifications",
            ],
        }
    )

# Create agent
tools = [search_jobs, compare_salaries, analyze_resume]
model = ChatOpenAI(model=MODEL)