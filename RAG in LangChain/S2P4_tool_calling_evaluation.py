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
from collections import defaultdict
import pandas as pd


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

# Create agent
tools = [search_jobs, compare_salaries, analyze_resume]
model = ChatOpenAI(model=MODEL)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a career consultant. Use your tools when you need "
        "real data. For general advice questions, respond directly "
        "without calling tools."
    ),
)
print("Agent created with tools:", [t.name for t in tools])

test_cases = [
    # ── Easy: clear intent, obvious tool ──
    {
        "query": "Average salary for data scientist in Berlin?",
        "expected_tool": "compare_salaries",
        "expected_params": {"role": "data scientist", "location": "Berlin"},
        "difficulty": "easy",
    },
    {
        "query": "Find me ML engineer jobs in London",
        "expected_tool": "search_jobs",
        "expected_params": {"role": "ML engineer", "location": "London"},
        "difficulty": "easy",
    },
    {
        "query": (
            "Review my resume? Here it is: "
            "Experienced data scientist with 5 years in ML. "
            "Skills: Python, TensorFlow, SQL. "
            "Previous role: Senior Analyst at DataCorp."
        ),
        "expected_tool": "analyze_resume",
        "expected_params": None,  # free-form text, skip exact param check
        "difficulty": "easy",
    },
    # ── Medium: indirect phrasing, misleading keywords ──
    {
        "query": (
            "Thinking about moving to Berlin, wondering what people "
            "like me make — I do data science"
        ),
        "expected_tool": "compare_salaries",
        "expected_params": None,
        "difficulty": "medium",
    },
    {
        "query": "Friend told me to fix my CV. 3 years Python/ML at Google.",
        "expected_tool": "analyze_resume",
        "expected_params": None,
        "difficulty": "medium",
    },
    {
        "query": "Can you search for how much a product manager earns?",
        "expected_tool": "compare_salaries",
        "expected_params": None,
        "difficulty": "medium",
    },
    {
        "query": "What skills should I learn for ML engineering?",
        "expected_tool": None,
        "expected_params": None,
        "difficulty": "medium",
    },
    # ── Hard: blended intents, multi-tool ambiguity ──
    {
        "query": "Need to know about ML engineer positions and their pay in Amsterdam",
        "expected_tool": "compare_salaries",
        "expected_params": None,
        "difficulty": "hard",
    },
    {
        "query": "Compare the job markets in Berlin and London for data scientists",
        "expected_tool": "search_jobs",
        "expected_params": None,
        "difficulty": "hard",
    },
    {
        "query": "Resume: Python, SQL, 2 years. What salary should I expect?",
        "expected_tool": "compare_salaries",
        "expected_params": None,
        "difficulty": "hard",
    },
    {
        "query": "Tell me everything about data engineering in Munich",
        "expected_tool": "search_jobs",
        "expected_params": None,
        "difficulty": "hard",
    },
    # ── Edge cases: no tool should be called ──
    {
        "query": "I hate my job. Help me.",
        "expected_tool": None,
        "expected_params": None,
        "difficulty": "edge_case",
    },
    {
        "query": "What's hot in the tech job market right now?",
        "expected_tool": None,
        "expected_params": None,
        "difficulty": "edge_case",
    },
    {
        "query": "Tell me a joke",
        "expected_tool": None,
        "expected_params": None,
        "difficulty": "edge_case",
    },
]

print(f"Total test cases: {len(test_cases)}")

by_difficulty = defaultdict(list)#creates an empty list
for case in test_cases:
    by_difficulty[case["difficulty"]].append(case)

for diff, cases in by_difficulty.items():
    print(f"  {diff}: {len(cases)} cases")

# Run agent on each test case and collect results
results = []

for i, case in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] {case['query'][:60]}...")

    response = agent.invoke(
    {"messages": [{"role": "user", "content": case["query"]}]}
    )

    # Extract tool calls from all messages in the response
    tool_calls = []
    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    actual_tool = tool_calls[0]["name"] if tool_calls else None
    actual_params = tool_calls[0]["args"] if tool_calls else None

    # Check tool selection
    tool_correct = actual_tool == case["expected_tool"]

    # Check parameters (only meaningful if tool selection was correct)
    if case["expected_tool"] is None:
        params_correct = tool_correct  # No tool expected = no params to check
    elif case["expected_params"] is None:
        params_correct = tool_correct  # Skip param check for this case
    elif tool_correct and actual_params is not None:
        params_correct = actual_params == case["expected_params"]
    else:
        params_correct = False

    results.append(
        {
            "query": case["query"],
            "expected_tool": case["expected_tool"],
            "actual_tool": actual_tool,
            "tool_correct": tool_correct,
            "expected_params": case["expected_params"],
            "actual_params": actual_params,
            "params_correct": params_correct,
            "difficulty": case["difficulty"],
            "response": response["messages"][-1].content,
        }
    )

print(f"\nCompleted {len(results)}/{len(test_cases)} test cases.")

tool_accuracy = sum(r["tool_correct"] for r in results) / len(results)
param_accuracy = sum(r["params_correct"] for r in results) / len(results)

print(f"Tool Selection Accuracy: {tool_accuracy:.0%}")
print(f"Parameter Accuracy:      {param_accuracy:.0%}")
print()