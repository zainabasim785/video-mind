import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from rag_tool import rag_search_tool

load_dotenv()

LLM = os.environ.get("GOOGLE_LLM_MODEL", "gemini/gemini-2.5-flash")

retriever_agent = Agent(
    role="Video Content Retriever",
    goal="Search the YouTube video transcript and return the most relevant sections for the question",
    backstory=(
        "You are an expert at finding information inside video transcripts. "
        "You always use your search tool to find the most relevant parts of the video "
        "before passing them on. You never make things up."
    ),
    tools=[rag_search_tool],
    llm=LLM,
    verbose=True
)

writer_agent = Agent(
    role="Answer Writer",
    goal="Write a clear, accurate answer based only on the retrieved transcript sections",
    backstory=(
        "You are a concise communicator who turns raw transcript chunks into "
        "clean, easy-to-read answers. You only use facts from the provided sections "
        "and never add outside knowledge."
    ),
    llm=LLM,
    verbose=True
)

def run_crew(question: str) -> str:
    retrieve_task = Task(
        description=(
            f"Search the video transcript for information about: '{question}'\n"
            "Use the VideoRAGTool to find the top 4 most relevant sections.\n"
            "Return the sections exactly as found."
        ),
        expected_output="Top 4 relevant transcript sections related to the question.",
        agent=retriever_agent
    )

    write_task = Task(
        description=(
            f"Using ONLY the retrieved transcript sections, answer this question: '{question}'\n\n"
            "Rules:\n"
            "- Be concise and clear\n"
            "- Only use facts from the transcript sections\n"
            "- If the answer isn't in the transcript, say so"
        ),
        expected_output="A clear, concise answer based only on the video transcript.",
        agent=writer_agent
    )

    crew = Crew(
        agents=[retriever_agent, writer_agent],
        tasks=[retrieve_task, write_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)