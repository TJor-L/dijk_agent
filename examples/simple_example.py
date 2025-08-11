import os
from dijk_agent import pipelines
from dijk_agent.core.memory import ShortTermMemory

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    memory = ShortTermMemory(k=10)

    qa = pipelines.create(
        "tool-qa",
        model="gpt-4o-2024-08-06",
        temperature=0,
        memory_k=10,
        max_tool_hops=3,
        memory=memory
    )

    state = {"query": "Please calculate 12+34 for me, and then tell me the current UTC time."}
    out = qa.run(state)

    print("Q :", state["query"])
    print("A :", out["answer"])

    state = {"query": "Summarize the above conversation."}

    out = qa.run(state)
    print("Q :", state["query"])
    print("A :", out["answer"])

    direct_qa = pipelines.create(
        "direct-qa",
        model="gpt-4o-2024-08-06",
        temperature=0,
        memory_k=10,
        memory=memory
    )

    state = {"query": "Summarize the above conversation."}
    out = direct_qa.run(state)

    print("Q :", state["query"])
    print("A :", out["answer"])

if __name__ == "__main__":
    main()
