import os
from dijk_agent import pipelines

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❗请先设置 OPENAI_API_KEY 环境变量。")
        return

    qa = pipelines.create(
        "simple-qa",
        model="gpt-4o-2024-08-06",
        temperature=0,
        memory_k=10,
        max_tool_hops=3,
    )

    state = {"query": "帮我先算 12+34，然后告诉我当前UTC时间。"}
    out = qa.run(state)

    print("Q :", state["query"])
    print("A :", out["answer"])

if __name__ == "__main__":
    main()
