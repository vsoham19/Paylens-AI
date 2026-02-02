from llm.rag import RAGAssistant

rag = RAGAssistant("artifacts/metadata/run_metadata.json")

print("\n🔹 ML Assistant Ready")
print("Type your question and press Enter")
print("Type 'exit' to quit\n")

while True:
    question = input("🧠 Ask: ")

    if question.lower() == "exit":
        print("👋 Exiting assistant")
        break

    print("\n🤖 Answer:", rag.ask(question), "\n")
