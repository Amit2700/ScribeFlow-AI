from dotenv import load_dotenv
from services.audio_processor import process_input
from engine.transcriber import transcribe_all
from engine.summarizer import summarize, generate_title
from engine.extractor import extract_action_items, extract_key_decisions, extract_questions
from engine.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    print("\nStarting ScribeFlow AI Engine...")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language)
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    print("=" * 60)
    print(" 🎙️ ScribeFlow AI — Meeting Intelligence System")
    print("=" * 60)
    
    source = input("\nEnter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish) [default: english]: ").strip() or "english"
    
    if source:
        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print(f"📌 Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        # Interactive RAG Chat Mode
        print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 ScribeFlow Assistant: {answer}\n")
    else:
        print("❌ Error: No input source provided.")