"""
TTME Memory Example: Demonstrating the three-layer memory mechanism
for GUI agent inference.

This example simulates a GUI agent performing tasks on a smartphone,
showing how the three memory layers work together:

1. Episodic Memory: Tracks recent actions within a sliding window
2. Semantic Memory: Stores universal interaction rules
3. Experiential Memory: Recalls strategies from similar past tasks

Usage:
    python -m inference.memory_example
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.memory import TTMEMemory


def main():
    memory = TTMEMemory(
        episodic_horizon=5,
        semantic_top_k=2,
        experiential_top_k=2,
        experiential_lambda=0.7,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "your-api-key"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        storage_path="./ttme_memory_storage.json",
    )

    print("=" * 70)
    print("TTME Memory Example: GUI Agent with Three-Layer Memory")
    print("=" * 70)

    # ================================================================
    # 1. Semantic Memory: Pre-load universal interaction rules
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Semantic Memory (Universal Interaction Rules)")
    print("=" * 70)

    semantic_rules = [
        "Always log in before attempting to access restricted pages or personal data.",
        "When an app crashes or freezes, force stop it and reopen it from settings.",
        "Use the back button to return to the previous screen when navigation fails.",
        "Before deleting files, always verify the correct file is selected.",
        "When filling forms, complete required fields before optional ones.",
        "If a search returns no results, try alternative keywords or check spelling.",
    ]

    for rule in semantic_rules:
        memory.add_semantic_rule(rule)
        print(f"  Added rule: {rule[:60]}...")

    print(f"\n  Semantic Memory: {len(memory.semantic)} rules loaded")

    # ================================================================
    # 2. Experiential Memory: Pre-load past task experiences
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 2: Loading Experiential Memory (Past Task Strategies)")
    print("=" * 70)

    experiences = [
        {
            "trajectory": "Open Settings > Navigate to Wi-Fi > Select network > Enter password > Connect",
            "summary": "To connect to Wi-Fi: open Settings, find Wi-Fi section, select the target network, "
                       "enter password carefully, and wait for connection confirmation. If connection fails, "
                       "try forgetting the network first and reconnecting.",
        },
        {
            "trajectory": "Open Camera > Switch to video mode > Press record > Record content > Press stop > Save",
            "summary": "To record a video: open Camera app, switch to video mode using the toggle, "
                       "press the record button, capture the content, press stop, and the video auto-saves "
                       "to gallery. Ensure sufficient storage before recording.",
        },
        {
            "trajectory": "Open Messages > Compose new message > Select contact > Type message > Send",
            "summary": "To send a message: open Messages app, tap compose, select or search for the "
                       "contact, type the message in the text field, and press send. For group messages, "
                       "add multiple recipients before typing.",
        },
        {
            "trajectory": "Open Chrome > Type URL > Navigate webpage > Find information > Bookmark",
            "summary": "To browse and find information: open Chrome, type the URL or search query in the "
                       "address bar, navigate the page by scrolling, use find-on-page for specific text, "
                       "and bookmark useful pages for later reference.",
        },
    ]

    for exp in experiences:
        memory.add_experience(
            trajectory=exp["trajectory"],
            reflective_summary=exp["summary"],
        )
        print(f"  Added experience: {exp['trajectory'][:50]}...")

    print(f"\n  Experiential Memory: {len(memory.experiential)} experiences loaded")

    # ================================================================
    # 3. Simulate a task: "Send a message to John"
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 3: Simulating Task - 'Send a message to John'")
    print("=" * 70)

    task_query = "Send a message to John"

    # Step-by-step episodic recording
    steps = [
        ("Home screen with app icons", "Open Messages app", "Messages app main screen"),
        ("Messages app main screen", "Tap compose new message", "New message composition screen"),
        ("New message composition screen", "Search and select contact 'John'", "Chat with John opened"),
        ("Chat with John opened", "Type 'Hello John, how are you?'", "Message typed in text field"),
        ("Message typed in text field", "Press send button", "Message sent successfully"),
    ]

    for obs, action, next_obs in steps:
        memory.record_step(obs, action, next_obs)
        print(f"  Recorded: {action}")

    print(f"\n  Episodic Memory: {len(memory.episodic)} entries (horizon={memory.episodic.horizon})")

    # ================================================================
    # 4. Retrieve all three memory contexts
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 4: Retrieving Memory Contexts")
    print("=" * 70)

    contexts = memory.retrieve_context(query=task_query)

    print("\n--- Episodic Context (C_epi) ---")
    print(contexts["episodic_context"] if contexts["episodic_context"] else "(empty)")

    print("\n--- Semantic Context (C_sem) ---")
    print(contexts["semantic_context"] if contexts["semantic_context"] else "(empty)")

    print("\n--- Experiential Context (C_exp) ---")
    print(contexts["experiential_context"] if contexts["experiential_context"] else "(empty)")

    # ================================================================
    # 5. Build integrated M_retrieved
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 5: Integrated Retrieved Memory (M_retrieved)")
    print("=" * 70)

    m_retrieved = memory.build_retrieved_memory(query=task_query)
    print(m_retrieved if m_retrieved else "(no memory retrieved)")

    # ================================================================
    # 6. Simulate a second task to show episodic sliding window
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 6: New Task - 'Connect to Wi-Fi' (Episodic Window Resets)")
    print("=" * 70)

    memory.clear_episodic()

    wifi_steps = [
        ("Home screen", "Open Settings app", "Settings main menu"),
        ("Settings main menu", "Tap Wi-Fi", "Wi-Fi settings page"),
        ("Wi-Fi settings page", "Select 'HomeNetwork'", "Password prompt"),
        ("Password prompt", "Enter password 'mywifi123'", "Connecting..."),
        ("Connecting...", "Wait for connection", "Connected to HomeNetwork"),
        ("Connected screen", "Navigate back to home", "Home screen"),
    ]

    for obs, action, next_obs in wifi_steps:
        memory.record_step(obs, action, next_obs)

    print(f"  Recorded {len(wifi_steps)} steps")
    print(f"  Episodic Memory: {len(memory.episodic)} entries")

    wifi_contexts = memory.retrieve_context(query="Connect to Wi-Fi network")

    print("\n--- Episodic Context (sliding window, H=5) ---")
    print(wifi_contexts["episodic_context"])

    print("\n--- Semantic Context ---")
    print(wifi_contexts["semantic_context"])

    print("\n--- Experiential Context ---")
    print(wifi_contexts["experiential_context"])

    # ================================================================
    # 7. Save and reload memory
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 7: Save & Reload Memory")
    print("=" * 70)

    memory.save()
    print(f"  Memory saved to: {memory.storage_path}")
    print(f"  Memory state: {memory}")

    memory2 = TTMEMemory(
        episodic_horizon=5,
        semantic_top_k=2,
        experiential_top_k=2,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "your-api-key"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        storage_path="./ttme_memory_storage.json",
    )
    memory2.load()
    print(f"  Reloaded memory: {memory2}")

    reloaded_context = memory2.retrieve_context(query="Send a message")
    print("\n--- Reloaded Experiential Context ---")
    print(reloaded_context["experiential_context"])

    # Cleanup
    if os.path.exists("./ttme_memory_storage.json"):
        os.remove("./ttme_memory_storage.json")
        print("\n  Cleaned up temporary storage file")

    print("\n" + "=" * 70)
    print("TTME Memory Example Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
