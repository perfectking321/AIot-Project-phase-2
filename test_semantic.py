"""
Test semantic matching for UI elements.
"""

from brain.llm import get_llm_client
import json

def test_semantic_matching():
    print("=" * 60)
    print("SEMANTIC MATCHING TEST")
    print("=" * 60)

    llm = get_llm_client()
    print(f"Using LLM: {llm.__class__.__name__}")

    # Simulate screen elements (like File Explorer showing drives)
    elements = [
        {"id": 0, "label": "Home", "type": "text"},
        {"id": 1, "label": "Gallery", "type": "text"},
        {"id": 2, "label": "Devices and drives", "type": "text"},
        {"id": 3, "label": "Local Disk (C:)", "type": "text"},
        {"id": 4, "label": "244 GB free of 602 GB", "type": "text"},
        {"id": 5, "label": "Desktop", "type": "text"},
        {"id": 6, "label": "Downloads", "type": "text"},
        {"id": 7, "label": "Documents", "type": "text"},
        {"id": 8, "label": "Data (D:)", "type": "text"},
    ]

    # Test cases
    test_queries = [
        "C drive",
        "D drive",
        "local disc C",
        "go to downloads",
        "click desktop",
    ]

    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")

        element_list = []
        for elem in elements:
            element_list.append(f"[{elem['id']}] \"{elem['label']}\" (type: {elem['type']})")

        prompt = f"""You are a UI element matcher. Given a user's request and a list of visible screen elements, find the BEST matching element.

USER WANTS TO CLICK: "{query}"

VISIBLE ELEMENTS ON SCREEN:
{chr(10).join(element_list)}

MATCHING RULES:
1. Match by MEANING, not just exact text:
   - "C drive" = "Local Disk (C:)" or "C:" or "Local Disk C"
   - "D drive" = "Data (D:)" or any drive with "(D:)"
2. Prefer specific matches over generic ones

Respond with ONLY a JSON object:
{{"match": true/false, "element_id": <id>, "element_label": "<label>", "reason": "brief explanation"}}"""

        response = llm.generate(prompt)
        content = response.content.strip()

        # Parse JSON
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                if result.get("match"):
                    print(f"  MATCH: [{result['element_id']}] '{result['element_label']}'")
                    print(f"  Reason: {result['reason']}")
                else:
                    print(f"  NO MATCH: {result.get('reason')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Raw: {content[:200]}")


if __name__ == "__main__":
    test_semantic_matching()
