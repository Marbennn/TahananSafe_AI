"""
CLI tester for Incident Analyzer.
Lets you type an incident description and see the analysis.
"""

import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.analyzer import IncidentAnalyzer


def run_cli():
    """Interactive CLI for testing the incident analyzer."""
    print("Initializing Incident Analyzer...")
    print(f"Python executable: {sys.executable}")
    if ".venv" not in sys.executable.lower():
        print(
            "Warning: project virtual environment is not active. "
            "Use .\\.venv\\Scripts\\python.exe for consistent dependencies."
        )
    analyzer = IncidentAnalyzer()

    # Try to load model (will fall back to rule-based if not available)
    print("\nLoading model (will fall back to rule-based if fine-tuned model is missing)...")
    try:
        analyzer.load_model()
    except Exception as e:
        print(f"Warning: could not load fine-tuned model ({e}). Using rule-based analyzer only.")

    print("\n" + "=" * 80)
    print("TahananSafe AI - CLI Incident Tester")
    print("Type your incident description below.")
    print("Press Enter on an empty line or type 'q' to exit.")
    print("=" * 80)

    while True:
        text = input("\nIncident description:\n> ").strip()
        if not text or text.lower() in {"q", "quit", "exit"}:
            print("\nExiting tester.")
            break

        try:
            start = time.perf_counter()
            result = analyzer.analyze(text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except Exception as e:
            print(f"\nERROR while analyzing: {e}")
            continue

        print("\n--- Analysis ---")
        print(f"Incident Type     : {result.get('incident_type')}")
        print(f"Language          : {result.get('language')}")
        print(f"Risk Level        : {result.get('risk_level')}")
        print(f"Risk Percentage   : {result.get('risk_percentage')}%")
        print(f"Priority Level    : {result.get('priority_level')}")
        print(f"Children Involved : {'Yes' if result.get('children_involved') else 'No'}")
        print(f"Weapon Mentioned  : {'Yes' if result.get('weapon_mentioned') else 'No'}")
        print(f"Confidence Score  : {result.get('confidence_score')}%")
        print(f"Submission Status : {result.get('submission_decision')} ({'Allow' if result.get('allow_submission') else 'Blocked'})")
        print(f"Validation Reason : {result.get('validation_reason')}")
        print(f"Incident Tip      : {result.get('incident_tip')}")
        print(f"Response Time     : {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    run_cli()
