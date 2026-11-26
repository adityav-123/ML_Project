"""Command-line companion for the CareLens API.

The web application is served by ``uvicorn app:app``. This small utility is
kept for quick dataset/model checks without starting a browser.
"""

from app import get_engine, readable


if __name__ == "__main__":
    engine = get_engine()
    print("CareLens model check")
    print(f"Training records: {engine['records']}")
    print(f"Symptoms: {len(engine['symptoms'])}")
    print(f"Disease classes: {len(engine['diseases'])}")
    print(f"Accuracy: {engine['metrics']['accuracy']:.1%}")
    print("Top signals:")
    for feature in engine["importance"][:5]:
        print(f"  - {readable(feature['name'])}: {feature['value']:.1%}")
