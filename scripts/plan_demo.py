"""Demo: print the planner card for a menu prompt file and a backend.

Live calls happen only when a person runs this with keys set (MODEL_API_KEY for spark-1.3, FIREWORKS_API_KEY for
spark-open). Importing or running it without keys makes no model call.

usage: python3 scripts/plan_demo.py stillhere/public_text/q_plan_6yo.txt spark-1.3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stillhere import egress, planner


def main(argv: list) -> int:
    prompt_path, backend = argv[1], argv[2] if len(argv) > 2 else "spark-1.3"
    text = Path(prompt_path).read_text()
    try:
        question = egress.PublicText(text=text)
    except egress.EgressRefused:
        question = egress.PublicQuestion(text=text)
    client = planner.live_client()
    card = planner.plan(question, backend, client)
    if not card.declined:
        card = planner.build_card(card.spec, planner.run_tools(card.spec, backend, client), planner.load_enums())
    print(card.display())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
