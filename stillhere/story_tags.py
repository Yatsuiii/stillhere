"""The fixed content-note tags of docs/rules/story_menu_rule.md section 4, in the rule's order and spelling (shown as written). One list
for the notes call's json_schema, the code that keeps notes, and the page's story card. TAG_IDS gives each tag the identifier the
planner's enums carry (egress.TableEnums takes identifiers, not sentences)."""
import re

TAGS = ("death or dying", "injury or illness", "falls or head injury", "being trapped or unable to move or speak", "drowning",
        "suicide or self-harm", "grief or loss", "medical care", "violence", "cruelty to animals", "sexual content", "alcohol or drugs",
        "frightening scenes", "slurs or demeaning language")
TAG_IDS = {t: re.sub(r"[^a-z]+", "_", t).strip("_") for t in TAGS}
