"""First live call (Oct 27): auth, endpoint and strict json_schema through the egress wrapper. Synthetic tier-D flags only (no results)."""
import json
from pathlib import Path
from stillhere import egress, meta_client

PT = Path(egress.__file__).with_name("public_text")
schema = {"type": "object", "additionalProperties": False, "required": ["line_id", "decline"],
          "properties": {"line_id": {"type": "string", "enum": ["fade_line", "fallback_line"]}, "decline": {"type": "boolean"}}}
req = egress.Request(model="muse-spark-1.3", system=[egress.PublicText(text=(PT / "desk_system.txt").read_text())],
                     user=[egress.TierD(values={"synthetic_test": True, "bar_p7_pass": False, "bar_p1_pass": False}, enums={"question": "can_we_say_language_fades"})],
                     schema_name="desk_smoke", json_schema=schema)
print(json.dumps(meta_client.chat(req)))
