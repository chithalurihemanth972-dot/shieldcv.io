"""Intelligence layer: cross-module correlation, immunity scoring, briefing generation.

`threat_story` correlates findings from independent modules into attack
narratives; `immunity` reduces a run to a single 0-100 Adversarial Immunity
Score across four weighted pillars; `briefing` renders a commander-readable
summary using a local Ollama model when present and a deterministic template
otherwise.
"""

from src.intelligence.threat_story import ThreatStoryEngine, build_threat_story
from src.intelligence.immunity import ImmunityScorer, compute_immunity_score
from src.intelligence.briefing import BriefingGenerator, generate_briefing, DISCLAIMER

__all__ = [
    "ThreatStoryEngine", "build_threat_story",
    "ImmunityScorer", "compute_immunity_score",
    "BriefingGenerator", "generate_briefing", "DISCLAIMER",
]
