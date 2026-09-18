"""Multi-agent office: one scanning agent per contributor, plus a meeting room.

`agent` scans a single contributor in isolation so that no contributor can bias
the population statistics another is scored against; `manager` discovers
contributor folders and runs the agents in parallel; `meeting` compares the
resulting reports for disparity, class targeting, trust inversion and model
disagreement, then issues the office verdict.
"""

from src.agents.agent import AgentReport, agent_scan_contributor, agent_scan_dict
from src.agents.manager import OfficeManager, run_office
from src.agents.meeting import MeetingRoom, hold_meeting

__all__ = [
    "AgentReport", "agent_scan_contributor", "agent_scan_dict",
    "OfficeManager", "run_office",
    "MeetingRoom", "hold_meeting",
]
