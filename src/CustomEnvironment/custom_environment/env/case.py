import pandas as pd


class Case:
    """Represent a single case, i.e. trace, in the simulation."""

    def __init__(self, case_id: str, start_timestamp: pd.Timestamp) -> None:
        self.case_id = case_id  # Unique identifier for the case
        self.is_done: bool = False  # Whether the case has been completed
        # List of activities already performed in the case
        self.activities_performed: list[int] = []
        # Timestamp when the case started
        self.start_timestamp: pd.Timestamp = start_timestamp
        # Current timestamp of the case
        self.current_timestamp: pd.Timestamp = start_timestamp
        # List of additional activities that can be performed next
        self.additional_next_activities: list[int] = []
        self.potential_additional_agents: list[int] = []
        self.previous_agent = -1  # ID of the previous agent that performed an activity

    def get_last_activity(self) -> int | None:
        """Return the last activity performed in the case."""
        if len(self.activities_performed) == 0:
            return None
        else:
            return self.activities_performed[-1]

    def add_activity(self, activity: int) -> None:
        """Add an activity to the trace of the case."""
        self.activities_performed.append(activity)
