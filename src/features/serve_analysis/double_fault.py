"""Double fault detection — identify consecutive serve faults.

A double fault occurs when both the first and second serves are faults,
resulting in a point for the receiver.
"""

from dataclasses import dataclass

from src.features.serve_analysis.serve_detector import ServeEvent


@dataclass
class DoubleFault:
    """A detected double fault event."""
    fault_id: int
    first_serve: ServeEvent
    second_serve: ServeEvent
    server_end: str
    frame_range: tuple[int, int]  # (start of 1st serve, end of 2nd serve)

    @property
    def first_landing(self) -> tuple[float, float] | None:
        """Landing position of the first serve."""
        return self.first_serve.landing_position

    @property
    def second_landing(self) -> tuple[float, float] | None:
        """Landing position of the second serve."""
        return self.second_serve.landing_position


def detect_double_faults(serves: list[ServeEvent]) -> list[DoubleFault]:
    """Detect double faults from a sequence of serve events.

    A double fault is two consecutive faults where the second serve
    has serve_number == 2.

    Args:
        serves: List of serve events in chronological order.

    Returns:
        List of DoubleFault events.
    """
    double_faults: list[DoubleFault] = []
    fault_id = 1

    i = 0
    while i < len(serves) - 1:
        first = serves[i]
        second = serves[i + 1]

        if (
            first.is_fault
            and first.serve_number == 1
            and second.is_fault
            and second.serve_number == 2
        ):
            df = DoubleFault(
                fault_id=fault_id,
                first_serve=first,
                second_serve=second,
                server_end=first.server_end,
                frame_range=(first.start_frame, second.end_frame),
            )
            double_faults.append(df)
            fault_id += 1
            i += 2  # skip past both serves
        else:
            i += 1

    return double_faults
