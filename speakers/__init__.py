"""Speaker identification.

Contract shared by every implementation (pyannote now, a fake/ECAPA one later):

    identify(segment) -> "Me" | "Speaker 1" | "Speaker 2" | ...

The microphone is always "Me". Only system-audio segments are clustered.
"""

from typing import Protocol

from core.types import Segment

ME = "Me"


class SpeakerIdentifier(Protocol):
    def identify(self, segment: Segment) -> str: ...
