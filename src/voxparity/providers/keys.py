"""Per-provider API key pools with failover.

``<NAME>_API_KEYS`` (comma-separated) takes precedence over ``<NAME>_API_KEY``.
On a quota/rate-limit response the provider calls ``rotate()`` and retries with
the next key; when every key is exhausted the original error propagates.
Rotation is process-local and sticky (the pool remembers which key last worked).
Keys are never logged; only their index is.

``<NAME>_PAID_KEY`` (Sep 6, ~$5 authorization): appended LAST so every free key
exhausts before a request bills, and ``release_paid()`` snaps the pool back to
the free keys after each paid success — stickiness would otherwise route every
subsequent call through the paid key even after the free quotas recover.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

EXHAUSTED_STATUSES = {402, 429}


@dataclass
class KeyPool:
    name: str
    keys: list[str] = field(default_factory=list)
    idx: int = 0
    paid_idx: int | None = None

    def __post_init__(self) -> None:
        if not self.keys:
            raw = os.environ.get(f"{self.name}_API_KEYS", "")
            self.keys = [k.strip() for k in raw.split(",") if k.strip()]
            single = os.environ.get(f"{self.name}_API_KEY", "").strip()
            if single and single not in self.keys:
                self.keys.insert(0, single)
        paid = os.environ.get(f"{self.name}_PAID_KEY", "").strip()
        if paid and paid not in self.keys:
            self.keys.append(paid)
            self.paid_idx = len(self.keys) - 1
        elif paid:
            self.paid_idx = self.keys.index(paid)
        if not self.keys:
            raise RuntimeError(f"{self.name}_API_KEY(S) not set (put it in .env)")

    @property
    def current(self) -> str:
        return self.keys[self.idx]

    @property
    def size(self) -> int:
        return len(self.keys)

    def rotate(self) -> bool:
        """Advance to the next key; False once every key has been tried this cycle."""
        if self.size == 1:
            return False
        self.idx = (self.idx + 1) % self.size
        return True

    @property
    def on_paid(self) -> bool:
        return self.paid_idx is not None and self.idx == self.paid_idx

    def release_paid(self) -> None:
        """After a successful billed call, point the NEXT request back at the
        free keys — the paid key fills rate-limit gaps, it never becomes the
        default route."""
        if self.on_paid:
            self.idx = 0
