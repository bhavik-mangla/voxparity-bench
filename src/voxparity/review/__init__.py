"""Local review console (D088): the website that captures Bhavik's ear.

`voxparity review serve` builds a queue of every clip needing a human ruling —
cue-judge failures first (the judge over-rejects, D040/D042), then unmeasured
clips, plus capability-probe WAVs and open design flags — and serves a
single-page console on localhost. Each listen-verdict lands in a game-shaped
export JSON and, on finish, flows through the SAME `stimuli human-gate` pooling
path the game uses (one codepath, D055 accumulation, Rakov thresholds), so a
human pass here immediately overrides the judge and unblocks the clip for
scored runs. Design-flag answers append to docs/REVIEW-DECISIONS.md.
"""
