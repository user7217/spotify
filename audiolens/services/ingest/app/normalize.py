"""Title/artist normalization + variant detection.

Used for dedup key generation and remaster/live/edit/deluxe/explicit grouping.
Pure stdlib — safe to run anywhere.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Variant markers, checked against parenthetical/suffix chunks of the title.
# Order matters: first match wins for `variant_type`; all matches kept as tags.
_VARIANT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("remaster", re.compile(r"\b(?:\d{4}\s*)?remaster(?:ed)?(?:\s*\d{4})?\b|\banniversary\b", re.I)),
    ("live", re.compile(r"\blive\b|\bunplugged\b|\bconcert\b|\bmtv\b", re.I)),
    ("radio_edit", re.compile(r"\bradio\s*(?:edit|mix|version)\b|\bsingle\s*(?:edit|version)\b|\bshort\s*(?:edit|version)\b", re.I)),
    ("extended", re.compile(r"\bextended\b|\bclub\s*mix\b|\b12[\"']?\s*(?:mix|version)\b", re.I)),
    ("remix", re.compile(r"\bremix\b|\bre-?work\b|\bflip\b|\bedit\b(?!\w)", re.I)),
    ("acoustic", re.compile(r"\bacoustic\b|\bstripped\b|\bpiano\s*version\b", re.I)),
    ("demo", re.compile(r"\bdemo\b|\brough\s*mix\b|\bouttake\b", re.I)),
    ("clean", re.compile(r"\bclean\b|\bcensored\b|\bedited\s*version\b", re.I)),
    ("explicit", re.compile(r"\bexplicit\b|\buncensored\b", re.I)),
    ("instrumental", re.compile(r"\binstrumental\b", re.I)),
    ("sped_up", re.compile(r"\bsped\s*up\b|\bslowed\b(?:\s*\+?\s*reverb)?|\bnightcore\b", re.I)),
    ("version", re.compile(r"\b(?:mono|stereo|album|original|deluxe|bonus(?:\s*track)?|tv|japanese|uk|us)\s*(?:version|mix|edition)?\b", re.I)),
]

_ALBUM_DELUXE = re.compile(
    r"\b(?:deluxe|expanded|special|anniversary|super|platinum|tour|collector'?s?)"
    r"\s*(?:edition|version)?\b|\(.*bonus.*\)",
    re.I,
)

# chunks: (...) [...] and " - suffix" tails
_PAREN = re.compile(r"[(\[]([^)\]]*)[)\]]")
_DASH_TAIL = re.compile(r"\s[-–—]\s(.+)$")
_FEAT = re.compile(r"\b(?:feat\.?|featuring|ft\.?|with|w/)\s+.*$", re.I)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def normalize(s: str) -> str:
    """Aggressive normalization for matching: casefold, deaccent, strip punctuation."""
    s = _ascii_fold(s).casefold()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


@dataclass
class TitleParse:
    base_title: str          # title with variant/feat chunks removed
    norm_title: str          # normalized base title
    variant_type: str        # primary variant ("original" if none)
    variant_tags: list[str] = field(default_factory=list)


def parse_title(title: str, album: str | None = None) -> TitleParse:
    """Split a raw title into base + variant markers.

    "One (2011 Remaster)"          -> base "One", remaster
    "Layla - Acoustic; Live"       -> base "Layla", acoustic + live
    "HUMBLE. - Clean Version"      -> base "HUMBLE.", clean
    Deluxe detection also looks at the album name.
    """
    tags: list[str] = []
    base = title

    chunks: list[str] = _PAREN.findall(title)
    m = _DASH_TAIL.search(title)
    if m:
        chunks.append(m.group(1))

    for chunk in chunks:
        chunk_tags = [name for name, pat in _VARIANT_PATTERNS if pat.search(chunk)]
        if chunk_tags:
            tags.extend(chunk_tags)
            base = base.replace(f"({chunk})", " ").replace(f"[{chunk}]", " ")
            if m and chunk == m.group(1):
                base = _DASH_TAIL.sub(" ", base)

    # variant words living in the bare title (no parens), e.g. "Song Live at Wembley"
    if not tags:
        for name, pat in _VARIANT_PATTERNS[:2]:  # only remaster/live are safe bare
            if pat.search(title):
                tags.append(name)

    if album and _ALBUM_DELUXE.search(album):
        tags.append("deluxe")

    base = _FEAT.sub("", base).strip(" -–—")
    primary = next((t for t in tags if t not in ("version", "deluxe")), None)
    if primary is None and "deluxe" in tags:
        primary = "deluxe"
    return TitleParse(
        base_title=base or title,
        norm_title=normalize(base or title),
        variant_type=primary or "original",
        variant_tags=sorted(set(tags)),
    )


def norm_key(artist: str, title: str, album: str | None = None) -> tuple[str, TitleParse]:
    """Dedup key: normalized primary artist + normalized base title."""
    tp = parse_title(title, album)
    primary_artist = re.split(r",|&|\bfeat\.?\b|\bft\.?\b", artist, flags=re.I)[0]
    return f"{normalize(primary_artist)}|{tp.norm_title}", tp
