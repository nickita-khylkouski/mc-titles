"""Strip a channel's repeated boilerplate (socials, merch, server IPs, music credits) out of its descriptions,
so what's left is the part that's about this one video. A line is boilerplate if it shows up in >=15% of the
channel's videos (and at least 3). Links, @handles-only and #hashtag-only lines go too."""
import re
from collections import Counter

URL = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|gg|tv|ly|io|me|co)(?:/\S*)?\b", re.I)


def norm(line):
    return re.sub(r"\W+", " ", URL.sub("", line)).strip().lower()


def cleaner(descs, share=0.15, max_chars=600):
    count = Counter(n for d in descs for n in {norm(l) for l in d.splitlines()} if n)
    cut = max(3, share * len(descs))
    def clean(d):
        keep = []
        for l in d.splitlines():
            n = norm(l)
            if not n or count[n] >= cut or re.fullmatch(r"[\s#@\w]*", l) and l.strip().startswith(("#", "@")):
                continue
            l = URL.sub("", l).strip()
            if not l.endswith(":"):  # "MERCH:" whose link was the content
                keep.append(l)
        return " ".join(keep)[:max_chars].strip()
    return clean


EPISODE = re.compile(r"(?i)\b(?:ep(?:isode)?|part|pt|day|s\d+\s*e|season\s*\d+\s*episode)\.?\s*#?\d+\b|#\d+\b|\[\s*\d+\s*\]|\(\s*\d+\s*\)|\be\d{1,3}\b")
WORD = re.compile(r"\w+(?:'\w+)?", re.U)


def title_cleaner(titles, share=0.2, min_n=3, max_n=8):
    """Strip a channel's title template: any 3-8 word phrase in >=20% of its titles (and at least 5), longest
    first, plus episode numbers. Single words are kept, so a channel that is all 'BTD6' still says BTD6."""
    toks = [[w.lower() for w in WORD.findall(EPISODE.sub(" ", t))] for t in titles]
    count = Counter()
    for ws in toks:
        count.update({tuple(ws[i:i + n]) for n in range(min_n, max_n + 1) for i in range(len(ws) - n + 1)})
    cut = max(5, share * len(titles))
    tmpl = sorted((p for p, c in count.items() if c >= cut), key=len, reverse=True)

    def clean(t):
        ws = [w.lower() for w in WORD.findall(EPISODE.sub(" ", t))]
        keep = [True] * len(ws)
        for p in tmpl:
            n = len(p)
            for i in range(len(ws) - n + 1):
                if tuple(ws[i:i + n]) == p:
                    keep[i:i + n] = [False] * n
        raw = WORD.findall(EPISODE.sub(" ", t))
        out = [w for w, k in zip(raw, keep) if k]
        return " ".join(out if len(out) >= 4 else raw) or t  # a one-format channel keeps its format words
    return clean
