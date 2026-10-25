"""Throw ~1,000+ descriptive checks at the title map and rank what's surprising -> data/scan.json + a printed report.
Not "what gets views": only how things changed, when they peak, how seasonal they are.

  words      every word/phrase used 300+ times by 20+ channels: share per year, peakiness, rise/fall, comebacks, month seasonality
  patterns   ~200 hand regexes (mobs, ores, crossovers, pronouns, punctuation, emoji, numbers, title shapes)
  numbers    the biggest number in each title, per year
  versions   which "1.X" a year's titles talk about
  lengths    exact-second spikes in video length (ad-break rules, Shorts limits)
  days       busiest upload days ever, weekday/hour shifts
  emoji      top emoji per year
  nice uv run python pipeline/scan.py
"""
import glob, json, os, re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, W = os.path.join(ROOT, "data"), os.path.join(ROOT, "web", "data")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
YEARS = list(range(2011, 2027))

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=PT)
N = len(r)
titles = open(os.path.join(W, "titles.txt"), newline="").read().split("\n")[:N]
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:N]
chname = [c["name"] for c in meta["channels"]]
year = (r["m"] // 12 + 2005).astype(int)
moy = (r["m"] % 12).astype(int)
keep = (r["game"] == 0) & (year >= 2011) & (year <= 2026)
idx = np.flatnonzero(keep)
ytot = {y: int((keep & (year == y)).sum()) for y in YEARS}
out = {"years": YEARS}

date = {}
for p in glob.glob(os.path.join(D, "videos", "*.jsonl")):
    for l in open(p):
        i, j = l.find('"date": "'), l.find('"id": "')
        date[l[j + 7:l.index('"', j + 7)]] = l[i + 9:i + 29]
dur = {}
for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
    for l in open(p):
        d = json.loads(l)
        dur[d["id"]] = d.get("dur") or 0
PST = timezone(timedelta(hours=-8))
when = {}
for i in idx:
    d = date.get(ids[i])
    if d:
        when[i] = datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(PST)


def series(hit):
    """share of each year's titles that hit"""
    return [round(float((hit & keep & (year == y)).sum() / ytot[y]), 5) for y in YEARS]


def shape(s, n):
    s = np.array(s)
    pk = int(np.argmax(s))
    med = float(np.median(s[s > 0])) if (s > 0).any() else 0
    first, last = s[:3].mean(), s[-3:].mean()
    # comeback: a trough under 40% of the peaks on both sides
    cb = 0.0
    for t in range(1, len(s) - 1):
        a, b = s[:t].max(), s[t + 1:].max()
        if s[t] < 0.4 * min(a, b):
            cb = max(cb, min(a, b) / max(s[t], 1e-6) if min(a, b) > 0.002 else 0)
    return {"peak": YEARS[pk], "peak_share": round(float(s[pk]), 4), "peaky": round(float(s[pk] / max(med, 1e-6)), 1),
            "rise": round(float(np.log2((last + 1e-4) / (first + 1e-4))), 2), "comeback": round(float(min(cb, 99)), 1), "n": int(n)}


def season(hit):
    """month-of-year concentration across all years (each year weighted the same)"""
    mm = np.zeros(12)
    for y in YEARS:
        s = keep & (year == y)
        tot = np.bincount(moy[s], minlength=12)
        h = np.bincount(moy[s & hit], minlength=12)
        if h.sum() >= 5:
            mm += h / np.maximum(tot, 1) / max(h.sum() / s.sum(), 1e-9) / len(YEARS)
    mm = mm / max(mm.mean(), 1e-9)
    return int(np.argmax(mm)), round(float(mm.max()), 2), [round(float(v), 2) for v in mm]


# ---- words
tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())
STOP = set("the a an of in on to and is for with it this at be by from or are so but not all what minecraft mc shorts short video "
           "ep part episode s1 s2 s3 s4 lets let's".split())
grams = [None] * N
cnt, chs = Counter(), defaultdict(set)
for i in idx:
    w = tok(titles[i])
    g = {x for x in w if x not in STOP and len(x) > 1} | {f"{a} {b}" for a, b in zip(w, w[1:]) if a not in STOP and b not in STOP}
    grams[i] = g
    for x in g:
        cnt[x] += 1
        if len(chs[x]) < 20:
            chs[x].add(int(r["ch"][i]))
vocab = [g for g, c in cnt.items() if c >= 300 and len(chs[g]) >= 20]
print(len(vocab), "words/phrases")
vi = {g: k for k, g in enumerate(vocab)}
Yc = np.zeros((len(vocab), len(YEARS)))
Mc = np.zeros((len(vocab), len(YEARS), 12))
for i in idx:
    yi = year[i] - 2011
    for g in grams[i]:
        k = vi.get(g)
        if k is not None:
            Yc[k, yi] += 1
            Mc[k, yi, moy[i]] += 1
ymt = np.zeros((len(YEARS), 12))
for i in idx:
    ymt[year[i] - 2011, moy[i]] += 1
words = []
for g, k in vi.items():
    s = Yc[k] / np.array([ytot[y] for y in YEARS])
    sh = shape(s, Yc[k].sum())
    rate = Mc[k] / np.maximum(ymt, 1)  # per year-month share, then normalise within each year
    yr_mean = rate.mean(1, keepdims=True)
    ok = Yc[k] >= 5
    mm = (rate[ok] / np.maximum(yr_mean[ok], 1e-9)).mean(0) if ok.any() else np.ones(12)
    sh.update({"w": g, "series": [round(float(v), 5) for v in s], "season_month": int(np.argmax(mm)), "season": round(float(mm.max() / mm.mean()), 2),
               "months": [round(float(v), 2) for v in mm]})
    words.append(sh)
out["words"] = words

# ---- hand patterns
P = {
 # title shapes
 "starts 'Minecraft:'": r"^minecraft\s*[:\-|]", "ends 'in Minecraft'": r"in minecraft[\s!?.]*(\(|\[|#|$)", "has 'Minecraft' at all": r"minecraft",
 "all lowercase": None, "mostly CAPS": None, "question mark": r"\?", "!": r"!", "!!!": r"!!!", "?!": r"\?!|!\?", "...": r"\.\.\.|…",
 "*asterisk*": r"\*[^*]+\*", "(parentheses)": r"\(", "[brackets]": r"\[", "pipe |": r"\|", "hashtag": r"#\w", "colon": r":", "emoji": None,
 "episode/part/day number": r"\b(ep|episode|part|day)\.?\s*#?\d+|#\d+", "'but'": r"\bbut\b", "'vs'": r"\bvs\.?\b", "w/ or ft.": r"\bw/|\bft\.?\s|\bfeat\.?\s|\bwith\b",
 "'I' first person": r"^i\b|\bi\b", "'we'": r"\bwe\b|\bwe're\b", "'you'": r"\byou\b|\byour\b", "'my'": r"\bmy\b", "'he/she'": r"\b(he|she)\b",
 "'every'": r"\bevery\b", "'only'": r"\bonly\b", "'never'": r"\bnever\b", "'world's'": r"world'?s", "'most'": r"\bmost\b", "'ever'": r"\bever\b",
 "'secret'": r"\bsecret", "'impossible'": r"impossible", "'cursed'": r"cursed", "'scary'": r"scar(y|iest)", "'funny'": r"funny", "'epic'": r"\bepic\b",
 "'insane'": r"insane", "'op'": r"\bop\b|overpowered", "'ultimate'": r"ultimate", "'challenge'": r"challenge", "'survival'": r"surviv",
 "'hardcore'": r"hardcore", "'tutorial/how to'": r"tutorial|how to", "'build'": r"\bbuil(d|t)", "'house'": r"\bhouse", "'base'": r"\bbase\b",
 "'farm'": r"\bfarm", "'redstone'": r"redstone", "'mod'": r"\bmods?\b|modded", "'modpack'": r"mod ?pack|ftb|tekkit", "'seed'": r"\bseeds?\b",
 "'map'": r"\bmaps?\b", "'server'": r"server", "'smp'": r"\bsmp\b", "'update'": r"update", "'snapshot'": r"snapshot", "'trolling'": r"troll",
 "'prank'": r"prank", "'girlfriend/boyfriend'": r"girlfriend|boyfriend|\bgf\b", "'sister/brother'": r"sister|brother", "'mom/dad'": r"\bmom\b|\bdad\b|mother|father",
 "'baby'": r"\bbaby", "'family'": r"family", "'noob'": r"noob", "'pro'": r"\bpro\b", "'hacker'": r"hacker", "'god'": r"\bgod\b", "'school'": r"school",
 "'life'": r"\blife\b", "'hours'": r"\bhours?\b", "'days'": r"\bdays?\b", "'years'": r"\byears?\b", "'$ money'": r"\$\d|\bmoney|\bdollars?", "'rich/poor'": r"\brich\b|\bpoor\b",
 "'speedrun'": r"speedrun", "'manhunt'": r"manhunt", "'hide and seek'": r"hide (and|&|n) seek", "'parkour'": r"parkour", "'pvp'": r"\bpvp\b",
 "'skyblock'": r"sky ?block", "'bedwars'": r"bed ?wars", "'hunger games'": r"hunger games", "'uhc'": r"\buhc\b", "'lucky block'": r"lucky ?block",
 "'animation'": r"animation", "'song/parody'": r"\bsong\b|parody|music video", "'reacting'": r"\breact", "'top 10'": r"\btop \d+", "'facts'": r"\bfacts?\b",
 "'myth'": r"\bmyths?\b", "'tested'": r"\btest(ed|ing)\b", "'be like'": r"be like", "'types of'": r"types of", "'pov'": r"\bpov\b", "'when you'": r"when you",
 "'ai'": r"\bai\b|chatgpt", "'realistic'": r"realistic", "'1 block'": r"one block|1 block", "'100 players'": r"100 players", "'100 days'": r"100 days",
 "'24 hours'": r"24 hours", "'1000'": r"\b1,?000\b", "'1,000,000'": r"1,?000,?000|million",
 # mobs
 "creeper": r"creeper", "zombie": r"zombie", "skeleton": r"skeleton", "enderman": r"enderm[ae]n", "ender dragon": r"ender ?dragon|\bdragon\b",
 "wither": r"\bwither\b", "villager": r"villager", "pillager": r"pillager", "iron golem": r"golem", "pig": r"\bpigs?\b", "cow": r"\bcows?\b",
 "chicken": r"chicken", "sheep": r"\bsheep\b", "wolf/dog": r"\bwol(f|ves)\b|\bdogs?\b", "cat": r"\bcats?\b", "horse": r"\bhorses?\b", "slime": r"\bslime",
 "ghast": r"\bghast", "blaze": r"\bblaze", "spider": r"spider", "witch": r"\bwitch", "guardian": r"guardian", "phantom": r"phantom", "piglin": r"piglin",
 "hoglin": r"hoglin", "strider": r"strider", "axolotl": r"axolotl", "goat": r"\bgoats?\b", "warden": r"warden", "allay": r"\ballay", "frog": r"\bfrogs?\b",
 "sniffer": r"sniffer", "camel": r"\bcamels?\b", "armadillo": r"armadillo", "breeze": r"\bbreeze", "creaking": r"creaking", "happy ghast": r"happy ghast",
 "herobrine": r"herobrine", "entity 303": r"entity 303", "steve": r"\bsteve\b", "alex": r"\balex\b", "notch": r"\bnotch\b",
 # stuff
 "diamond": r"diamond", "emerald": r"emerald", "gold": r"\bgold", "iron": r"\biron\b", "netherite": r"netherite", "obsidian": r"obsidian", "tnt": r"\btnt\b",
 "bedrock (block/edition)": r"\bbedrock\b", "lava": r"\blava\b", "water": r"\bwater\b", "nether": r"\bnether\b", "end": r"\bthe end\b",
 "ancient city": r"ancient city", "deep dark": r"deep dark", "cherry": r"cherry", "trial chamber": r"trial chamber", "mace": r"\bmace\b", "elytra": r"elytra",
 "castle": r"castle", "mansion": r"mansion", "village": r"\bvillage\b", "island": r"island", "ocean": r"\bocean", "jungle": r"jungle", "desert": r"desert",
 "cave": r"\bcaves?\b", "sky": r"\bsky\b", "ice/snow": r"\bice\b|\bsnow", "potion": r"potion", "enchant": r"enchant", "sword": r"sword", "armor": r"armou?r",
 "rainbow": r"rainbow", "giant": r"\bgiant\b", "tiny": r"\btiny\b",
 # crossovers
 "pokemon/pixelmon": r"pok[eé]mon|pixelmon", "fnaf": r"fnaf|five nights|freddy", "among us": r"among us", "fortnite": r"fortnite", "roblox": r"roblox",
 "mario": r"\bmario\b", "sonic": r"\bsonic\b", "spider-man": r"spider-?man", "hulk": r"\bhulk\b", "batman": r"batman", "superhero": r"superhero|super hero",
 "harry potter": r"harry potter|hogwarts", "star wars": r"star wars", "frozen/elsa": r"frozen|\belsa\b", "minions": r"minion", "squid game": r"squid game",
 "skibidi": r"skibidi", "poppy playtime": r"poppy playtime|huggy", "rainbow friends": r"rainbow friends", "garten of banban": r"banban", "digital circus": r"digital circus",
 "sprunki": r"sprunki", "brainrot": r"brainrot|tralalero|tung tung", "mrbeast": r"mr ?beast", "titanic": r"titanic", "zombie apocalypse": r"apocalypse",
 "christmas": r"christmas|xmas", "halloween": r"halloween|spooky", "easter": r"easter", "valentine": r"valentine", "april fools": r"april fool", "summer": r"summer",
 "birthday": r"birthday", "school (back to)": r"back to school", "thanksgiving": r"thanksgiving", "new year": r"new year",
}
low = [t.lower() for t in titles]
EMO = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
pats = {}
for name, rx in P.items():
    if name == "all lowercase":
        hit = np.array([t == t.lower() and any(c.isalpha() for c in t) for t in titles])
    elif name == "mostly CAPS":
        hit = np.array([(lambda a: len(a) >= 8 and sum(c.isupper() for c in a) / len(a) > 0.7)([c for c in t if c.isalpha()]) for t in titles])
    elif name == "emoji":
        hit = np.array([bool(EMO.search(t)) for t in titles])
    else:
        c = re.compile(rx)
        hit = np.array([bool(c.search(t)) for t in low])
    s = series(hit)
    sh = shape(s, (hit & keep).sum())
    sm, sv, mm = season(hit)
    sh.update({"name": name, "series": s, "season_month": sm, "season": sv, "months": mm,
               "shorts_ratio": round(float((hit & keep & (r["kind"] == 1) & (year >= 2023)).sum() / max(1, (keep & (r["kind"] == 1) & (year >= 2023)).sum()) /
                                     max(1e-6, (hit & keep & (r["kind"] == 0) & (year >= 2023)).sum() / max(1, (keep & (r["kind"] == 0) & (year >= 2023)).sum()))), 2)})
    pats[name] = sh
out["patterns"] = pats

# ---- numbers
num = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?![\w.])")
bign = defaultdict(list)
nc = {y: Counter() for y in YEARS}
for i in idx:
    ns = [int(x.replace(",", "")) for x in num.findall(titles[i])]
    ns = [n for n in ns if not (1990 <= n <= 2030)]
    if ns:
        bign[int(year[i])].append(max(ns))
        for n in set(ns):
            nc[int(year[i])][n] += 1
out["numbers"] = {y: {"has": round(len(bign[y]) / ytot[y], 3), "median_max": float(np.median(bign[y])), "p90": float(np.percentile(bign[y], 90)),
                      "share_ge_1000": round(float(np.mean(np.array(bign[y]) >= 1000)), 3), "top": nc[y].most_common(8)} for y in YEARS}

# ---- versions
ver = re.compile(r"\b1\.(\d{1,2})\b")
vy = {y: Counter() for y in YEARS}
for i in idx:
    for v in set(ver.findall(low[i])):
        if 0 < int(v) <= 22:
            vy[int(year[i])][int(v)] += 1
out["versions"] = {y: {"share": round(sum(vy[y].values()) / ytot[y], 4), "top": vy[y].most_common(4)} for y in YEARS}

# ---- exact lengths: spikes at round numbers
lens = {}
for era, a, b in (("2011-2015", 2011, 2015), ("2016-2019", 2016, 2019), ("2021-2026", 2021, 2026)):
    s = [dur.get(ids[i], 0) for i in idx if a <= year[i] <= b and r["kind"][i] == 0]
    c = Counter(x for x in s if 60 < x < 3600)
    tot = sum(c.values())
    # spike = this second vs the 10 seconds either side
    spikes = []
    for x in range(120, 3600):
        nb = np.mean([c[x + d] for d in range(-10, 11) if d])
        if c[x] >= 30 and c[x] > 2.5 * nb:
            spikes.append((x, c[x], round(c[x] / max(nb, 1), 1)))
    over = lambda t: round(sum(v for k, v in c.items() if t <= k < t + 60) / max(1, sum(v for k, v in c.items() if t - 60 <= k < t)), 2)
    lens[era] = {"n": tot, "spikes": sorted(spikes, key=lambda z: -z[2])[:12], "just_over_10min": over(600), "just_over_8min": over(480)}
    hist = np.histogram([x for x in s if 0 < x < 1800], bins=180, range=(0, 1800))[0]
    lens[era]["hist10s"] = hist.tolist()
out["lengths"] = lens

# ---- days
dc = Counter(w.date() for w in when.values())
top_days = dc.most_common(10)
out["busiest_days"] = []
for d, n in top_days:
    ws = Counter(g for i, w in when.items() if w.date() == d for g in grams[i])
    out["busiest_days"].append({"date": str(d), "n": n, "words": [w for w, _ in ws.most_common(8)]})
wd = defaultdict(lambda: np.zeros(7))
for i, w in when.items():
    wd[w.year][w.weekday()] += 1
out["weekday"] = {y: (wd[y] / wd[y].sum()).round(3).tolist() for y in YEARS}

# ---- emoji per year
emo = {y: Counter() for y in YEARS}
for i in idx:
    for e in set(EMO.findall(titles[i])):
        emo[int(year[i])][e] += 1
out["emoji"] = {y: emo[y].most_common(6) for y in YEARS}

# ---- title length
tl = {y: [len(titles[i]) for i in idx if year[i] == y] for y in YEARS}
out["title_chars"] = {y: float(np.median(tl[y])) for y in YEARS}

json.dump(out, open(os.path.join(D, "scan.json"), "w"), ensure_ascii=False)

# ---- report
def top(xs, key, n=15, f=lambda x: x):
    return sorted(xs, key=key)[:n]
W_ = [w for w in words if w["n"] >= 300]
print("\n== ONE-YEAR WONDERS (peaky)")
for w in top(W_, lambda w: -w["peaky"]): print(f"  {w['w']:24s} peak {w['peak']} {w['peak_share']:.3%} x{w['peaky']}")
print("\n== BIGGEST RISERS")
for w in top(W_, lambda w: -w["rise"]): print(f"  {w['w']:24s} rise {w['rise']} peak {w['peak']}")
print("\n== BIGGEST FALLERS")
for w in top(W_, lambda w: w["rise"]): print(f"  {w['w']:24s} rise {w['rise']} peak {w['peak']}")
print("\n== COMEBACKS")
for w in top(W_, lambda w: -w["comeback"]): print(f"  {w['w']:24s} comeback x{w['comeback']} {' '.join(f'{v:.3f}' for v in w['series'])}")
print("\n== MOST SEASONAL WORDS")
MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
for w in top([w for w in W_ if w["n"] >= 500], lambda w: -w["season"], 25): print(f"  {w['w']:24s} {MON[w['season_month']]} x{w['season']}")
print("\n== PATTERNS")
for p in sorted(pats.values(), key=lambda p: -abs(p["rise"])):
    print(f"  {p['name']:26s} rise {p['rise']:+.2f} peak {p['peak']} {p['peak_share']:.2%} season {MON[p['season_month']]} x{p['season']} shorts/long {p['shorts_ratio']} comeback {p['comeback']}")
print("\n== NUMBERS"); [print(" ", y, v["has"], v["median_max"], v["p90"], v["share_ge_1000"], v["top"][:5]) for y, v in out["numbers"].items()]
print("\n== VERSIONS"); [print(" ", y, v) for y, v in out["versions"].items()]
print("\n== LENGTHS"); [print(" ", e, v["n"], "over10", v["just_over_10min"], "over8", v["just_over_8min"], v["spikes"][:8]) for e, v in lens.items()]
print("\n== BUSIEST DAYS"); [print(" ", d) for d in out["busiest_days"]]
print("\n== WEEKDAY (Mon..Sun)"); [print(" ", y, v) for y, v in out["weekday"].items() if y in (2011, 2014, 2017, 2020, 2023, 2026)]
print("\n== EMOJI"); [print(" ", y, v) for y, v in out["emoji"].items() if v]
print("\n== TITLE CHARS", out["title_chars"])
