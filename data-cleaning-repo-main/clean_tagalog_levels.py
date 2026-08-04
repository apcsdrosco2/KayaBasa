"""
clean_tagalog_levels.py — Narrow the Tagalog corpus down to narrative text using
one shared metadata configuration.  By default it rebuilds the two levels the
shared config handles well:

    data/tag_lvl{2,3}_data.txt   ->   data/tag_lvl{2,3}_cleaned_data.txt
    (one cleaned narrative per record, records separated by "\\n\\n=====\\n\\n",
     matching the data/tag_lvl1_cleaned_data.txt convention; raw input untouched)

The metadata vocabulary (_META, _CREDIT, _PUBTAIL, …) is identical for every
level — what differs is the container format and how the front matter is trimmed:

  * lvl1  ABC+/DepEd decodables.  Heavy, REPEATED credit blocks (Isinurat /
          Idrinowing / Inedit nira …), bylines, GRADE 1 / DONATED PROPERTY
          markers, bilingual Bikol.  These need bespoke handling that the shared
          config regresses, so lvl1 keeps its hand-curated cleaned file and is
          NOT rebuilt by main() (see the LEVELS note below).  clean_doc(t, 1)
          exists for completeness.
  * lvl2  Let's Read / StoryWeaver translations shipped as a COMMA container
          "<title>,2,<story>".  We strip the title+level prefix; the body is then
          clean prose with only light noise.
  * lvl3  The messiest: the narrative is buried in the MIDDLE of each record,
          wrapped in large front- and back-matter.  We drop metadata SENTENCES in
          place (never truncate at a leading anchor), then locate the narrative
          start as the first sustained run of native Tagalog prose.

Pipeline per record: container split -> encoding/junk repair -> credit/role/code/
page stripping -> per-sentence metadata drop -> narrative-start trim (a doc needs
one strong native sentence or a run of >=2 weaker ones, else it is metadata-only
and dropped) -> tail cut (activity / comprehension / bio / publisher).
"""

import os
import regex

import build_datasets as bd
import clean_tagalog as ct

bd_ENG = ct._ENG                       # English boilerplate vocabulary (reused)

LEVELS = (1, 2, 3)
SRC = "data/tag_lvl{lvl}_data.txt"
DST = "data/tag_lvl{lvl}_cleaned_data.txt"

# ── native Tagalog/Filipino function words ────────────────────────────────────
# Dense in narrative, absent from English boilerplate and from capitalised
# name/role runs.  Used to recognise where real prose begins.
_NATIVE = set((
    "ang ng nang sa na ay mga si ni nina kay kina ako ka ikaw siya niya kanya "
    "kami tayo namin naming natin kayo ninyo sila nila ito iyon iyan nito niyan niyon "
    "dito diyan doon hindi wala may mayroon meron kung dahil pero ngunit subalit "
    "at o kaya naman lang lamang din rin ba pa para upang nga raw daw po ho "
    "isang isa noong nung habang tuwing matapos bago saka sapagkat kaysa "
    "kapag pag bawat kahit maging dapwat ako'y siya'y ako’y siya’y"
).split())

# Short native words that must NOT be mistaken for a stray leading stub.
_NATIVE_SHORT = {"si", "sa", "ng", "ay", "na", "at", "ni", "o", "ba", "pa",
                 "po", "ho", "ka", "ko", "mo", "a"}

# ── metadata sentence patterns (Tagalog + English fragments) ──────────────────
# A sentence/segment matching any of these is front/back-matter, not story.
_META = regex.compile(
    r"(?i)(?:"
    # Tagalog copyright / production boilerplate
    r"karapatang[\s\-]?ari|atas\s+ng\s+pangulo|seksiyon\s+9|"
    r"walang\s+karapatang|republika\s+ng\s+pilipinas|"
    r"kagawaran\s+ng\s+edukasyon|rehiyon\s+[ivx]+|sangay\s+ng|"
    r"ang\s+kagamitan\s+na\s+ito|sanayang[\s\-]?aralin|"
    r"maaa?ring\s+maireprodyus|maaa?ring\s+(?:batayan|kopyahin)|"
    r"inilathala\s+ng|inilimbag\s+ng|pinaunlad\s+na|"
    r"curriculum\s+implementation|learning\s+resources?\s+management|lrmds|lrms|"
    r"pag[\s\-]?aari\s+ng\s+pamahalaan|hindi\s+ipinagbibili|"
    r"paunang\s+salita|paalala\s+sa\s+mambabasa|"
    r"leveled\s+reader|(?:first|second|third|fourth)\s+edition|ikalawang\s+edisyon|"
    r"all\s+rights\s+reserved|"
    r"\bcopyright\b|presidential\s+decree|department\s+of\s+education|deped|"
    r"no\s+part\s+of\s+this|this\s+material|brought\s+to\s+you\s+by|"
    r"basa\s+pilipinas|u\.?s\.?\s+agency|usaid|asia\s+foundation|"
    r"schools?\s+division|district\s+supervisor|public\s+schools|"
    r"quality\s+assurance|education\s+program\s+supervisor|"
    r"project\s+development\s+officer|school\s+head|principal\s+[–\-]?i+\b|"
    r"\bceso\b|\boic\b|ph\.?\s?d\b|ed\.?\s?d\b|assistant\s+schools|"
    r"visual\s+graphic\s+artist|tagamasid\s+pampurok|distrito\s+ng|"
    r"masusing\s+sinuri|ipinagtibay|walang\s+(?:bahagi|anumang)|"
    r"maaa?ring\s+ipagbili|de\s+kalidad\s+na\s+edukasyon|"
    r"planong\s+pagkatuto|instruksyunal\s+na\s+kagamitan|"
    # lvl1 ABC+/DepEd structural markers + SIL/Resources-for-the-Blind/CC license
    r"donated\s+property|not\s+for\s+sale|\bgrade\b|©|\bsil\s+international|"
    r"art\s+of\s+reading|resources\s+for\s+the\s+blind|all\s+children\s+reading|"
    r"by[\s\-]?sa\b|community\s+living|"
    # lvl1 foreword / intro blurb
    r"\bpanimula\b|\binaasahan\b|malaking\s+aklat|para\s+sa\s+mga\s+mag[\s\-]?aaral|"
    # prefatory reader-note / learning-objective blurbs (read like prose)
    r"para\s+sa\s+(?:mga\s+)?bata|para\s+sa\s+(?:mga\s+)?(?:magulang|guro)|"
    r"ginawa\s+ang\s+aklat|ang\s+aklat\s+na\s+ito|ang\s+(?:kuwentong|kwentong|kuentong)\s+ito|"
    r"may\s+pama[\s\-]?gat|\blayun(?:in|on)\b|\bakda\s+ni|may[\s\-]?akda|ng\s+akda|"
    r"asignaturang|ika[\s\-]?\d+\s+(?:linggo|markahan)|"
    r"acting\s+asst|division\s+superinten|"
    # DepEd learning-competency / curriculum-standard sheets
    r"learning\s+competenc|(?:mga\s+)?kasanayan\s+sa\s+pagkatuto|"
    r"pamantayang?\s+pang|pamantayan\s+sa\s+pagganap|content\s+standard|"
    r"performance\s+standard|pag[\s\-]?uugnay\s+sa\s+kurikulum|kurikulum|\bk[\s\-]?12\b|"
    r"curriculum\s+information|\bkompetensi\b|\d(?:st|nd|rd|th)\s+quarter|quarter\s*:|"
    r"bachelor\s+in|elementary\s+education|"
    r"pag[\s\-]?unawa\s+sa\s+(?:binasa|napakinggan)|kagamitan\s+ng\s+mga\s+mag[\s\-]?aaral|"
    r"story\s?book|maikling\s+kuwento\s+sa\s+filipino|para\s+sa\s+ika\p{L}*\s+baitang|"
    r"sariling\s+karapatan|unang\s+edisyon|treasury\s+of\s+storybook|"
    r"national\s+competition|intellectual\s+property|copyright\s+page|kathang[\s\-]?isip|"
    r"grade\s+level\s+filipino|paglalarawan\s*:|"
    # more Tagalog copyright-page boilerplate
    r"isinasaad\s+sa\s+batas|batas\s+republika|seksiyon\s+\d+|karapatang[\s\-]?sipi|"
    r"kailangang?\s+muna\s+ang\s+pahintu|pamahalaan\s+ng\s+pilipinas|"
    r"paramihin\s+ang\s+alinmang|pahintulot\s+ng\s+may[\s\-]?ari|balimbagan|"
    r"management\s+and\s+development|alin\s*mang\s+bahagi|pinahihintulutang|"
    r"kilalanin\s+ang\s+may[\s\-]?ari|"
    r"contextualized\s+story|lesson\s+exemplar|activity\s+sheet|"
    r"to\s+the\s+teachers|development\s+team|learning\s+resource\s+manager|"
    r"deped[\s\-]?blr|\bblr\b|bureau\s+of\s+learning|"
    # English production-credit / approval fragments
    r"cid\s+chief|division\s+librarian|recommending\s+approval|approved\s+by|"
    r"\bapproval\b|reproduced\s+for\s+educational|educational\s+purposes|"
    r"digital\s+edition|for\s+print\s+and\s+online|lr\s+(?:evaluator|editor|portal|management)|"
    r"\bevaluator\b|head\s+teacher|master\s+teacher|\beps\b|\bpdo\b|"
    # competency skill-area / subject section headers
    r"pagpapahalaga\s*:|\bscience\b|\bmathematics\b|araling\s+panlipunan|"
    # bio fragments without a named header
    r"nagturo\s+sa|taong\s+nagturo|mahilig\s+siyang|"
    # residual copyright continuation + comprehension-question residue
    r"nasabing\s+ahensiya|kaukulang\s+bayad|pinagsumikapang|tagapaglathala|"
    r"ang\s+mga\s+tauhan|sino[\s\-]?sino|"
    # labeled metadata fields (Let's Read / DepEd record headers)
    r"learning\s+area|competenc|pool\s+id|\btitle\s*:|\bcode\s*:|\bid\s*:\s*\d|"
    # teacher-foreword / purpose / story-description blurb
    r"ito\s+ay\s+binuo\s+upang|mag[\s\-]?aaral\s+ng\s+kasanayan|bigyan\s+ng\s+oras|"
    r"magpapa(?:unlad|yaman)\s+|nawa\s+ito\s+ay|ay\s+isang\s+kuwento\s+na\s+maka|"
    r"ito\s+ay\s+batay\s+sa|ugaliin\s+ang\s+pagiging|"
    r"higit\s+sa\s+lahat\s+matutunan|ito\s+ay\s+makakatulong|makakatulong\s+(?:rin|din)|"
    r"makatutulong\s+sa\s+mga\s+(?:guro|mag)|"
    # publisher / license boilerplate (whole-sentence tails)
    r"released\s+under|pratham\s+books?|book\s+dash|storyweaver|story\s?weaver|"
    r"creative\s+commons|some\s+rights\s+reserved|terms\s+of\s+use|aral\s+na\s+makukuha|"
    # interleaved activity-section markers (drop the sentence, never truncate)
    r"skill\s+builder|\bgawain\s*:|\bhalimbawa\s*:|sundin\s+ang\s+halimbawa|"
    r"bumuo\s+ng\s+tanong|gamitin\s+ang\s+mga\s+salita|magbigay\s+ng\s+\w+\s+tanong|"
    r"sundin\s+(?:din\s+)?ang\s+(?:mga\s+)?(?:nakasulat\s+na\s+)?panuto|"
    r"nakapaloob\s+sa\s+ibang\s+kahon|"
    # author-bio markers ("About the author" prose that flanks the story)
    r"tungkol\s+sa\s+(?:mga\s+)?may[\s\-]?akda|ang\s+may[\s\-]?akda|kasalukuyang\s+guro|"
    r"maligayang\s+pagbasa|kasanayang?\s+pang(?:katuto|nilalaman)?|kasanayang\s+nakapaloob|"
    r"inaasahang\s+sa\s+pagbabasa|inaasahan\s+(?:ko|namin)|represents\s+number|variety\s+of\s+materials|"
    r"memorial\s+elementary|elementary\s+school|\bteacher\s+i+\b|\bt[\s\-]i{1,3}\b|"
    r"nagtapos\s+sa\s+(?:pamantasan|paaralan|unibersidad|pamana?tasan)|cum\s+laude|"
    r"batsilyer|bachelor\s+of|dalubhasa\s+sa|national\s+pool|"
    # competency action-verbs (nominalised -ang forms) that head objective lists,
    # and the skill-area labels (Pakikinig:/Pagbasa:/Pagsasalita:) before them
    r"\b(?:pakikinig|pagbasa|pagsasalita|pagsulat|pag[\s\-]?unawa)\s*[:\d]|"
    r"naiguguhit\b|napag[\s\-]?uugnay|nakapaglalarawan\w*|"
    r"\b(?:nasasagot|naka?kagamit|naka?gagamit|naka?susunod|natutukoy|nailalarawan|"
    r"naipa?pakita|nagagamit|naibibigay|nababago|napagsusunod|naiuugnay|naka?gagawa|"
    r"napatutunayan|naisasakilos|naisasagawa|naisasalaysay|naipahahayag|nababasa|"
    r"naipakikita|nakikinig|natatalakay|nasusuri|natatandaan|naipapaliwanag|"
    r"nakadarama|naisasaayos|nakasusulat|nakikilala|nahuhulaanang|nababaybay|"
    r"naisalaysay|nasasabi|nakapaglalarawan|naikukumpara|nakakabuo|naikukuwento)\b"
    r")"
)

# Credit clauses (verbs).  Verb + optional ni/nina/nira/nila + capitalised name
# run (up to ~8 tokens, Bikol "saka/asin" and "at/and" allowed as connectors).
_CREDIT = regex.compile(
    r"(?i)\b(?:isinulat|isinurat|isulat|sinulat|sumulat|iginuhit|ginuhit|"
    r"iguhit|idinrowing|idrinowing|inilarawan|isinalarawan|naglarawan|kinulayan|"
    r"kinukulay|tagawasto|isinatitik|sinuri|inedit|isinalin|pinatnugutan|"
    r"writer|author|illustrator|illustrated\s+by|written\s+by|story\s+by|"
    r"stories\s+by|illustrations?\s+by|reviewed\s+by|edited\s+by|adapted\s+by|"
    r"translated\s+by|layout|editor|reviewer|contributor|graphic\s+artist|"
    r"layout\s+artist|artist)"
    r"s?\b\s*(?:ni(?:na|ra)?|nila|ng|by)?\s*[:\-]?\s*"
    # name run: ni/nina/nira are connectors too, so chained DepEd credit blocks
    # ("Isinulat ni X Iginuhit ni Y Inedit nira Z, …") get consumed in one sweep.
    r"(?:\p{Lu}[\p{L}.'’\-]*\.?(?:\s+|,\s*|\s+(?:at|and|saka|asin|ni|nina|nira)\s+)){0,40}"
)

# Noun-form bylines ("Kuwento ni …", "Guhit ni …", "Salaysay ni …", "May-akda:").
# These words double as ordinary story words ("Tatlong kwento …"), so they only
# count as a credit when an explicit ni/nina/nira follows — never on their own.
_CREDIT_NI = regex.compile(
    r"(?i)\b(?:kuwento|kwento|salaysay|guhit|may[\s\-]?akda)\s+"
    r"(?:ni(?:na|ra)?|nila)\b\s*[:\-]?\s*"
    r"(?:\p{Lu}[\p{L}.'’\-]*\.?(?:\s+|,\s*|\s+(?:at|and|saka|asin)\s+)){0,8}"
)

# A run of >=2 consecutive ALL-CAPS word-tokens (names, role titles, initials).
_CAPS_RUN = regex.compile(r"(?:\b\p{Lu}[\p{Lu}'’.\-]*\b[ ,]*){2,}")

# Unambiguous back-matter SECTION headers (never inside narrative).
_TAIL = regex.compile(
    r"(?i)\b(?:pamprosesong\s+(?:mga\s+)?tanong|tungkol\s+sa\s+may[\s\-]?akda|"
    r"about\s+the\s+(?:author|illustrator|writer)|talasalitaan|sanggunian|"
    r"tabang\s+sa\s+pag[\s\-]?aadal|gabay\s+sa\s+pag[\s\-]?aaral|"
    r"mga\s+gawain\s*:|mga\s+tanong\s*:|sagutin\s+ang\s+mga\s+(?:sumusunod|tanong))"
)

# Publisher / license tail anchor (shared bd._PUB + StoryWeaver/Pratham closers).
# NOTE: byline ("Ang May-akda") and competency ("Mga Kasanayan") markers are
# deliberately NOT here — they can sit as FRONT/mid-matter before the story, so
# cutting at them would decapitate the narrative; they are per-sentence drops.
_PUBTAIL = regex.compile(
    r"(?i)(?:" + bd._PUB + r"|released\s+under|pratham\s+books?|book\s+dash|"
    r"story\s?weaver|©|cc[\s\-]?by|inilimbag\s+ng|end\s+of\s+story|"
    r"contributing\s+translators|city\s+college|general\s+education|"
    r"sa\s+suporta\s+ng|pag\s*rerecord|parasurat\s+bikolnon|"
    r"\bretrieved\b|viewtopic|\bhttps?\b|sanggunian)"
)

_PAGE = regex.compile(r"(?<!\S)\d{1,3}(?!\S)(?!\.\d)")
_CODE = regex.compile(r"\(?\b\p{Lu}{1,6}\d[\p{L}\d.\-]+\b\)?")
_JUNK = regex.compile("[�•●·­​‎‏]+")
_EMPTY_Q = regex.compile(r"[\"“”]\s*[\"“”]")
_ORPHAN_CREDIT = regex.compile(r"(?i)\b(?:ni|nina?|nila|kina)\s*:")
_PAGE_BANNER = regex.compile(r"(?i)\bpage\s*\|?\s*\d*")


def _container(text: str, lvl: int) -> str:
    """lvl2 ships as '<title>,<lvl>,<story>' — strip the title+level prefix.
    Harmless on lvl1/lvl3 (their records do not open with a ',<lvl>,' marker)."""
    m = regex.match(rf"^.{{1,100}}?,\s*{lvl}\s*,\s*", text)
    return text[m.end():] if m else text


def _is_meta_sentence(s: str) -> bool:
    toks = [w.strip(".,!?\"'()[]:;·—–“”’") for w in s.split()]
    toks = [w for w in toks if w and any(c.isalpha() for c in w)]
    if not toks:
        return True
    if _META.search(s):
        return True
    eng = sum(1 for w in toks if w.lower() in bd_ENG) / len(toks)
    if len(toks) >= 3 and eng >= 0.5:
        return True
    caps = sum(1 for w in toks if len(w) >= 2 and w.isupper())
    native = sum(1 for w in toks if w.lower() in _NATIVE)
    if caps >= 2 and native == 0:
        return True
    # English sentence (e.g. an appended parallel-translation block): a real
    # Tagalog sentence of this length always carries a native function word, so
    # >=4 ASCII-alphabetic tokens with NONE of them is English prose.
    ascii_alpha = sum(1 for w in toks if w.isascii() and w.isalpha())
    if len(toks) >= 4 and native == 0 and ascii_alpha >= 0.7 * len(toks):
        return True
    return False


def _narr_native(s: str) -> int:
    """If `s` is a clean candidate story sentence — >=4 tokens, not metadata, not
    English-dominated, not a capitalised name run, not interrogative — return its
    native-function-word count; otherwise -1.  Metadata is already stripped by
    _META upstream, so this only has to separate story from byline/credit noise."""
    if _is_meta_sentence(s) or s.rstrip().endswith("?"):
        return -1
    toks = [w.strip(".,!?\"'()[]:;·—–“”’") for w in s.split()]
    toks = [w for w in toks if w and any(c.isalpha() for c in w)]
    if len(toks) < 4:
        return -1
    eng = sum(1 for w in toks if w.lower() in bd_ENG) / len(toks)
    caps = sum(1 for w in toks if len(w) >= 2 and w.isupper())
    if eng >= 0.25 or caps > max(2, len(toks) // 4):
        return -1
    return sum(1 for w in toks if w.lower() in _NATIVE)


def _strip_name_runs(text: str) -> str:
    """Remove a run of >=4 consecutive non-native, Title-case tokens — a
    credit-verb-less author byline (incl. non-Tagalog names: "Syeda Basarat Midhat
    Kazim Maria Riaz", "Vinita Krishna Suvidha Mistry").  The >=4 threshold keeps
    ordinary multi-word proper nouns INSIDE the narrative ("Paaralang Marcos
    Pimentel", "Sina Ate, Kuya, at Nanay") — native words and the length cap keep
    those short."""
    toks = text.split()
    out, run = [], []
    for t in toks:
        w = t.strip(".,!?\"'’()[]:;-–—")
        if w and w[0].isupper() and not w.isupper() and w.lower() not in _NATIVE:
            run.append(t)
            if t[-1] in ".!?":          # don't let a run cross a sentence boundary
                if len(run) < 4:
                    out.extend(run)
                run = []
        else:
            if len(run) < 4:
                out.extend(run)
            run = []
            out.append(t)
    if len(run) < 4:
        out.extend(run)
    return " ".join(out)


def _strip_leading_names(text: str) -> str:
    """Drop a leading run of capitalised tokens that carry no native function word
    — a title + author byline (incl. non-Tagalog names like "Vinita Krishna
    Suvidha Mistry") glued to the first sentence.  Requires a run of >=2 such
    tokens so a story that simply opens on one capitalised content word
    ("Maganda ang umaga …") is left intact; stops at the first native word
    (Ang/Si/Sa/Noong …) or any lowercase-initial token, i.e. where prose begins.
    Also clears a stray leading 1-2 letter stub ("S Si Kaka …")."""
    toks = text.split()
    i = 0
    while i < min(len(toks), 14):
        w = toks[i].strip(".,!?\"'’()[]:;-–—")
        if w and w[0].isupper() and w.lower() not in _NATIVE:
            i += 1
        else:
            break
    if i >= 2:
        text = " ".join(toks[i:])
    elif (len(toks) > 1 and len(toks[0].strip("\".,!?")) <= 2
            and toks[0].strip("\".,!?").lower() not in _NATIVE_SHORT
            and toks[1][:1].isupper()):
        text = " ".join(toks[1:])
    return text


def clean_doc(text: str, lvl: int) -> str:
    text = _container(text, lvl)
    text = bd._basic_clean(text)
    text = bd._CANVAS_LICENSE.sub(" ", text)   # leading CANVAS attribution block
    text = _JUNK.sub(" ", text)
    text = regex.sub(r"_{2,}", " ", text)      # blank-line / fill-in underscores
    text = _CODE.sub(" ", text)
    # credit clauses: verbs (looped — lvl1 repeats the whole credit block up to 3x
    # and a swallowed 2nd verb shifts the next block), noun-form "… ni X" bylines,
    # orphan "ni:" residue, empty quotes
    for _ in range(4):
        new = _CREDIT.sub(" ", text)
        if new == text:
            break
        text = new
    text = _CREDIT_NI.sub(" ", text)
    text = _ORPHAN_CREDIT.sub(" ", text)
    text = _strip_name_runs(text)              # non-Tagalog author bylines
    text = _CAPS_RUN.sub(" ", text)
    text = _EMPTY_Q.sub(" ", text)
    text = _PAGE_BANNER.sub(" ", text)
    text = _PAGE.sub(" ", text)

    # Split into sentences; drop every metadata sentence in place.
    sents = regex.split(r"(?<=[.!?])\s+", text)
    sents = [s for s in sents if not _is_meta_sentence(s)]

    # Skip leading title/byline/blurb noise to the first genuine native sentence.
    # A doc only counts as a story if it has one STRONG sentence (>=3 function
    # words) or a RUN of >=2 weak ones (>=2 each) — a lone title sentence
    # ("Ang Prinsesang Nabura ang Mukha") followed by credit residue does NOT
    # qualify, so metadata-only records drop; simple repetitive readers
    # ("Gusto namin … si Samnang." / lvl1 "Lima ang lata ni Mila.") still pass.
    counts = [_narr_native(s) for s in sents]
    weak = [i for i, n in enumerate(counts) if n >= 2]
    strong = any(n >= 3 for n in counts)
    if not (strong or len(weak) >= 2) or not weak:
        return ""
    sents = sents[weak[0]:]

    text = regex.sub(r"\s+", " ", " ".join(sents)).strip()

    # Cut trailing activity / comprehension / bio / publisher sections.
    cut = len(text)
    for pat in (_TAIL, _PUBTAIL):
        m = pat.search(text)
        if m and m.start() >= 40:
            cut = min(cut, m.start())
    text = text[:cut]

    # Strip a residual leading title + author byline (incl. non-Tagalog names)
    # and stray page stubs.
    text = _strip_leading_names(text)

    # Drop a trailing run of comprehension questions / orphan stubs ("Si .", "K.").
    tail_sents = regex.split(r"(?<=[.!?])\s+", text.strip())

    def _is_tail_junk(s):
        if s.rstrip().endswith("?"):
            return True
        a = [w for w in s.split() if any(ch.isalpha() for ch in w)]
        return len(a) < 3

    while tail_sents and _is_tail_junk(tail_sents[-1]):
        tail_sents.pop()
    text = " ".join(tail_sents)

    return regex.sub(r"\s+", " ", text).strip()


def clean_level(lvl: int) -> int:
    src, dst = SRC.format(lvl=lvl), DST.format(lvl=lvl)
    if not os.path.isfile(src):
        print(f"lvl{lvl}: {src} not found, skipped")
        return 0
    lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
    docs = [b for b in (clean_doc(l, lvl) for l in lines)
            if b and len(b.split()) >= 8]
    with open(dst, "w", encoding="utf-8", newline="\n") as out:
        out.write("\n\n=====\n\n".join(docs) + "\n")
    print(f"lvl{lvl}: raw {len(lines)}  cleaned kept {len(docs)}  -> {dst}")
    return len(docs)


def main():
    for lvl in LEVELS:
        clean_level(lvl)


if __name__ == "__main__":
    main()
