"""Check the app's archive assistant: does it find, cite and decline correctly?

Uses the app's own search and model call (app/localrim/assistant.py). A light check, not
a full evaluation: the questions were written from the records themselves.

Round 1: 14 questions with one known record each (2 per department), phrased in other
words than the title, and 3 questions the archive can't answer. Per question: found (the
known record is among the 8 handed to the model, and its rank), cited, answered or
declined.
Round 2: 15 questions, 10 of them written client-style before searching the archive,
then matched to the records that answer them by keyword search; broad questions have a
set of relevant records. Cited records outside the set are listed for a person to judge.

Faithfulness (is every claim supported by the cited page) is judged by a person in the
"faithful" column.

Reads outputs/app/archive.csv (built by app/build_archive.py).
Writes results/archive_check/archive_check.csv and archive_check_summary.csv (round 1),
archive_check_r2.csv and archive_check_r2_summary.csv (round 2).

Usage (Ollama running):
    python pipeline/05_validate/archive_check.py      # round 1
    python pipeline/05_validate/archive_check.py 2    # round 2
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))  # the app's search
from localrim.assistant import DECLINE, ask_llm, cited, retrieve, type_words

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "outputs" / "app"
OUT = ROOT / "results" / "archive_check"

QUESTIONS = [
    # (department, question, known record's file ID); None = not in the archive
    ("MKT", "How did Marlboro's volume change in supermarkets compared with convenience stores at the end of 1992?", "PM2078275238"),
    ("MKT", "Which out-of-home media contract did Starcom set up with Springfield Advertising in 1999?", "PM2084600827"),
    ("RND", "Is there a study on cough and phlegm in schoolchildren whose parents smoke?", "PM2024225643"),
    ("RND", "What was the goal of the nicotine bioconversion work reported for the fourth quarter of 1991?", "PM2021398174"),
    ("AFF", "What did the Civil Aeronautics Board circulate about no-smoking sections on airplanes in 1975?", "PM2025014763"),
    ("AFF", "What does the 2000 CEO issues book say about the company's environmental policy?", "PM2082105914"),
    ("ADM", "Who directed human resource planning and development in the 1987 organization chart?", "PM2055006084"),
    ("ADM", "Was there a records transfer of 1993 files?", "PM2054600396"),
    ("LEG", "What did the Department of Justice write to Arnold & Porter about tobacco advertising in May 1995?", "PM2041207160"),
    ("LEG", "Find the draft certificate of incorporation for the Center for Indoor Air Research.", "PM2023555383"),
    ("QSH", "What tar and nicotine levels were measured for Marlboro Lights 25's soft pack after it was introduced?", "PM2074446823"),
    ("QSH", "What are the phosphorus and ammonia targets for the reconstituted blend sheet?", "PM2036001037"),
    ("FIN", "How much would it cost to print an extra million parent brochures?", "PM2085724227C"),
    ("FIN", "What did Metromedia Technologies invoice Philip Morris for in July 1997?", "PM2071227701"),
    ("NONE", "What dividend did Philip Morris pay per share in the second quarter of 2010?", None),
    ("NONE", "What did the company decide about remote work for employees during the COVID-19 pandemic?", None),
    ("NONE", "What dishes were served at the 1985 company holiday party?", None),
]


# Round 2: (kind, department, question, relevant file IDs, what a right answer says)
CLIENT, MULTI, HAND, NOWORDS, NONE = "client", "several records", "handwritten", "no shared words", "not in archive"
QUESTIONS_2 = [
    (CLIENT, "MKT", "Do we have any billboard or outdoor advertising contracts from the 1980s?",
     ["PM2040758683", "PM2043894635"], "1982 outdoor bulletin display contract (Leo Burnett)"),
    (CLIENT, "MKT", "What Marlboro promotions or giveaways did we run?",
     ['PM2041621969', 'PM2042958488', 'PM2044797227', 'PM2045250429', 'PM2048486325', 'PM2048671404',
      'PM2060154558', 'PM2063462397', 'PM2070119358', 'PM2070538323', 'PM2075861935', 'PM2077588583',
      'PM2078261039A', 'PM2083349931'], "Any Marlboro promotion records"),
    (CLIENT, "RND", "Do we have research on the health effects of secondhand smoke?",
     ['PM2023377413', 'PM2023523493', 'PM2024015048B', 'PM2024195436', 'PM2025523983', 'PM2028651587',
      'PM2029041787', 'PM2029234370', 'PM2041232477', 'PM2046228126', 'PM2046349673', 'PM2057342738',
      'PM2063790521', 'PM2063832428', 'PM2063906236', 'PM2064771051', 'PM2064821526', 'PM2065409918C',
      'PM2072148796', 'PM2081001711', 'PM2505624254', 'PM2505933402', 'PM3007216217', 'PM3052280698'],
     "Passive smoking / ETS studies and articles"),
    (CLIENT, "RND", "Is there any lab testing on menthol?",
     ['PM1001526247', 'PM1001856607', 'PM1003396051', 'PM2001288503', 'PM2024545692', 'PM2024760779',
      'PM2031305426', 'PM2040064097', 'PM2053424276', 'PM2053429374', 'PM2053429549', 'PM2056171503',
      'PM2057659221', 'PM2057672112', 'PM2069720595', 'PM2069728140', 'PM2072569078', 'PM2074851275',
      'PM2082850210', 'PM3007216217', 'PM3012472726', 'PM3034664598'],
     "R&D or Quality records on menthol products"),
    (CLIENT, "AFF", "Do we have anything on smoking bans in workplaces or restaurants?",
     ["PM2081001711", "PM2022972194", "PM2041232477", "PM2078324408", "PM2025435229A", "PM2065485855",
      "PM2024195436", "PM2025014763"], "e.g. 2000 testimony on New Hampshire's restaurant ban; 1987 workplace ban article; OSHA talking points"),
    (CLIENT, "AFF", "Are there letters to members of Congress?",
     ["PM2046982178", "PM2071166320", "PM2021336012"], "1995 letter to Congressman Bliley; responses to Senator Grassley"),
    (CLIENT, "LEG", "Do we have any subpoenas?",
     ["PM2074979592", "PM2074249653"], "Two litigation records that mention subpoenas (1998, 1999)"),
    (CLIENT, "LEG", "Are there trademark registrations for our brands?",
     ['PM2020049182', 'PM2020050367', 'PM2026169326', 'PM2026369296', 'PM2026464255', 'PM2076071066'],
     "Patents and trademarks records"),
    (NONE, "QSH", "Were there any complaints or problems with cigarette packaging quality?",
     [], "Nothing on packaging quality; should decline or say so"),
    (CLIENT, "FIN", "What did we budget for event sponsorships?",
     ["PM2073071105", "PM2041046634", "PM2073063433", "PM2042826250", "PM2043635910", "PM2040884923",
      "PM2061010703", "PM2041920185"], "1996-97 Virginia Slims music program budget (club nights, tours)"),
    (MULTI, "MKT", "Which records do we have about Marlboro's auto racing sponsorship?",
     ['PM2040599061', 'PM2045269580', 'PM2045412361', 'PM2048671404', 'PM2070510342', 'PM2070538323',
      'PM2073063433', 'PM2074509913', 'PM2080721257', 'PM2083790239', 'PM2083794235'], "11 racing records"),
    (MULTI, "QSH", "List the material safety data sheets we have on file.",
     ['PM2051535074', 'PM2051535221', 'PM2051541660', 'PM2057410088', 'PM2057439341'], "5 MSDS (1985-1991)"),
    (HAND, "MKT", "How many people came to the South Dallas bar night on Saturday, and in which market?",
     ["PM2048564690"], "1,260 on Saturday (975 Friday); McAllen, TX (handwritten form)"),
    (HAND, "RND", "In the July 1990 PMT update notes, how much IEF sample was available for Bruce?",
     ["PM2024836940"], "45 µl at 2 µg/µl, about 50% of what Bruce wanted (handwritten notes)"),
    (NOWORDS, "MKT", "Did we set up a booth at a music festival in the mid-90s?",
     ["PM2041920185"], "1996 Country Jam U.S.A. sponsorship agreement (Marlboro Country Cookbook grill)"),
]


def round2() -> None:
    """Run round 2 and save its results."""
    archive = pd.read_csv(APP / "archive.csv", dtype=str).fillna("")
    tax = yaml.safe_load((ROOT / "taxonomy" / "rim_taxonomy_final.yaml").read_text(encoding="utf-8"))
    types = type_words({t["id"]: t["name"] for f in tax["functions"] for s in f["series"] for t in s["types"]})
    ids = archive["id"].tolist()
    rows = []
    for n, (kind, dept, q, rel, expect) in enumerate(QUESTIONS_2, 1):
        t0 = time.time()
        docs = retrieve(q, archive, ids, types)
        answer = ask_llm(q, docs)
        secs = round(time.time() - t0, 1)
        if answer is None:
            raise SystemExit("Ollama is not answering. Start Ollama and run again.")
        c = cited(answer, len(docs))
        declined = DECLINE in answer.lower() or not c
        got = docs["file_id"].tolist()
        rel_set = set(rel)
        ranks = [i + 1 for i, f in enumerate(got) if f in rel_set]
        cited_ids = [got[i - 1] for i in c]
        rows.append({
            "n": n, "kind": kind, "department": dept, "question": q, "right_answer": expect,
            "relevant_records": len(rel),
            "found": bool(ranks) if rel else None, "first_rank": ranks[0] if ranks else None,
            "relevant_in_8": len(ranks) if rel else None,
            "cited_relevant": sum(f in rel_set for f in cited_ids) if rel else None,
            "cited_other": "; ".join(f for f in cited_ids if f not in rel_set),
            "declined": declined, "faithful": "", "seconds": secs, "answer": answer,
            "cited_file_ids": "; ".join(cited_ids)})
        r = rows[-1]
        print(f"{n:2d}. {kind:15s} found={r['found']} first={r['first_rank']} in8={r['relevant_in_8']} "
              f"cited_rel={r['cited_relevant']} other=[{r['cited_other']}] declined={declined} ({secs}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "archive_check_r2.csv", index=False)
    ans = df[df["relevant_records"] > 0]
    summ = []
    for kind, g in [("all answerable", ans)] + list(ans.groupby("kind")):
        summ.append({"group": kind, "questions": len(g), "found_in_top_8": int(g["found"].sum()),
                     "found_first": int((g["first_rank"] == 1).sum()),
                     "cited_a_relevant_record": int((g["cited_relevant"] > 0).sum()),
                     "declined_wrongly": int(g["declined"].sum())})
    out = df[df["relevant_records"] == 0]
    summ.append({"group": "not in archive", "questions": len(out), "correctly_declined": int(out["declined"].sum())})
    summ.append({"group": "time", "median_seconds": float(df["seconds"].median())})
    pd.DataFrame(summ).to_csv(OUT / "archive_check_r2_summary.csv", index=False)
    print("\n" + pd.DataFrame(summ).to_string(index=False))


def main() -> None:
    """Run round 1 (or round 2 with argument 2) and save the results."""
    archive = pd.read_csv(APP / "archive.csv", dtype=str).fillna("")
    tax = yaml.safe_load((ROOT / "taxonomy" / "rim_taxonomy_final.yaml").read_text(encoding="utf-8"))
    types = type_words({t["id"]: t["name"] for f in tax["functions"] for s in f["series"] for t in s["types"]})
    ids = archive["id"].tolist()
    by_file = dict(zip(archive["file_id"], archive["id"]))

    rows = []
    for n, (dept, q, file_id) in enumerate(QUESTIONS, 1):
        t0 = time.time()
        docs = retrieve(q, archive, ids, types)
        answer = ask_llm(q, docs)
        secs = round(time.time() - t0, 1)
        if answer is None:
            raise SystemExit("Ollama is not answering. Start Ollama and run again.")
        c = cited(answer, len(docs))
        declined = DECLINE in answer.lower() or not c
        found_ids = docs["id"].tolist()
        target = by_file.get(file_id) if file_id else None
        rank = found_ids.index(target) + 1 if target in found_ids else None
        rows.append({
            "n": n, "department": dept, "question": q, "known_file_id": file_id or "",
            "found": None if file_id is None else rank is not None, "rank": rank,
            "cited": None if file_id is None else (rank is not None and rank in c),
            "answered": None if file_id is None else not declined,
            "declined": declined if file_id is None else None,
            "faithful": "", "seconds": secs, "answer": answer,
            "cited_file_ids": "; ".join(docs.loc[[i - 1 for i in c], "file_id"]) if c else ""})
        print(f"{n:2d}. {dept:4s} found={rows[-1]['found']} rank={rank} cited={rows[-1]['cited']} "
              f"declined={declined} ({secs}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "archive_check.csv", index=False)
    ans = df[df["known_file_id"] != ""]
    out = df[df["known_file_id"] == ""]
    summary = pd.DataFrame([{
        "answerable_questions": len(ans),
        "found_in_top_8": int(ans["found"].sum()),
        "found_first": int((ans["rank"] == 1).sum()),
        "cited_the_known_record": int(ans["cited"].sum()),
        "answered_not_declined": int(ans["answered"].sum()),
        "out_of_archive_questions": len(out),
        "correctly_declined": int(out["declined"].sum()),
        "median_seconds": float(df["seconds"].median()),
    }])
    summary.to_csv(OUT / "archive_check_summary.csv", index=False)
    print("\n" + summary.T.to_string(header=False))
    print(f"\nSaved to {OUT}. Fill the 'faithful' column by reading each answer against its cited page.")


if __name__ == "__main__":
    import sys
    round2() if sys.argv[1:] == ["2"] else main()
