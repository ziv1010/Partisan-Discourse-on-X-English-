#!/usr/bin/env python3
"""
Build keyword-specific stance comparison tables for two experiments.

Experiments:
1. Normal stance: fewshot_label_for_against from FINAL_STANCE.
2. Keyword temporal stance: lora_stance from TEMPORAL_SPLIT.

The main comparison is computed on the shared 2022-2023 temporal-labeled
rows, because that file contains both the normal label and the temporal
LoRA label for exactly the same tweet-keyword rows.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
JOIN_DIR = BASE_DIR.parent

FINAL_CSV = (
    JOIN_DIR
    / "FINAL_STANCE"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
)
TEMPORAL_LABELED_CSV = JOIN_DIR / "TEMPORAL_SPLIT" / "split_2022_2023_lora_labeled.csv"

VALID_PARTIES = ("pro ruling", "pro opposition")
PARTY_LABELS = {
    "pro ruling": "Pro-Ruling",
    "pro opposition": "Pro-Opposition",
}
DISPLAY_STANCES = ("favor", "against", "neutral")
STANCE_LABELS = {
    "favor": "Fav",
    "against": "Ag",
    "neutral": "Neu",
}

KEYWORD_DISPLAY_OVERRIDES = {
    "caa": "CAA",
    "gdp": "GDP",
    "msp": "MSP",
    "rahulgandhi": "Rahul Gandhi",
}


@dataclass
class AggregateResult:
    name: str
    source_csv: Path
    label_column: str
    counts: dict[str, dict[str, Counter]]
    rows_read: int
    rows_used: int
    skipped_keyword: int
    skipped_party: int
    skipped_stance: int


def clean_text(value: object) -> str:
    return str(value or "").strip().lower()


def normalize_party(value: object) -> str | None:
    party = clean_text(value)
    return party if party in VALID_PARTIES else None


def normalize_keyword(value: object) -> str | None:
    keyword = clean_text(value)
    return keyword or None


def normalize_stance(value: object) -> str | None:
    stance = clean_text(value)
    if stance in {"favor", "favour", "fav", "for", "support", "supports"}:
        return "favor"
    if stance in {"against", "ag", "oppose", "opposes"}:
        return "against"
    if stance in {"neutral", "neu"}:
        return "neutral"
    return None


def display_keyword(keyword: str) -> str:
    if keyword in KEYWORD_DISPLAY_OVERRIDES:
        return KEYWORD_DISPLAY_OVERRIDES[keyword]
    return keyword.title()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def party_counter(counts: dict[str, dict[str, Counter]], keyword: str, party: str) -> Counter:
    return counts.get(keyword, {}).get(party, Counter())


def stance_n(counts: Counter, stance: str) -> int:
    return int(counts.get(stance, 0))


def total_n(counts: Counter) -> int:
    return sum(stance_n(counts, stance) for stance in DISPLAY_STANCES)


def stance_pct(counts: Counter, stance: str) -> float:
    total = total_n(counts)
    if total == 0:
        return 0.0
    return stance_n(counts, stance) / total * 100.0


def rounded_pct_map(counts: Counter) -> dict[str, float]:
    """Round one-decimal stance percentages so displayed values sum to 100.0."""
    total = total_n(counts)
    if total == 0:
        return {stance: 0.0 for stance in DISPLAY_STANCES}

    raw_tenths = {stance: stance_n(counts, stance) * 1000.0 / total for stance in DISPLAY_STANCES}
    rounded_tenths = {stance: int(raw_tenths[stance]) for stance in DISPLAY_STANCES}
    remaining = 1000 - sum(rounded_tenths.values())

    by_remainder = sorted(
        DISPLAY_STANCES,
        key=lambda stance: (raw_tenths[stance] - rounded_tenths[stance], stance_n(counts, stance)),
        reverse=True,
    )
    for stance in by_remainder[:remaining]:
        rounded_tenths[stance] += 1

    return {stance: rounded_tenths[stance] / 10.0 for stance in DISPLAY_STANCES}


def keyword_total(counts: dict[str, dict[str, Counter]], keyword: str) -> int:
    return sum(total_n(party_counter(counts, keyword, party)) for party in VALID_PARTIES)


def keyword_order(*aggregates: AggregateResult) -> list[str]:
    keys: set[str] = set()
    for aggregate in aggregates:
        keys.update(aggregate.counts.keys())
    return sorted(keys)


def build_aggregate(name: str, source_csv: Path, label_column: str) -> AggregateResult:
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    rows_read = 0
    rows_used = 0
    skipped_keyword = 0
    skipped_party = 0
    skipped_stance = 0

    with source_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required = {"keyword", "_label_norm", label_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_csv} is missing required columns: {sorted(missing)}")

        for row in reader:
            rows_read += 1
            keyword = normalize_keyword(row.get("keyword"))
            party = normalize_party(row.get("_label_norm"))
            stance = normalize_stance(row.get(label_column))

            if keyword is None:
                skipped_keyword += 1
                continue
            if party is None:
                skipped_party += 1
                continue
            if stance is None:
                skipped_stance += 1
                continue

            counts[keyword][party][stance] += 1
            rows_used += 1

    return AggregateResult(
        name=name,
        source_csv=source_csv,
        label_column=label_column,
        counts={keyword: dict(parties) for keyword, parties in counts.items()},
        rows_read=rows_read,
        rows_used=rows_used,
        skipped_keyword=skipped_keyword,
        skipped_party=skipped_party,
        skipped_stance=skipped_stance,
    )


def distribution_rows(aggregate: AggregateResult, keywords: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keyword in keywords:
        row: dict[str, object] = {
            "aspect": display_keyword(keyword),
            "keyword": keyword,
            "total": keyword_total(aggregate.counts, keyword),
        }
        for party in VALID_PARTIES:
            prefix = party.replace(" ", "_")
            counts = party_counter(aggregate.counts, keyword, party)
            pct_map = rounded_pct_map(counts)
            for stance in DISPLAY_STANCES:
                row[f"{prefix}_{stance}_pct"] = pct_map[stance]
                row[f"{prefix}_{stance}_n"] = stance_n(counts, stance)
            row[f"{prefix}_n"] = total_n(counts)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_csv(path: Path, aggregates: list[AggregateResult], keywords: list[str]) -> None:
    fieldnames = [
        "experiment",
        "source_csv",
        "label_column",
        "keyword",
        "aspect",
        "party",
        "favor_n",
        "against_n",
        "neutral_n",
        "total_n",
        "favor_pct",
        "against_pct",
        "neutral_pct",
    ]
    rows: list[dict[str, object]] = []
    for aggregate in aggregates:
        for keyword in keywords:
            for party in VALID_PARTIES:
                counts = party_counter(aggregate.counts, keyword, party)
                pct_map = rounded_pct_map(counts)
                rows.append(
                    {
                        "experiment": aggregate.name,
                        "source_csv": str(aggregate.source_csv),
                        "label_column": aggregate.label_column,
                        "keyword": keyword,
                        "aspect": display_keyword(keyword),
                        "party": party,
                        "favor_n": stance_n(counts, "favor"),
                        "against_n": stance_n(counts, "against"),
                        "neutral_n": stance_n(counts, "neutral"),
                        "total_n": total_n(counts),
                        "favor_pct": pct_map["favor"],
                        "against_pct": pct_map["against"],
                        "neutral_pct": pct_map["neutral"],
                    }
                )
    write_csv(path, rows, fieldnames)


def distribution_csv_fieldnames() -> list[str]:
    fields = ["aspect", "keyword"]
    for party in VALID_PARTIES:
        prefix = party.replace(" ", "_")
        for stance in DISPLAY_STANCES:
            fields.append(f"{prefix}_{stance}_pct")
        fields.append(f"{prefix}_n")
    fields.append("total")
    return fields


def write_distribution_csv(path: Path, aggregate: AggregateResult, keywords: list[str]) -> None:
    rows = distribution_rows(aggregate, keywords)
    trimmed_rows = []
    for row in rows:
        out = {"aspect": row["aspect"], "keyword": row["keyword"]}
        for party in VALID_PARTIES:
            prefix = party.replace(" ", "_")
            for stance in DISPLAY_STANCES:
                out[f"{prefix}_{stance}_pct"] = row[f"{prefix}_{stance}_pct"]
            out[f"{prefix}_n"] = row[f"{prefix}_n"]
        out["total"] = row["total"]
        trimmed_rows.append(out)
    write_csv(path, trimmed_rows, distribution_csv_fieldnames())


def write_comparison_csv(path: Path, normal: AggregateResult, temporal: AggregateResult, keywords: list[str]) -> None:
    fieldnames = ["aspect", "keyword"]
    for party in VALID_PARTIES:
        prefix = party.replace(" ", "_")
        for stance in DISPLAY_STANCES:
            fieldnames.extend(
                [
                    f"{prefix}_normal_{stance}_pct",
                    f"{prefix}_temporal_{stance}_pct",
                    f"{prefix}_delta_{stance}_pp",
                ]
            )
        fieldnames.extend([f"{prefix}_normal_n", f"{prefix}_temporal_n"])
    fieldnames.extend(["normal_total", "temporal_total"])

    rows: list[dict[str, object]] = []
    for keyword in keywords:
        row: dict[str, object] = {
            "aspect": display_keyword(keyword),
            "keyword": keyword,
        }
        for party in VALID_PARTIES:
            prefix = party.replace(" ", "_")
            normal_counts = party_counter(normal.counts, keyword, party)
            temporal_counts = party_counter(temporal.counts, keyword, party)
            normal_pct_map = rounded_pct_map(normal_counts)
            temporal_pct_map = rounded_pct_map(temporal_counts)
            for stance in DISPLAY_STANCES:
                normal_pct = normal_pct_map[stance]
                temporal_pct = temporal_pct_map[stance]
                row[f"{prefix}_normal_{stance}_pct"] = normal_pct
                row[f"{prefix}_temporal_{stance}_pct"] = temporal_pct
                row[f"{prefix}_delta_{stance}_pp"] = round(temporal_pct - normal_pct, 1)
            row[f"{prefix}_normal_n"] = total_n(normal_counts)
            row[f"{prefix}_temporal_n"] = total_n(temporal_counts)
        row["normal_total"] = keyword_total(normal.counts, keyword)
        row["temporal_total"] = keyword_total(temporal.counts, keyword)
        rows.append(row)
    write_csv(path, rows, fieldnames)


def latex_distribution_table(
    aggregate: AggregateResult,
    keywords: list[str],
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        (
            r"\multirow{2}{*}{aspect} & "
            r"\multicolumn{4}{c}{Pro-Ruling} & "
            r"\multicolumn{4}{c}{Pro-Opposition} & "
            r"\multirow{2}{*}{Total} \\"
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& Fav\% & Ag\% & Neu\% & N & Fav\% & Ag\% & Neu\% & N & \\",
        r"\midrule",
    ]

    for keyword in keywords:
        cells = [latex_escape(display_keyword(keyword))]
        for party in VALID_PARTIES:
            counts = party_counter(aggregate.counts, keyword, party)
            pct_map = rounded_pct_map(counts)
            cells.extend(
                [
                    fmt_pct(pct_map["favor"]),
                    fmt_pct(pct_map["against"]),
                    fmt_pct(pct_map["neutral"]),
                    fmt_int(total_n(counts)),
                ]
            )
        cells.append(fmt_int(keyword_total(aggregate.counts, keyword)))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def latex_side_by_side_table(normal: AggregateResult, temporal: AggregateResult, keywords: list[str]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrrrrrrrrrrrrrrr}",
        r"\toprule",
        (
            r"\multirow{2}{*}{aspect} & "
            r"\multicolumn{4}{c}{Normal: Pro-Ruling} & "
            r"\multicolumn{4}{c}{Temporal: Pro-Ruling} & "
            r"\multicolumn{4}{c}{Normal: Pro-Opposition} & "
            r"\multicolumn{4}{c}{Temporal: Pro-Opposition} & "
            r"\multirow{2}{*}{Total} \\"
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}",
        (
            r"& Fav\% & Ag\% & Neu\% & N "
            r"& Fav\% & Ag\% & Neu\% & N "
            r"& Fav\% & Ag\% & Neu\% & N "
            r"& Fav\% & Ag\% & Neu\% & N & \\"
        ),
        r"\midrule",
    ]

    for keyword in keywords:
        cells = [latex_escape(display_keyword(keyword))]
        for party in VALID_PARTIES:
            normal_counts = party_counter(normal.counts, keyword, party)
            temporal_counts = party_counter(temporal.counts, keyword, party)
            for counts, pct_map in (
                (normal_counts, rounded_pct_map(normal_counts)),
                (temporal_counts, rounded_pct_map(temporal_counts)),
            ):
                cells.extend(
                    [
                        fmt_pct(pct_map["favor"]),
                        fmt_pct(pct_map["against"]),
                        fmt_pct(pct_map["neutral"]),
                        fmt_int(total_n(counts)),
                    ]
                )
        cells.append(fmt_int(keyword_total(normal.counts, keyword)))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            (
                r"\caption{Keyword-specific stance distribution comparison on the shared 2022--2023 rows. "
                r"Normal uses \texttt{fewshot\_label\_for\_against}; temporal uses \texttt{lora\_stance}.}"
            ),
            r"\label{tab:keyword-stance-normal-vs-temporal}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_latex_files(normal_full: AggregateResult, normal_matched: AggregateResult, temporal: AggregateResult) -> None:
    full_keywords = keyword_order(normal_full)
    matched_keywords = keyword_order(normal_matched, temporal)

    (BASE_DIR / "normal_full_keyword_stance_table.tex").write_text(
        latex_distribution_table(
            normal_full,
            full_keywords,
            "Normal stance distribution by aspect for the full 2020--2023 FINAL_STANCE dataset.",
            "tab:normal-full-keyword-stance",
        ),
        encoding="utf-8",
    )
    (BASE_DIR / "normal_matched_2022_2023_keyword_stance_table.tex").write_text(
        latex_distribution_table(
            normal_matched,
            matched_keywords,
            "Normal stance distribution by aspect on the matched 2022--2023 temporal split.",
            "tab:normal-matched-keyword-stance",
        ),
        encoding="utf-8",
    )
    (BASE_DIR / "temporal_lora_2022_2023_keyword_stance_table.tex").write_text(
        latex_distribution_table(
            temporal,
            matched_keywords,
            "Keyword temporal LoRA stance distribution by aspect on the 2022--2023 temporal split.",
            "tab:temporal-lora-keyword-stance",
        ),
        encoding="utf-8",
    )
    (BASE_DIR / "joint_normal_vs_temporal_2022_2023_table.tex").write_text(
        latex_side_by_side_table(normal_matched, temporal, matched_keywords),
        encoding="utf-8",
    )

    all_tables = "\n".join(
        [
            "% Required packages: booktabs, multirow, adjustbox",
            "% Main apples-to-apples comparison table:",
            latex_side_by_side_table(normal_matched, temporal, matched_keywords),
            "% Reference table: full normal stance distribution.",
            latex_distribution_table(
                normal_full,
                full_keywords,
                "Normal stance distribution by aspect for the full 2020--2023 FINAL_STANCE dataset.",
                "tab:normal-full-keyword-stance",
            ),
            "% Reference table: temporal stance distribution.",
            latex_distribution_table(
                temporal,
                matched_keywords,
                "Keyword temporal LoRA stance distribution by aspect on the 2022--2023 temporal split.",
                "tab:temporal-lora-keyword-stance",
            ),
        ]
    )
    (BASE_DIR / "all_keyword_stance_tables.tex").write_text(all_tables, encoding="utf-8")


def write_readme(normal_full: AggregateResult, normal_matched: AggregateResult, temporal: AggregateResult) -> None:
    def aggregate_summary(aggregate: AggregateResult) -> str:
        return (
            f"- {aggregate.name}: {aggregate.rows_used:,} usable rows from "
            f"{aggregate.rows_read:,} read rows; label column `{aggregate.label_column}`."
        )

    readme = "\n".join(
        [
            "# Keyword Stance Experiment Comparison",
            "",
            "This folder contains keyword-specific stance tables for the normal stance experiment and the keyword temporal stance experiment.",
            "",
            "## Main comparison",
            "",
            "The main comparison uses `TEMPORAL_SPLIT/split_2022_2023_lora_labeled.csv` because it contains both:",
            "",
            "- normal stance in `fewshot_label_for_against`",
            "- keyword temporal stance in `lora_stance`",
            "",
            "That makes `joint_normal_vs_temporal_2022_2023_table.tex` and `joint_experiment_comparison_2022_2023.csv` apples-to-apples comparisons on the exact same 2022-2023 tweet-keyword rows.",
            "",
            "## Generated files",
            "",
            "- `joint_normal_vs_temporal_2022_2023_table.tex`: side-by-side LaTeX table for both experiments.",
            "- `joint_experiment_comparison_2022_2023.csv`: wide CSV with normal %, temporal %, and temporal-minus-normal percentage-point deltas.",
            "- `full_normal_vs_temporal_percentage_comparison.csv`: non-matched percentage comparison using all normal labels and the temporal-label subset.",
            "- `normal_full_keyword_stance_table.tex`: reference LaTeX table for the full `FINAL_STANCE` normal labels.",
            "- `normal_matched_2022_2023_keyword_stance_table.tex`: normal-label LaTeX table on the shared temporal split rows.",
            "- `temporal_lora_2022_2023_keyword_stance_table.tex`: temporal-label LaTeX table on the shared temporal split rows.",
            "- `all_keyword_stance_tables.tex`: all LaTeX table snippets in one file.",
            "- `stance_distribution_long.csv`: tidy long-format CSV for all generated experiment scopes.",
            "",
            "## LaTeX packages",
            "",
            "Use these packages in your paper preamble:",
            "",
            "```tex",
            "\\usepackage{booktabs}",
            "\\usepackage{multirow}",
            "\\usepackage{adjustbox}",
            "```",
            "",
            "## Run",
            "",
            "```bash",
            "python3 build_keyword_stance_comparison.py",
            "```",
            "",
            "## Row accounting",
            "",
            aggregate_summary(normal_full),
            aggregate_summary(normal_matched),
            aggregate_summary(temporal),
            "",
        ]
    )
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    print("Building keyword stance comparison outputs...")
    print(f"Output folder: {BASE_DIR}")

    normal_full = build_aggregate(
        "normal_full_2020_2023",
        FINAL_CSV,
        "fewshot_label_for_against",
    )
    normal_matched = build_aggregate(
        "normal_matched_2022_2023",
        TEMPORAL_LABELED_CSV,
        "fewshot_label_for_against",
    )
    temporal = build_aggregate(
        "keyword_temporal_lora_2022_2023",
        TEMPORAL_LABELED_CSV,
        "lora_stance",
    )

    full_keywords = keyword_order(normal_full)
    matched_keywords = keyword_order(normal_matched, temporal)

    write_distribution_csv(
        BASE_DIR / "normal_full_keyword_party_stance_table.csv",
        normal_full,
        full_keywords,
    )
    write_distribution_csv(
        BASE_DIR / "normal_matched_2022_2023_keyword_party_stance_table.csv",
        normal_matched,
        matched_keywords,
    )
    write_distribution_csv(
        BASE_DIR / "temporal_lora_2022_2023_keyword_party_stance_table.csv",
        temporal,
        matched_keywords,
    )
    write_comparison_csv(
        BASE_DIR / "joint_experiment_comparison_2022_2023.csv",
        normal_matched,
        temporal,
        matched_keywords,
    )
    write_comparison_csv(
        BASE_DIR / "full_normal_vs_temporal_percentage_comparison.csv",
        normal_full,
        temporal,
        keyword_order(normal_full, temporal),
    )
    write_long_csv(
        BASE_DIR / "stance_distribution_long.csv",
        [normal_full, normal_matched, temporal],
        full_keywords,
    )
    write_latex_files(normal_full, normal_matched, temporal)
    write_readme(normal_full, normal_matched, temporal)

    for aggregate in (normal_full, normal_matched, temporal):
        print(
            f"{aggregate.name}: {aggregate.rows_used:,} usable rows "
            f"from {aggregate.rows_read:,} read rows"
        )
        if aggregate.skipped_keyword or aggregate.skipped_party or aggregate.skipped_stance:
            print(
                "  skipped "
                f"keyword={aggregate.skipped_keyword:,}, "
                f"party={aggregate.skipped_party:,}, "
                f"stance={aggregate.skipped_stance:,}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
