# Benchmark Item Schema

One JSON object per line in `benchmark.jsonl`. This document is the
authoritative per-item field specification. The loader in
`src/s1s2/benchmark/loader.py` mirrors it, and the validator in
`src/s1s2/benchmark/validate.py` enforces it.

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | yes | Unique item identifier. Snake-case. Paraphrases use `<id>__pN` suffixes but each line is a distinct record. |
| `category` | str | yes | One of `crt`, `base_rate`, `syllogism`, `anchoring`, `framing`, `conjunction`, `arithmetic`. Must match `s1s2.utils.types.TaskCategory`. |
| `subcategory` | str | yes | Free-form finer classification (e.g. `ratio`, `belief_bias_valid_unbelievable`, `linda_isomorph`). |
| `conflict` | bool | yes | `true` if the item contains a System-1 lure (dominant incorrect response). `false` for the matched control. |
| `difficulty` | int | yes | Integer 1-5 (1 = easiest, 5 = hardest). Conflict and control items in a matched pair should share the same difficulty. |
| `prompt` | str | yes | The user-facing question. Plain text, no chat template wrapping (the extract workstream handles that). |
| `system_prompt` | str or null | yes | Optional system prompt override. Usually `null`. |
| `correct_answer` | str | yes | Canonical correct answer string. |
| `lure_answer` | str | yes | The System-1 lure. Empty string `""` if `conflict == false`. Must be distinct from `correct_answer` whenever non-empty. |
| `answer_pattern` | str | yes | Python regex that matches `correct_answer` in a model response. Anchored with word boundaries where possible. |
| `lure_pattern` | str | yes | Python regex that matches `lure_answer`. Empty string `""` iff `lure_answer` is empty. |
| `matched_pair_id` | str | yes | Shared identifier linking a conflict item to its no-conflict control. Both items carry the same `matched_pair_id`. Paraphrases of the same base item share the same `matched_pair_id`. |
| `source` | str | yes | Provenance: `novel`, `hagendorff_2023`, `template`, `adapted`. |
| `provenance_note` | str | yes | Short note describing structural inspiration, the isomorph design, and any published source being paraphrased. |
| `paraphrases` | list[str] | yes | Alternative phrasings of `prompt`. Used by `templates.expand_paraphrases()` to generate sibling records. Empty list `[]` allowed. The primary record's `prompt` is always ALSO the first entry conceptually, but the `paraphrases` list only contains the *alternative* surface forms. |

## Matching rules

- Every `conflict: true` item MUST have at least one `conflict: false` sibling with the same `matched_pair_id`.
- Paraphrase records share `matched_pair_id` with their primary item.
- A `matched_pair_id` can cover multiple paraphrase records, but must contain exactly one conflict / control pair of "primaries."

## Regex conventions

- Use `\b` word boundaries for numeric answers: e.g. `\b5\b`.
- For multiple-choice answers, accept both letter and word forms where reasonable: e.g. `\b(?:A|Feinman|feminist bank teller)\b`.
- Keep the pattern as narrow as possible so the extractor's scoring doesn't accidentally match intermediate arithmetic.

## Example

```json
{
  "id": "crt_widget_gadget_001",
  "category": "crt",
  "subcategory": "ratio",
  "conflict": true,
  "difficulty": 2,
  "prompt": "A widget and a gadget cost $1.10 in total. The widget costs $1.00 more than the gadget. How much does the gadget cost, in cents?",
  "system_prompt": null,
  "correct_answer": "5",
  "lure_answer": "10",
  "answer_pattern": "\\b5\\b",
  "lure_pattern": "\\b10\\b",
  "matched_pair_id": "crt_widget_gadget_001_pair",
  "source": "novel",
  "provenance_note": "Novel isomorph of Frederick (2005) bat-and-ball structure; widget/gadget surface elements.",
  "paraphrases": [
    "Together, a widget and a gadget cost $1.10. The widget is $1.00 more expensive than the gadget. In cents, how much does the gadget cost?"
  ]
}
```
