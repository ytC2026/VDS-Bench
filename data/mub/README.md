# Supplemented MUB subset

This directory contains the exact 508 examples used by the MUB experiments:
30 development examples and 478 held-out test examples. The original annotation
files and fixed split are preserved from the supplied experiment copy.

| Path | Contents |
| --- | --- |
| `annotations/train.jsonl` | 30 development examples, paired D+/D- labels, and training-only option overrides |
| `annotations/test.jsonl` | 478 test examples with one explicit incorrect feedback target each |
| `splits/split.json` | Fixed `train_ids` and `test_ids` |
| `raw/images/` | Only the 508 images referenced by these annotations, at their original paths |
| `raw/annotations/` | Three referenced upstream snapshots used to recover original question text and trace provenance |

## What was supplemented

| Split | Original explicit incorrect targets | Supplemented incorrect targets | D+/D- development labels | Training-only option overrides |
| --- | ---: | ---: | ---: | ---: |
| Development (30) | 29 | 1 | 30 | 21 |
| Test (478) | 231 | 247 | 0 | 0 |

`misleading_instruction` records the target used in evaluation. A nonempty
`source.misleading_instruction` points to an upstream explicit target; a null
or absent reference identifies a supplemented target. Every target is different
from `true_option`.

Development rows additionally provide `dplus` and `dminus`. When present,
`training_option_edits` overrides one incorrect option only during paired
direction extraction. The original `options` and correct option are retained;
evaluation uses the original choices. The test set has no training-option edits.

`source.question` and `source.misleading_instruction` use `filename#index`
references. The three required snapshots are `all_combined_items.official.jsonl`,
`implicit_gpt9.official.jsonl`, and `implicit_gpt12.official.jsonl`. Their original
bytes and indices are retained. Despite the filenames, these upstream snapshots
contain JSON arrays with incomplete trailing content; the existing loader reads
complete entries. Every reference used by the 508 selected examples has been
verified to resolve. `implicit_gpt6` and unreferenced images are not needed by
this subset and are omitted.

The full original question text is recovered from these snapshots because it
can contain inline choices or prompt context not present in the compact
`question` field. Do not replace it with only the compact field when reproducing
the evaluation. All three model adapters use the same fixed annotations and
split. This directory contains dataset inputs, not saved model responses.
