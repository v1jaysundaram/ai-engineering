# jev: TypeSafe SDK experiments

Small experiments with the [TypeSafe SDK](https://pypi.org/project/typesafe-sdk/) for typed, structured judgements over text.

| Script | What it does |
|---|---|
| `jev.py` | Classifies a support ticket in one call: department (`Choice`), frustration (`Score`), urgency (`Noul`) |
| `jev_speed.py` | Times one TypeSafe yes/no judgement against the same question asked to `gpt-4o` |

## Run

Needs `TYPESAFE_API_KEY` (and `OPENAI_API_KEY` for the speed test) in the repo-root `.env`.

```bash
uv pip install -r requirements.txt
python jev.py
python jev_speed.py
```
