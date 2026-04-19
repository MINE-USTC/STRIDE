# Prompts shipped with STRIDE

These files match the **default** CLI flags in `pipeline` / `meta_planer` / `supervisor` / `fallback_qa`:

| File | Role | Default flag |
|------|------|----------------|
| `meta_plan/meta_plan.txt` | Meta-Planner system prompt | `--prompt_file meta_plan` |
| `supervisor/default.txt` | Supervisor system prompt | `--s_prompt_file default` |
| `extractor/default.txt` | Extractor system prompt | `--e_prompt_file default` |
| `reasoner/default.txt` | Reasoner system prompt (CoT JSON: `analysis` + `answer`) | `--r_prompt_file default` |
| `fallback_qa.txt` | Fallback QA user template | (fixed path in `fallback_qa.py`) |

**Reasoner output:** `supervisor.py` parses the model reply as JSON and reads the **`answer`** field (supports a fenced Markdown code block or a raw `{...}` object).

You can point `--s_prompt_file`, `--e_prompt_file`, or `--r_prompt_file` at another **stem** if you add `prompt/<role>/<stem>.txt` yourself.
