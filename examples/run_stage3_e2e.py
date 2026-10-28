"""Stage 3 End-to-End Showcase Runner."""

import json
import time
from pathlib import Path

from matchquant.adapter import execute_adapter_request

market_input = "D:/Codex/matchquant_runs/odds_snapshots/2026-10-31_Chelsea_Man_United/market_input.json"

req = {
    "action": "analyze",
    "home": "切尔西",
    "away": "曼联",
    "date": "2026-10-31",
    "season": "2026-27",
    "as_of": "2026-10-28",
    "market_input_path": market_input,
    "allow_stale_fallback": True,
    "max_data_age_days": 200,
}

t0 = time.perf_counter()
res = execute_adapter_request(req)
t1 = time.perf_counter()

print("=== 1. ANALYZE RESULT ===")
print("Status:", res.get("status"))
print("Run ID:", res.get("run_id"))
print("Run Dir:", res.get("run_dir"))
print(f"Execution Time: {t1 - t0:.2f}s")

run_id = res["run_id"]
artifacts = res["artifacts"]
print("\nGenerated Artifacts:")
for k, v in artifacts.items():
    print(f"  {k}: {v}")

# Verify Section 4 in agent_report.md
md_content = Path(artifacts["agent_markdown"]).read_text(encoding="utf-8")
html_content = Path(artifacts["agent_html"]).read_text(encoding="utf-8")

# Extract Section 4 from Markdown
sec4_start = md_content.find("## 四、 市场参照与让球对位 `[E5]`")
sec5_start = md_content.find("## 五、 数据版本与时序截断回执 `[E6]`")
sec4_text = md_content[sec4_start:sec5_start].strip()

# Three follow-up questions
qa_log = []

# Follow-up 1
q1 = "盘口态度与模型差异怎么看？有背离吗？"
t_f1_0 = time.perf_counter()
f1 = execute_adapter_request({"action": "followup", "run_id": run_id, "question": q1})
t_f1_1 = time.perf_counter()
d1 = t_f1_1 - t_f1_0
qa_log.append({"round": 1, "question": q1, "duration_sec": round(d1, 4), "response": f1})

# Follow-up 2
q2 = "多时点变动走势和聪明钱趋势如何？让球盘怎么分析？"
t_f2_0 = time.perf_counter()
f2 = execute_adapter_request({"action": "followup", "run_id": run_id, "question": q2})
t_f2_1 = time.perf_counter()
d2 = t_f2_1 - t_f2_0
qa_log.append({"round": 2, "question": q2, "duration_sec": round(d2, 4), "response": f2})

# Follow-up 3
q3 = "给我一个精简的核心结论与量化建议"
t_f3_0 = time.perf_counter()
f3 = execute_adapter_request({"action": "followup", "run_id": run_id, "question": q3})
t_f3_1 = time.perf_counter()
d3 = t_f3_1 - t_f3_0
qa_log.append({"round": 3, "question": q3, "duration_sec": round(d3, 4), "response": f3})

output_transcript = {
    "match": "Chelsea vs Man United (2026-10-31)",
    "run_id": run_id,
    "artifacts": artifacts,
    "markdown_section_4": sec4_text,
    "qa_rounds": qa_log,
}

transcript_path = Path(artifacts["agent_markdown"]).parent / "stage3_e2e_transcript.json"
transcript_path.write_text(json.dumps(output_transcript, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSaved transcript to:", transcript_path)

print("\n=== 2. SECTION 4 [E5] FROM MARKDOWN ===")
print(sec4_text)

print("\n=== 3. THREE FOLLOW-UP QA RESULTS ===")
for r in qa_log:
    resp = r["response"]
    rnd = r["round"]
    qst = r["question"]
    dur = r["duration_sec"]
    print(f"\n--- [Round {rnd}] Question: {qst}")
    print(f"Execution Time: {dur}s (Fast <0.1s: {dur < 0.1})")
    print(f"Recomputed: {resp.get('recomputed')}")
    print(f"Referenced Evidences: {resp.get('referenced_evidence_ids')}")
    print(f"Answer:\n{resp.get('answer')}")
