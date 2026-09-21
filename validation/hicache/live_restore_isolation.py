"""Bounded concurrent conversation-isolation probe; retains per-request cache tiers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5331")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def post(path, body):
        request = urllib.request.Request(args.base_url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    # Long shared prefix, then divergent records: accidental sibling reuse is detectable.
    prefix = "This is an isolated ledger. Only the final AUTHORITATIVE record answers the question.\n"
    prefix += "".join(f"Background shipment {i:04d}: ordinary freight, depot east, amount {i % 191}.\n" for i in range(640))
    fixtures = []
    for i in range(16):
        expected = f"ALDER-{4100+i}|BIRCH-{5200+i}|CEDAR-{6300+i}"
        text = prefix + f"\nAUTHORITATIVE record for this conversation: {expected}\nReturn exactly the three codes in that record, separated by |, with nothing else."
        ids = post("/v1/tokenize", {"model": args.model, "messages": [{"role": "user", "content": text}], "reasoning_effort": "none"})["tokens"]
        fixtures.append({"id": i, "expected": expected, "input_ids": ids})
    (args.output / "fixtures.json").write_text(json.dumps(fixtures))
    results = []
    start = time.monotonic()
    for wave in range(3):
        order = list(fixtures)
        random.Random(730 + wave).shuffle(order)

        def check(item):
            began = time.monotonic()
            record = {"wave": wave, "id": item["id"], "expected": item["expected"]}
            try:
                response = post("/generate", {"input_ids": item["input_ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 64}, "stream": False})
                record["response"] = response
                record["pass"] = response["text"].strip() == item["expected"] and response["meta_info"]["finish_reason"]["type"] == "stop"
            except Exception as error:
                record.update({"pass": False, "error": str(error)})
            record["seconds"] = time.monotonic() - began
            (args.output / f"wave-{wave}-case-{item['id']:02d}.json").write_text(json.dumps(record, indent=2))
            return record

        with ThreadPoolExecutor(max_workers=8) as executor:
            wave_results = list(executor.map(check, order))
        results.extend(wave_results)
        print(json.dumps({"wave": wave, "passed": sum(row["pass"] for row in wave_results), "total": len(wave_results)}), flush=True)
    summary = {"passed": sum(row["pass"] for row in results), "total": len(results), "seconds": time.monotonic() - start,
               "input_tokens": [len(item["input_ids"]) for item in fixtures],
               "storage_hit_requests": sum((row.get("response", {}).get("meta_info", {}).get("cached_tokens_details") or {}).get("storage", 0) > 0 for row in results)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
