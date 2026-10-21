"""Paraphrase attack: responses are rewritten by a local instruction model with the paraphrasing prompt of dgMARK.

The request, PROMPT followed by a blank line and the response, is the single message of a conversation in the chat template of the
paraphrasing model (Qwen2.5-32B-Instruct in the paper). Decoding with vLLM in bfloat16: temperature 0.2, at most 256 new tokens,
sampling seed 0 for every request. The paraphrase is the first output, stripped of surrounding whitespace. Output rows are the input
rows with the paraphrase in the added field "attacked"; they are the --paraphrases input of scripts/attack_eval.py.

Example:
  python scripts/attack_paraphrase.py --rows runs/attack_ours/rows_shard*.jsonl --arm ours_d4 --out attack/paraphrase_ours.jsonl
"""
import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT = ("As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output "
          "contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do "
          "not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, "
          "organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases. Respond only with "
          "the paraphrased text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="JSONL files of responses (field 'completion'), e.g. rows_shard*.jsonl")
    ap.add_argument("--arm", help="paraphrase only the rows of this arm (default: every row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0, help="GB of the weights kept in CPU memory (for small GPUs)")
    args = ap.parse_args()

    rows = []
    for path in sorted(set(args.rows)):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if args.arm is None or r.get("arm") == args.arm:
                    rows.append(r)
    print(len(rows), "responses", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, disable_custom_all_reduce=True, cpu_offload_gb=args.cpu_offload_gb)
    sp = SamplingParams(temperature=0.2, max_tokens=256, seed=0)
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT + "\n\n" + r["completion"]}], tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, sp)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps(dict(r, attacked=o.outputs[0].text.strip()), ensure_ascii=False) + "\n")
    print("written", args.out, flush=True)
    os._exit(0)         # vLLM worker processes can keep a normal interpreter exit from returning


if __name__ == "__main__":
    main()
