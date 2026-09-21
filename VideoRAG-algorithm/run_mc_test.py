import os, sys, json, time, functools, warnings, logging, multiprocessing, string

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

os.environ["OPENAI_API_KEY"] = open(os.path.expanduser("~/.config/openai/api_key")).read().strip()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
# torch cuDNN disabled to avoid a conv3d engine conflict with faster-whisper's cuDNN
torch.backends.cudnn.enabled = False
import tiktoken
from videorag.videorag import VideoRAG, QueryParam
from videorag._llm import openai_4o_mini_config

VIDEO_EMB_BATCH = 2  # cuDNN-off native conv3d handles batch fine

ENC = tiktoken.encoding_for_model("gpt-4o")
PICK = json.load(open(sys.argv[1]))
VIDEO = sys.argv[2]
WORKDIR = sys.argv[3]
OUT = sys.argv[4]
PRICE_IN, PRICE_OUT = 0.15, 0.60

stats = {"calls": 0, "in_tok": 0, "out_tok": 0}

def track(func):
    @functools.wraps(func)
    async def inner(prompt, *a, **k):
        stats["calls"] += 1
        stats["in_tok"] += len(ENC.encode(prompt if isinstance(prompt, str) else str(prompt)))
        if k.get("system_prompt"):
            stats["in_tok"] += len(ENC.encode(k["system_prompt"]))
        for m in k.get("history_messages", []) or []:
            stats["in_tok"] += len(ENC.encode(m.get("content", "")))
        r = await func(prompt, *a, **k)
        stats["out_tok"] += len(ENC.encode(r if isinstance(r, str) else str(r)))
        return r
    return inner

def snap(): return dict(stats)
def diff(a, b):
    d = {k: b[k]-a[k] for k in stats}
    d["est_cost_usd"] = round(d["in_tok"]/1e6*PRICE_IN + d["out_tok"]/1e6*PRICE_OUT, 5)
    return d


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    LETTERS = string.ascii_uppercase
    gt_label = LETTERS[PICK["correct_choice"]]
    results = {"video": PICK, "video_path": VIDEO, "gt_label": gt_label}

    # ---------------- CONSTRUCTION ----------------
    t0 = time.time()
    vr = VideoRAG(llm=openai_4o_mini_config, working_dir=WORKDIR, video_embedding_batch_num=VIDEO_EMB_BATCH)
    vr.llm.best_model_func = track(vr.llm.best_model_func)
    vr.llm.cheap_model_func = track(vr.llm.cheap_model_func)
    s0 = snap()
    vr.insert_video(video_path_list=[VIDEO])
    construct_sec = time.time() - t0
    construct_llm = diff(s0, snap())

    vname = os.path.basename(VIDEO).split(".")[0]
    segs = vr.video_segments._data.get(vname, {})
    results["num_segments"] = len(segs)
    # keep a few sample captions
    sample = {}
    for i in list(segs.keys())[:6]:
        sample[i] = {"time": segs[i]["time"], "content": segs[i]["content"][:600]}
    results["segment_samples"] = sample
    G = vr.chunk_entity_relation_graph._graph
    nodes = [{"name": n.strip('"'), "type": d.get("entity_type","").strip('"'),
              "deg": G.degree(n)} for n, d in G.nodes(data=True)]
    edges = [{"src": u.strip('"'), "tgt": v.strip('"'), "weight": d.get("weight",0)}
             for u, v, d in G.edges(data=True)]
    results["graph"] = {"num_nodes": G.number_of_nodes(), "num_edges": G.number_of_edges(),
                        "nodes": nodes, "edges": edges}
    results["num_chunks"] = len(vr.text_chunks._data)
    results["construct_sec"] = round(construct_sec, 1)
    results["construct_llm"] = construct_llm
    print(f"\n[BUILD DONE] {construct_sec:.1f}s | segs={len(segs)} chunks={len(vr.text_chunks._data)} "
          f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} | LLM {construct_llm}", flush=True)

    # ---------------- GENERATION (MULTIPLE CHOICE) ----------------
    t1 = time.time()
    vr2 = VideoRAG(llm=openai_4o_mini_config, working_dir=WORKDIR, video_embedding_batch_num=VIDEO_EMB_BATCH)
    vr2.llm.best_model_func = track(vr2.llm.best_model_func)
    vr2.llm.cheap_model_func = track(vr2.llm.cheap_model_func)
    vr2.load_caption_model(debug=False)

    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(PICK["candidates"]))
    query_text = f"Question: {PICK['question']}\nOptions:\n{opts}\n\nSelect the single correct option."
    param = QueryParam(mode="videorag_multiple_choice")
    s1 = snap()
    answer = vr2.query(query=query_text, param=param)
    query_sec = time.time() - t1
    query_llm = diff(s1, snap())

    # answer is a dict {"Answer","Explanation"}
    pred = answer.get("Answer", "") if isinstance(answer, dict) else str(answer)
    pred_letter = ""
    for ch in str(pred).upper():
        if ch in LETTERS[:len(PICK["candidates"])]:
            pred_letter = ch; break
    correct = (pred_letter == gt_label)

    results.update({
        "query_text": query_text,
        "answer_raw": answer,
        "pred_label": pred_letter or str(pred),
        "correct": correct,
        "query_sec": round(query_sec, 1),
        "query_llm": query_llm,
        "total_llm": diff(s0, snap()),
    })
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[MC DONE] {query_sec:.1f}s | pred={pred_letter} gt={gt_label} correct={correct} | LLM {query_llm}", flush=True)
    print(f"[TOTAL LLM] {results['total_llm']}", flush=True)
    print(f"\n===== ANSWER =====\n{json.dumps(answer, indent=2) if isinstance(answer, dict) else answer}", flush=True)
    print(f"[Results saved] {OUT}", flush=True)
