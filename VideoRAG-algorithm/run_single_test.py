import os, sys, json, time, functools, warnings, logging, multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

os.environ["OPENAI_API_KEY"] = open(os.path.expanduser("~/.config/openai/api_key")).read().strip()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tiktoken
from videorag.videorag import VideoRAG, QueryParam
from videorag._llm import openai_4o_mini_config

ENC = tiktoken.encoding_for_model("gpt-4o")
PICK = json.load(open(sys.argv[1]))
VIDEO = sys.argv[2]
WORKDIR = sys.argv[3]
OUT = sys.argv[4]

# gpt-4o-mini pricing ($/1M tokens)
PRICE_IN, PRICE_OUT = 0.15, 0.60

stats = {"calls": 0, "in_tok": 0, "out_tok": 0}

def track(func):
    @functools.wraps(func)
    async def inner(prompt, *a, **k):
        stats["calls"] += 1
        stats["in_tok"] += len(ENC.encode(prompt if isinstance(prompt, str) else str(prompt)))
        for m in k.get("history_messages", []) or []:
            stats["in_tok"] += len(ENC.encode(m.get("content", "")))
        r = await func(prompt, *a, **k)
        stats["out_tok"] += len(ENC.encode(r if isinstance(r, str) else str(r)))
        return r
    return inner

def snapshot():
    return dict(stats)

def diff(a, b):
    d = {k: b[k] - a[k] for k in stats}
    d["est_cost_usd"] = round(d["in_tok"]/1e6*PRICE_IN + d["out_tok"]/1e6*PRICE_OUT, 5)
    return d


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    results = {"video": PICK, "video_path": VIDEO}

    # ---------------- CONSTRUCTION ----------------
    t0 = time.time()
    vr = VideoRAG(llm=openai_4o_mini_config, working_dir=WORKDIR)
    vr.llm.best_model_func = track(vr.llm.best_model_func)
    vr.llm.cheap_model_func = track(vr.llm.cheap_model_func)
    s_before = snapshot()
    vr.insert_video(video_path_list=[VIDEO])
    construct_sec = time.time() - t0
    construct_llm = diff(s_before, snapshot())

    # capture segment content + chunks + graph
    vname = os.path.basename(VIDEO).split(".")[0]
    segs = vr.video_segments._data.get(vname, {})
    results["segments"] = {i: {"time": segs[i]["time"], "content": segs[i]["content"]} for i in segs}
    G = vr.chunk_entity_relation_graph._graph
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"name": n, "type": d.get("entity_type", ""),
                      "description": (d.get("description", "") or "")[:400]})
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"src": u, "tgt": v, "weight": d.get("weight", 0),
                      "description": (d.get("description", "") or "")[:300]})
    results["graph"] = {"num_nodes": G.number_of_nodes(), "num_edges": G.number_of_edges(),
                        "nodes": nodes, "edges": edges}
    results["num_chunks"] = len(vr.text_chunks._data)
    results["construct_sec"] = round(construct_sec, 1)
    results["construct_llm"] = construct_llm
    print(f"\n[CONSTRUCTION DONE] {construct_sec:.1f}s | nodes={G.number_of_nodes()} edges={G.number_of_edges()} | LLM {construct_llm}")

    # ---------------- GENERATION (QUERY) ----------------
    t1 = time.time()
    vr2 = VideoRAG(llm=openai_4o_mini_config, working_dir=WORKDIR)
    vr2.llm.best_model_func = track(vr2.llm.best_model_func)
    vr2.llm.cheap_model_func = track(vr2.llm.cheap_model_func)
    vr2.load_caption_model(debug=False)
    q = PICK["question"]
    cand = "\n".join(f"({i}) {c}" for i, c in enumerate(PICK["candidates"]))
    query_text = f"{q}\nOptions:\n{cand}\nExplain which option is correct and why."
    param = QueryParam(mode="videorag")
    param.wo_reference = True
    s_before2 = snapshot()
    answer = vr2.query(query=query_text, param=param)
    query_sec = time.time() - t1
    query_llm = diff(s_before2, snapshot())

    results["query_text"] = query_text
    results["answer"] = answer
    results["correct_choice"] = PICK["correct_choice"]
    results["query_sec"] = round(query_sec, 1)
    results["query_llm"] = query_llm
    total = diff(s_before, snapshot())
    results["total_llm"] = total

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[QUERY DONE] {query_sec:.1f}s | LLM {query_llm}")
    print(f"[TOTAL LLM] {total}")
    print(f"\n===== ANSWER =====\n{answer}\n")
    print(f"[Ground-truth correct choice index] {PICK['correct_choice']}")
    print(f"[Results saved] {OUT}")
