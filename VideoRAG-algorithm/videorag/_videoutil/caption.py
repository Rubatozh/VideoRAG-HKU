import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

# >>> BENCH SEAM BEGIN (multimodal-RAG) -- strip this block to revert
# Identity-defaulting hooks; with none installed this block changes nothing.
try:
    from .. import bench_hooks as _bench
except Exception:                       # pragma: no cover
    class _bench:                       # no-op fallback == upstream
        frames_for = staticmethod(lambda v, **k: None)
# <<< BENCH SEAM END
def encode_video(video, frame_times):
    # >>> BENCH SEAM BEGIN (multimodal-RAG) -- strip this block to revert
    _sub = _bench.frames_for(frame_times, video=video)
    if _sub is not None:
        return _sub
    # <<< BENCH SEAM END
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
def _pil_jpeg_b64(img, max_px=512):
    import base64
    import io
    img = img.convert('RGB')
    if max(img.size) > max_px:          # encode_video hands 1280x720; downscale like media.jpeg_b64
        img = img.copy()
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    # spawn re-imports this module fresh in the child, so a hook installed on
    # the parent's bench_hooks never arrives here -- env vars do.
    backend = os.environ.get('HKU_CAPTION_BACKEND', 'local')
    try:
        if backend == 'api':
            from openai import OpenAI
            client = OpenAI(base_url=os.environ.get('HKU_CAPTION_BASE_URL') or None,
                            api_key=os.environ.get('OPENAI_API_KEY', 'none'))
            caption_model = os.environ.get('HKU_CAPTION_MODEL', 'gpt-4o-mini')
            workers = max(1, int(os.environ.get('HKU_CAPTION_WORKERS', '1')))
        else:
            model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
            model.eval()
            workers = 1

        prepared = []
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Preparing Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                prepared.append((index, video_frames, query))

        def _one(item):
            index, video_frames, query = item
            if backend == 'api':
                content = [{"type": "text", "text": query}] + [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_pil_jpeg_b64(im)}"}}
                    for im in video_frames]
                resp = client.chat.completions.create(
                    model=caption_model, temperature=0, max_tokens=600,
                    messages=[{"role": "user", "content": content}])
                text = (resp.choices[0].message.content or "").strip()
            else:
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {"use_image_id": False, "max_slice_nums": 2}
                text = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
                torch.cuda.empty_cache()
            return index, text.replace("\n", "").replace("<|endoftext|>", "")

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for index, text in tqdm(ex.map(_one, prepared), total=len(prepared),
                                        desc=f"Captioning Video {video_name}"):
                    caption_result[index] = text
        else:
            for item in tqdm(prepared, desc=f"Captioning Video {video_name}"):
                index, text = _one(item)
                caption_result[index] = text
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    def _one(this_segment):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {"use_image_id": False, "max_slice_nums": 2}
        segment_caption = caption_model.chat(image=None, msgs=msgs, tokenizer=caption_tokenizer, **params)
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        if not isinstance(caption_tokenizer, type):     # local model only, not the served sentinel
            torch.cuda.empty_cache()
        return this_segment, f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"

    # >>> BENCH SEAM BEGIN (multimodal-RAG) -- strip this block to revert
    workers = max(1, int(os.environ.get('HKU_QUERY_CAPTION_WORKERS', '1')))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(tqdm(ex.map(_one, retrieved_segments), total=len(retrieved_segments),
                             desc='Captioning Segments for Given Query'))
    # <<< BENCH SEAM END
    return dict(_one(s) for s in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'))