import os
import time
import math
import shutil
import numpy as np
from tqdm import tqdm
from moviepy.video import fx as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
from .._utils import logger

def split_video(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
):  
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)
    
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    with VideoFileClip(video_path) as video:
    
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))
        # if the last segment is shorter than 5 seconds, we merged it to the last segment
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]
        
        for start in tqdm(start_times, desc=f"Spliting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False)
            frame_times += start
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)}
            
            # save audio
            audio_file_base_name = segment_index2name[f"{segment_index}"]
            audio_file = f'{audio_file_base_name}.{audio_output_format}'
            try:
                subaudio = subvideo.audio
                subaudio.write_audiofile(os.path.join(video_segment_cache_path, audio_file), codec='mp3', verbose=False, logger=None)
            except Exception as e:
                logger.warning(f"Warning: Failed to extract audio for video {video_name} ({start}-{end}). Probably due to lack of audio track.")

            segment_index += 1

    return segment_index2name, segment_times_info

def _write_video_segment(subvideo, final_path, retries=3):
    """ffmpeg's subprocess occasionally breaks its pipe under load (many
    concurrent runs competing for CPU/FDs); retry rather than leave a
    missing or truncated segment for the embedder to trip over."""
    last = None
    for attempt in range(retries):
        try:
            subvideo.write_videofile(final_path, codec='libx264', verbose=False, logger=None)
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                return
            last = RuntimeError(f"write_videofile produced no/empty file: {final_path}")
        except Exception as e:                     # noqa: BLE001
            last = e
        time.sleep(0.5 * (attempt + 1))
    raise last


def _pool_segment_clip(times, paths, start, end):
    """ImageBind clip built from pool JPEGs in [start, end); holds existing
    frames longer for a sparse pool rather than decoding new ones. None if
    the pool has nothing in this window."""
    got = [(t, p) for t, p in zip(times, paths) if start <= t < end]
    if not got:
        return None
    from moviepy.editor import ImageSequenceClip
    dur = max(end - start, 1e-3)
    n_min = max(10, math.ceil(dur / 1.5))
    if len(got) == 1:
        got = got * n_min
    elif len(got) < n_min:
        idx = [round(i * (len(got) - 1) / (n_min - 1)) for i in range(n_min)]
        got = [got[i] for i in idx]
    fps = max(len(got) / dur, 0.1)
    return ImageSequenceClip([p for _, p in got], fps=fps)


def saving_video_segments(
    video_name,
    video_path,
    working_dir,
    segment_index2name,
    segment_times_info,
    error_queue,
    video_output_format='mp4',
    pool_frames=None,
):
    # pool_frames: optional (times, jpeg_paths) for the whole video; when
    # given, ImageBind's clips come from pool JPEGs instead of the source,
    # so ImageBind sees the same pool as the captioner rather than the
    # full-resolution, full-fps source video.
    try:
        times, paths = pool_frames if pool_frames else (None, None)
        video = None
        video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
        for index in tqdm(segment_index2name, desc=f"Saving Video Segments {video_name}"):
            start, end = segment_times_info[index]["timestamp"][0], segment_times_info[index]["timestamp"][1]
            video_file = f'{segment_index2name[index]}.{video_output_format}'
            out_path = os.path.join(video_segment_cache_path, video_file)
            clip = _pool_segment_clip(times, paths, start, end) if times else None
            if clip is None:
                if video is None:
                    video = VideoFileClip(video_path)
                clip = video.subclip(start, end)
            _write_video_segment(clip, out_path)
        if video is not None:
            video.close()
    except Exception as e:
        error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
        raise RuntimeError