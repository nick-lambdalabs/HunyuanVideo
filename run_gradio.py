import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from loguru import logger

from hyvideo.config import parse_args

VIDEO_FPS = 24
MAX_VIDEO_SEC = 20
DEFAULT_VIDEO_SEC = 5

args = parse_args()
args.prompt = None
args.flow_reverse = True
args.use_cpu_offload = True
print(args)
models_root_path = Path(args.model_base)
if not models_root_path.exists():
    raise ValueError(f"`models_root` not exists: {models_root_path}")

# Create save folder to save the samples
save_path = (
    args.save_path
    if args.save_path_suffix == ""
    else f"{args.save_path}_{args.save_path_suffix}"
)
if not os.path.exists(args.save_path):
    os.makedirs(save_path, exist_ok=True)

hunyuan_video_sampler = None


def maybe_load_model(progress: gr.Progress):
    from hyvideo.inference import HunyuanVideoSampler  # lazy import for faster startup

    global hunyuan_video_sampler
    global args

    if hunyuan_video_sampler is not None:
        return

    progress(0, "Loading model...")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args
    )

    # Get the updated args
    args = hunyuan_video_sampler.args


def generate_video(
    prompt,
    neg_prompt,
    video_length,
    size,
    infer_steps,
    seed,
    cfg_scale,
    embedded_cfg_scale,
    flow_shift,
    progress=gr.Progress(),
) -> str:
    from hyvideo.utils.file_utils import (
        save_videos_grid,  # lazy import for faster startup
    )

    print(f"Prompt: {prompt}")
    print(f"Video Length: {video_length}")
    print(f"Size: {size}")
    print(f"Inference Steps: {infer_steps}")

    maybe_load_model(progress)
    assert hunyuan_video_sampler is not None

    def progress_callback(
        pipeline, step_i, step_t, callback_kwargs_that_arent_really_kwargs
    ):
        progress((step_i, infer_steps), "Generating video...")
        return {}  # inference pipeline expects a dict return

    width, height = map(int, size.split("x"))
    global output_video
    output_video.width = width
    output_video.height = height

    outputs = hunyuan_video_sampler.predict(
        prompt=prompt,
        height=height,
        width=width,
        video_length=video_length,
        seed=seed,
        negative_prompt=neg_prompt,
        infer_steps=infer_steps,
        guidance_scale=cfg_scale,
        num_videos_per_prompt=1,
        flow_shift=flow_shift,
        batch_size=1,
        embedded_guidance_scale=embedded_cfg_scale,
        callback_on_step_end=progress_callback,
    )
    samples = outputs["samples"]

    # Save samples
    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        result_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','').replace(' ', '_')}.mp4"
        save_videos_grid(sample, result_path, fps=VIDEO_FPS)
        logger.info(f"Sample saved to: {result_path}")

        # TODO: handle multiple results
        return result_path
    return ""


with gr.Blocks() as demo:
    gr.Markdown("""
                # HunyuanVideo Text-to-Video Demo
                
                Video lengths are in frames. Videos are 24 FPS.

                The first time you click "Generate" it will take an extra long time as the model loads.

                Generation time is a function of video size, video length, and diffusion steps. These are
                multiplicative - i.e., `generation_time = O(video_size * video_length * diffusion_steps)`. 
                I recommend that you start with small values (e.g., 320x180, 29 frames, 25 steps) while you
                refine your prompt, then increase the values to generate a final high quality video. Generating
                a 10 second 1280x720 video with 50 diffusion steps takes like 30 min. Generating a 1 second
                320x180 video with 25 diffusion steps takes like 30 sec.
                """)
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Type something...",
            lines=3,
            max_lines=10,
        )
        neg_prompt_input = gr.Textbox(
            label="Negative prompt",
            placeholder="Type something...",
            lines=3,
            max_lines=10,
            value="aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        )
    with gr.Row():
        size_input = gr.Dropdown(
            label="Size",
            choices=[
                "1280x720",
                "640x360",
                "336x192",
                "320x180",
            ],
            value="336x192",
        )
        video_length_input = gr.Slider(  # video length values have to be 4n+1
            minimum=1,
            maximum=MAX_VIDEO_SEC * VIDEO_FPS + 1,
            step=4,
            label="Video Length",
            value=DEFAULT_VIDEO_SEC * VIDEO_FPS + 1,
        )
        infer_steps_input = gr.Slider(
            label="Diffusion Steps",
            minimum=1,
            maximum=200,
            step=1,
            value=25,
        )
        seed_input = gr.Number(label="Seed", value=0xC0FFEE)
    with gr.Row():
        cfg_scale_input = gr.Slider(
            label="Guidance Scale", minimum=0.0, maximum=20.0, value=1.0
        )
        embed_cfg_scale_input = gr.Slider(
            label="Embedded Guidance Scale", minimum=0.0, maximum=20.0, value=6.0
        )
        flow_shift_input = gr.Slider(
            label="Flow Shift", minimum=0.0, maximum=10.0, value=5.0
        )
        submit_btn = gr.Button("Generate Video")

    output_video = gr.Video(label="Generated Video", width=1280, height=720, live=True)

    submit_btn.click(
        generate_video,
        inputs=[
            prompt_input,
            neg_prompt_input,
            video_length_input,
            size_input,
            infer_steps_input,
            seed_input,
            cfg_scale_input,
            embed_cfg_scale_input,
            flow_shift_input,
        ],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
