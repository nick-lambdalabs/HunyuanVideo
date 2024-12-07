import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from loguru import logger

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid

VIDEO_FPS = 24
MAX_VIDEO_SEC = 20
DEFAULT_VIDEO_SEC = 5

args = parse_args()
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

# Load models
hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

# Get the updated args
args = hunyuan_video_sampler.args


def generate_video(prompt, video_length, size, infer_steps) -> str:
    width, height = map(int, size.split("x"))
    outputs = hunyuan_video_sampler.predict(
        prompt=prompt,
        height=height,
        width=width,
        video_length=video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=1,  # todo: configurable
        flow_shift=args.flow_shift,
        batch_size=1,  # todo: configurable
        embedded_guidance_scale=args.embedded_cfg_scale,
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


with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Video Generator")
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Enter your prompt", placeholder="Type something..."
        )
        video_length_input = gr.Slider(  # video length values have to be 4n+1
            minimum=1,
            maximum=MAX_VIDEO_SEC * VIDEO_FPS + 1,
            step=4,
            label="Video Length",
            value=DEFAULT_VIDEO_SEC * VIDEO_FPS + 1,
        )
        size_input = gr.Dropdown(
            label="Size", choices=["1280x720", "640x360", "320x180"], value="640x360"
        )
        infer_steps_input = gr.Slider(
            label="Inference Steps", minimum=1, maximum=100, value=25
        )
        submit_btn = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video")

    submit_btn.click(
        generate_video,
        inputs=[
            prompt_input,
            video_length_input,
            size_input,
            infer_steps_input,
        ],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=1337)