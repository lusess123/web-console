import torch
from diffusers import DiffusionPipeline, LCMScheduler
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         variant="fp16",
                                         torch_dtype=torch.float16
                                         )
# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LoRAs
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("ming-yang/sdxl_chinese_ink_lora", adapter_name="Chinese Ink")

# Combine LoRAs
pipe.set_adapters(["lcm", "Chinese Ink"], adapter_weights=[1.0, 0.8])

prompts = ["Chinese Ink, mona lisa picture, 8k", "mona lisa, 8k"]
generator = torch.manual_seed(1)
images = [pipe(prompt, num_inference_steps=8, guidance_scale=1, generator=generator).images[0] for prompt in prompts]

fig, axs = plt.subplots(1, 2, figsize=(40, 20))

axs[0].imshow(images[0])
axs[0].axis('off')  # 不显示坐标轴

axs[1].imshow(images[1])
axs[1].axis('off')
plt.show()
