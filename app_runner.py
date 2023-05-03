from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image
import utils
import datetime
import time
import psutil
import random


start_time = time.time()
is_colab = utils.is_google_colab()
state = None
current_steps = 25

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

models = [
     Model("Arcane", "nitrosocke/Arcane-Diffusion", "arcane style "),
     Model("Dreamlike Diffusion 1.0", "dreamlike-art/dreamlike-diffusion-1.0", "dreamlikeart "),
     Model("Archer", "nitrosocke/archer-diffusion", "archer style "),
     Model("Anything V4", "andite/anything-v4.0", ""),
     Model("Modern Disney", "nitrosocke/mo-di-diffusion", "modern disney style "),
     Model("Classic Disney", "nitrosocke/classic-anim-diffusion", "classic disney style "),
     Model("Loving Vincent (Van Gogh)", "dallinmackay/Van-Gogh-diffusion", "lvngvncnt "),
     Model("Wavyfusion", "wavymulder/wavyfusion", "wa-vy style "),
     Model("Analog Diffusion", "wavymulder/Analog-Diffusion", "analog style "),
     Model("Redshift renderer (Cinema4D)", "nitrosocke/redshift-diffusion", "redshift style "),
     Model("Midjourney v4 style", "prompthero/midjourney-v4-diffusion", "mdjrny-v4 style "),
     Model("Waifu", "hakurei/waifu-diffusion"),
     Model("Cyberpunk Anime", "DGSpitzer/Cyberpunk-Anime-Diffusion", "dgs illustration style "),
     Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
     Model("TrinArt v2", "naclbit/trinart_stable_diffusion_v2"),
     Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
     Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
     Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy "),
     Model("Pokémon", "lambdalabs/sd-pokemon-diffusers"),
     Model("Pony Diffusion", "AstraliteHeart/pony-diffusion"),
     Model("Robo Diffusion", "nousr/robo-diffusion"),
     Model("Epic Diffusion", "johnslegers/epic-diffusion")
  ]

custom_model = None
if is_colab:
  models.insert(0, Model("Custom model"))
  custom_model = models[0]

last_mode = "txt2img"
current_model = models[1] if is_colab else models[0]
current_model_path = current_model.path

if is_colab:
  pipe = StableDiffusionPipeline.from_pretrained(
      current_model.path,
      torch_dtype=torch.float16,
      scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
      safety_checker=lambda images, clip_input: (images, False)
      )

else:
  pipe = StableDiffusionPipeline.from_pretrained(
      current_model.path,
      torch_dtype=torch.float16,
      scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
      )
    
if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def update_state(new_state):
  global state
  state = new_state

def update_state_info(old_state):
  if state and state != old_state:
    return gr.update(value=state)

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def on_model_change(model_name):
  
  prefix = "Enter prompt. \"" + next((m.prefix for m in models if m.name == model_name), None) + "\" is prefixed automatically" if model_name != models[0].name else "Don't forget to use the custom model prefix in the prompt!"

  return gr.update(visible = model_name == models[0].name), gr.update(placeholder=prefix)

def on_steps_change(steps):
  global current_steps
  current_steps = steps

def pipe_callback(step: int, timestep: int, latents: torch.FloatTensor):
    update_state(f"{step}/{current_steps} steps")#\nTime left, sec: {timestep/100:.0f}")

def inference(model_name, prompt, guidance, steps, n_images=1, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  update_state(" ")

  print(psutil.virtual_memory()) # print memory usage

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  # generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
  if seed == 0:
    seed = random.randint(0, 2147483647)

  generator = torch.Generator('cuda').manual_seed(seed)

  try:
    if img is not None:
      return img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator, seed), f"Done. Seed: {seed}"
    else:
      return txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed), f"Done. Seed: {seed}"
  except Exception as e:
    return None, error_str(e)

def txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed):

    print(f"{datetime.datetime.now()} txt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        update_state(f"Loading {current_model.name} text-to-image model...")

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          # pipe = pipe.to("cpu")
          # pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt  
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator,
      callback=pipe_callback)

    # update_state(f"Done. Seed: {seed}")
    
    return replace_nsfw_images(result)

def img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator, seed):

    print(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        update_state(f"Loading {current_model.name} image-to-image model...")

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          # pipe = pipe.to("cpu")
          # pipe = current_model.pipe_i2i
        
        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        num_images_per_prompt=n_images,
        image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        # width = width,
        # height = height,
        generator = generator,
        callback=pipe_callback)

    # update_state(f"Done. Seed: {seed}")
        
    return replace_nsfw_images(result)

def replace_nsfw_images(results):

    if is_colab:
      return results.images
      
    for i in range(len(results.images)):
      if results.nsfw_content_detected[i]:
        pass
    return results.images

