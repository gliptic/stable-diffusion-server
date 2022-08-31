import argparse, os, sys, glob, random, re, io
import queue
import torch
import numpy as np
import json
import hashlib
import copy
import msgpack
import asyncio
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from server.server import Server
from threading import Thread
import traceback
import qoi

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
device = "cuda"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--small_batch",
    action='store_true',
    help="Reduce inference time when generate a smaller batch of images",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

grid_count = len(os.listdir(outpath)) - 1
#seed_everything(opt.seed)

sd = load_model_from_config(f"{ckpt}")
li = []
lo = []
for key, value in sd.items():
    sp = key.split('.')
    if(sp[0]) == 'model':
        if('input_blocks' in sp):
            li.append(key)
        elif('middle_block' in sp):
            li.append(key)
        elif('time_embed' in sp):
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd['model1.' + key[6:]] = sd.pop(key)
for key in lo:
    sd['model2.' + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")
config.modelUNet.params.ddim_steps = opt.ddim_steps

if opt.small_batch:
    config.modelUNet.params.small_batch = True
else:
    config.modelUNet.params.small_batch = False

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
    
modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
    
modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()

if opt.precision == "autocast":
    model.half()
    modelCS.half()

batch_size = opt.n_samples

def move_to_cpu(tensor):
    mem = torch.cuda.memory_allocated()
    tensor.to("cpu")
    while torch.cuda.memory_allocated() >= mem:
        time.sleep(1)

serv = Server()
serv.start()

precision_scope = autocast if opt.precision=="autocast" else nullcontext
try:
    with torch.no_grad():

        while serv.wait_for_requests(): # len(serv.buckets) > 0:
            batch = serv.batch_requests(batch_size)
            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if serv.scale_for_batch(batch) != 1.0:
                    uc = modelCS.get_learned_conditioning(len(batch) * [""]) # Cache
                prompts = serv.prompts_for_batch(batch)
                
                c = modelCS.get_learned_conditioning(prompts)
                shape = serv.shape_for_batch(batch)
                move_to_cpu(modelCS)

                start_code = serv.startcodes_for_batch(batch).to(device)

                # x0 = original image
                # x_T = seed noise
                # img_callback(pred_x0, i) for each step
                samples_ddim = model.sample(S=serv.steps_for_batch(batch),
                                conditioning=c,
                                batch_size=len(batch),
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=serv.scale_for_batch(batch),
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                seed=None,
                                #img_callback=preview_img,
                                x_T=start_code)

                modelFS.to(device)
                print("saving images")

                for i in range(len(batch)):
                    job = batch[i]
                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')

                    image = x_sample.astype(np.uint8, order='C')

                    image_bytes = serv.save_image(job, image, outpath)
                    serv.send_response(job, image_bytes)


                move_to_cpu(modelFS)

                del samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated()/1e6)
except Exception as e:
    print('Exception!', e)
    traceback.print_exc()

serv.stop()

#print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))
print('Exiting')