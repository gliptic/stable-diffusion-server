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
    default=84,
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
    default=9,
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

def start_serv():
    print("Started server thread")
    asyncio.run(serv.run())

serv_thread = Thread(target=start_serv)
serv_thread.start()

#batches = []
# input_data = msgpack.packb({
#     'ty': 'request',
#     'request': {
#         'id': 0,
#         'prompt': opt.prompt,
#         'w': opt.W,
#         'h': opt.H,
#         'seed': opt.seed, 'count': opt.n_samples * opt.n_iter, 'scale': opt.scale, 'steps': opt.ddim_steps
#     }
# })

# serv.feed(input_data)

    #batch = []
    #for j in range(batch_size):
    #    batch.append({ 'prompt': opt.prompt, 'seed': (opt.seed + i * batch_size + j) & 0x7fffffff })
    #batches.append(batch)


precision_scope = autocast if opt.precision=="autocast" else nullcontext
try:
    with torch.no_grad():

        while serv.wait_for_requests(): # len(serv.buckets) > 0:
            time.sleep(1) # Allow any straggling requests to arrive
            batch = serv.batch_requests(batch_size)
            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if batch[0]['scale'] != 1.0:
                    uc = modelCS.get_learned_conditioning(len(batch) * [""]) # Cache
                prompts = [job['prompt'] for job in batch]
                
                c = modelCS.get_learned_conditioning(prompts)
                #shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                shape = [opt.C, batch[0]['h'] // opt.f, batch[0]['w'] // opt.f]
                move_to_cpu(modelCS)

                start_codes = []
                for job in batch:
                    generator = torch.Generator()
                    generator.manual_seed(job['seed'])
                    #start_codes.append(torch.randn([opt.C, opt.H // opt.f, opt.W // opt.f], generator=generator))
                    start_codes.append(torch.randn([opt.C, job['h'] // opt.f, job['w'] // opt.f], generator=generator))
                start_code = torch.stack(start_codes, 0).to(device)

                def preview_img(img, pred_x0, stepi):
                    if stepi > 0 and (stepi % 10) == 0:
                        for i in range(len(batch)):
                            job = batch[i]
                            x_samples_ddim = modelFS.decode_first_stage(img[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            qoi_bytes = qoi.encode(x_sample.astype(np.uint8, order='C'))

                            serv.send_response({
                                'ty': 'result',
                                'result': {
                                    'id': job['id'],
                                    'index': job['index'],
                                    'seed': job['seed'],
                                    'scale': job['scale'],
                                    'steps': job['steps'],
                                    'image': qoi_bytes
                                }
                            }, job['client'])


                # x0 = original image
                # x_T = seed noise
                # img_callback(img, pred_x0, i) for each step
                samples_ddim = model.sample(S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=len(batch),
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=batch[0]['scale'],
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                #img_callback=preview_img,
                                x_T=start_code)

                modelFS.to(device)
                print("saving images")

                counts_in_dirs = {}
                for i in range(len(batch)):
                    job = batch[i]
                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                # for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')

                    hashkey = json.dumps({ 'prompt': job['prompt'] })
                    hash = hashlib.sha1(hashkey.encode('utf8')).hexdigest()
                    dirname = re.sub(r"[^A-Za-z0-9,]", lambda _: "_", job['prompt'])[:140] + hash

                    sample_path = os.path.join(outpath, "samples", dirname)

                    if not os.path.isdir(sample_path):
                        os.makedirs(sample_path, exist_ok=True)

                        with open(os.path.join(sample_path, 'options.txt'), 'a') as f:
                            json.dump({ 'prompt': job['prompt']}, f)

                    if sample_path not in counts_in_dirs:
                        counts_in_dirs[sample_path] = len(os.listdir(sample_path)) - 1

                    start_time = time.time()
                    qoi_bytes = qoi.encode(x_sample.astype(np.uint8, order='C'))
                    end_time = time.time()
                    qoi_size = len(qoi_bytes)
                    print(f"qoi: {end_time - start_time}, size: {qoi_size}")

                    start_time = time.time()
                    with io.BytesIO() as mem:
                        Image.fromarray(x_sample.astype(np.uint8)).save(mem, format="png")
                        end_time = time.time()

                        with mem.getbuffer() as buffer:

                            print(f"png: {end_time - start_time}, size: {len(buffer)}")
                            with open(os.path.join(sample_path, f"{counts_in_dirs[sample_path]:05}_{job['seed']}_{job['scale']}_{job['steps']}.png"), 'wb') as f:
                                f.write(buffer)

                    serv.send_response({
                        'ty': 'result',
                        'result': {
                            'id': job['id'],
                            'index': job['index'],
                            'seed': job['seed'],
                            'scale': job['scale'],
                            'steps': job['steps'],
                            'image': qoi_bytes
                        }
                    }, job['client'])

                    #Image.fromarray(x_sample.astype(np.uint8)).save(
                    #    os.path.join(sample_path, f"{counts_in_dirs[sample_path]:05}_{batch[i]['seed']}.png"))

                    counts_in_dirs[sample_path] += 1

                move_to_cpu(modelFS)

                del samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated()/1e6)
except Exception as e:
    print('Exception!', e)
    traceback.print_exc()

serv.stop()

#print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))
print('Exiting')