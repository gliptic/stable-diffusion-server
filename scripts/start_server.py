import argparse #, os, sys, glob
import torch
from torch import autocast, nn
import numpy as np
from omegaconf import OmegaConf
#from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
#import time
from pytorch_lightning import seed_everything
from contextlib import contextmanager, nullcontext
from server.server import Server

from ldm.util import instantiate_from_config

import k_diffusion as K
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    model = model.half()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
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
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    # if opt.plms:
    #     sampler = PLMSSampler(model)
    # elif opt.k:
    #     sampler = None
    #     #device = accelerator.device
    #     #seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    #     #torch.manual_seed(seeds[accelerator.process_index].item())
    #     model_wrap = K.external.CompVisDenoiser(model)
    #     sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
    # else:
    #     sampler = DDIMSampler(model)

    outpath = opt.outdir
    batch_size = opt.n_samples

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    
    serv = Server()
    serv.start()

    print('device', device)

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                while serv.wait_for_requests():
                    batch = serv.batch_requests(batch_size)

                    sampler_name = serv.sampler_for_batch(batch)
                    if sampler_name == 'PLMS':
                        sampler = PLMSSampler(model)
                    elif sampler_name == 'DDIM':
                        sampler = DDIMSampler(model)
                    elif sampler_name == 'k_dpm_2_a':
                        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
                    elif sampler_name == 'k_dpm_2':
                        sampler = KDiffusionSampler(model,'dpm_2')
                    elif sampler_name == 'k_euler_a':
                        sampler = KDiffusionSampler(model,'euler_ancestral')
                    elif sampler_name == 'k_euler':
                        sampler = KDiffusionSampler(model,'euler')
                    elif sampler_name == 'k_heun':
                        sampler = KDiffusionSampler(model,'heun')
                    elif sampler_name == 'k_lms':
                        sampler = KDiffusionSampler(model,'lms')

                    uc = None
                    if serv.scale_for_batch(batch) != 1.0:
                        uc = model.get_learned_conditioning(len(batch) * [""])
                    prompts = serv.prompts_for_batch(batch)
                    
                    c = model.get_learned_conditioning(prompts)
                    shape = serv.shape_for_batch(batch)

                    print('shape', shape)
                    print('batch len', len(batch))
                    start_code = serv.startcodes_for_batch(batch).to(device)

                    def preview(img, pred, i):
                        if (i % 10) == 9:
                            samples_ddim = img
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for i in range(len(batch)):
                                job = batch[i]
                                
                                x_sample = x_samples_ddim[i]
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

                                image = x_sample.astype(np.uint8, order='C')

                                image_bytes = serv.encode_image(job, image)
                                serv.send_response(job, image_bytes)
                    
                    samples_ddim, _ = sampler.sample(S=serv.steps_for_batch(batch),
                                                    conditioning=c,
                                                    batch_size=len(batch),
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=serv.scale_for_batch(batch),
                                                    unconditional_conditioning=uc,
                                                    #img_callback=preview,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(len(batch)):
                        job = batch[i]
                        
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

                        image = x_sample.astype(np.uint8, order='C')

                        image_bytes = serv.save_image(job, image, outpath)
                        serv.send_response(job, image_bytes)


if __name__ == "__main__":
    main()
