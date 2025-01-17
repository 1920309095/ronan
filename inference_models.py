from diffusers import DDIMPipeline
import torch
from torchvision import transforms
from dcgan import DCGAN
import dnnlib
import dnnlib.tflib as tflib
import pickle
from pl_bolts.models.autoencoders import VAE
from diffusers import StableDiffusionPipeline
from cm_inference import cm_inference
from cm.script_util import model_and_diffusion_defaults,create_model_and_diffusion,add_dict_to_argparser,args_to_dict,args_to_dict_
from cm.random_util import get_generator
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import argparse

def get_init_noise(args,model_type,bs=1):
    if model_type in ["ddpm_cifar10"]:
        init_noise = torch.randn(bs, args.cur_model.unet.in_channels, args.cur_model.unet.sample_size, args.cur_model.unet.sample_size).cuda()
    elif model_type in ["dcgan_cifar10"]:
        init_noise = torch.randn(bs, args.cur_model.nz, 1, 1).cuda()
    elif model_type in ["styleganv2ada_cifar10"]:
        init_noise = torch.randn([bs, args.cur_model.z_dim]).cuda()
    elif model_type in ["vae_cifar10"]:
        init_noise = torch.randn([bs, args.cur_model.latent_dim]).cuda()
    elif model_type in ["sd"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        init_noise = torch.randn([bs, args.cur_model.unet.in_channels, height // args.cur_model.vae_scale_factor, width // args.cur_model.vae_scale_factor]).cuda()
    elif model_type in ["sd_unet"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        init_noise_0 = torch.randn([bs, args.cur_model.unet.in_channels, height // args.cur_model.vae_scale_factor, width // args.cur_model.vae_scale_factor]).cuda()
        init_noise_text = torch.cat([torch.randn(args.cur_model.prompt_embeds_shape)] * bs).cuda()
        init_noise = (init_noise_0,init_noise_text)
    elif model_type in ["biggan"]:
        # Prepare a input
        truncation = 0.4
        class_vector = one_hot_from_names([args.classification], batch_size=1)
        assert class_vector is not None, f"Invalid classification: {args.classification}"
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        init_noise=(noise_vector,class_vector,truncation)
    elif "cm" in model_type:
        #init_noise = args.generator_.randn(*(args.batch_size, 3, args.image_size, args.image_size)).cuda()
        init_noise = torch.randn(*(bs, 3, 64, 64)).cuda()

    return init_noise

def from_noise_to_image(args,model,noise,model_type):
    if model_type in ["ddpm_cifar10"]:
        image = model.input2output(noise,num_inference_steps=50)
    elif model_type in ["dcgan_cifar10"]:
        image = model.input2output(noise)
        image = transforms.Resize(32)(image)
        print(image.shape)
    elif model_type in ["biggan"]:
        image=model(noise)
        image=image.clamp(0,1)
        image=transforms.Resize(32)(image)
    elif model_type in ["styleganv2ada_cifar10"]:
        label = torch.zeros([noise.shape[0], model.c_dim]).cuda()
        image = model(noise, label, noise_mode='none')
        image = (image / 2 + 0.5).clamp(0, 1)
        image = transforms.Resize(32)(image)
        print(image.shape)
    elif model_type in ["vae_cifar10"]:
        image = model.decoder(noise)
        image = image*args.vae_t_std + args.vae_t_mean
        image = image.clamp(0, 1)
        print(image.min())
        print(image.max())
    elif model_type in ["sd"]:
        image = model.latent2output(noise)
    elif model_type in ["sd_unet"]:
        image = model.half_unet2output(noise[0],noise[1])
    elif "cm" in model_type:
        #if noise.shape[0]==1:
        #    noise = noise.expand(2,noise.shape[1],noise.shape[2],noise.shape[3])
        image = cm_inference(model,noise)
    return image


def get_model(model_type,model_path,args):
    if model_type == "ddpm_cifar10":
        model_id = "google/ddpm-cifar10-32"
        model_id_1 = "google/ddpm-cifar10-32"
        cur_model = DDIMPipeline.from_pretrained(model_id).to("cuda")
        #ddim_1 = DDIMPipeline.from_pretrained(model_id).to("cuda")
        cur_model.unet.eval()

    elif model_type == "dcgan_cifar10":
        ngpu = 1
        cur_model = DCGAN(ngpu)
        if model_path:
            cur_model.load_state_dict(torch.load(model_path))
        else:
            cur_model.load_state_dict(torch.load("./dcgan_weights/netG_epoch_24.pth"))
        cur_model = cur_model.cuda()
        cur_model.eval()

    elif model_type =="biggan":
        model_id= "biggan-deep-256"
        cur_model = BigGAN.from_pretrained(model_id)
        model = model.cuda()
        model.eval()

    elif model_type == "styleganv2ada_cifar10":
        tflib.init_tf()
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl"
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            #_G, _D, Gs = pickle.load(fp)
            G = pickle.load(fp)['G_ema'].cuda()  # torch.nn.Module
        cur_model = G.eval()
        z = torch.randn([args.bs, cur_model.z_dim]).cuda()    # latent codes
        print(cur_model.c_dim)
        label = torch.zeros([args.bs, cur_model.c_dim]).cuda()                              # class labels (not used in this example)
        class_idx = 9
        label[:, class_idx] = 1
        img = cur_model(z, label,noise_mode='none')
        args.stylegan_class_idx = class_idx

    elif model_type == "vae_cifar10":
        cur_model = VAE(input_height=32)
        print(VAE.pretrained_weights_available())
        cur_model = cur_model.from_pretrained('cifar10-resnet18')
        cur_model.freeze()
        cur_model = cur_model.cuda()
        cur_model = cur_model.eval()

        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        
        channel = 3
        size = 32
        t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
        args.vae_t_mean = t_mean
        args.vae_t_std = t_std

    elif "cm" in model_type:
        
        '''def cm_create_argparser(parser):
            defaults = dict(
                training_mode="edm",
                generator="determ",
                clip_denoised=True,
                num_samples=10000,
                batch_size=16,
                sampler="heun",
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=float("inf"),
                s_noise=1.0,
                steps=40,
                model_path="",
                seed=42,
                ts="",
            )
            defaults.update(model_and_diffusion_defaults())
            add_dict_to_argparser(parser, defaults)
            return parser'''
        
        #parser_cm = argparse.ArgumentParser()
        #args_cm = cm_create_argparser(parser_cm).parse_args()

        defaults = dict(
            training_mode="edm",
            generator="determ",
            clip_denoised=True,
            num_samples=10000,
            batch_size=16,
            sampler="heun",
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0,
            steps=40,
            model_path="",
            seed=42,
            ts="",
        )
        defaults.update(model_and_diffusion_defaults())
        args_cm = defaults
        args_cm["batch_size"] = args.bs

        if model_type == "cm_cd_lpips":
            args_cm["training_mode"] = "consistency_distillation"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,22,39"
            args_cm["steps"] = 40
            args_cm["model_path_"]= "./consistency_models/scripts/cd_imagenet64_lpips.pt"
        elif model_type == "cm_cd_l2":
            args_cm["training_mode"] = "consistency_distillation"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,22,39"
            args_cm["steps"] = 40
            args_cm["model_path_"] = "./consistency_models/scripts/cd_imagenet64_l2.pt"
        elif model_type == "cm_ct":
            args_cm["training_mode"] = "consistency_training"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,106,200"
            args_cm["steps"] = 201
            args_cm["model_path_"] = "./consistency_models/scripts/cd_imagenet64_l2.pt"
        args_cm["attention_resolutions"] = "32,16,8"
        args_cm["class_cond"] = True
        args_cm["use_scale_shift_norm"] = True
        args_cm["dropout"] = 0.0
        args_cm["image_size"] = 64
        args_cm["num_channels"] = 192
        args_cm["num_head_channels"] = 64
        args_cm["num_res_blocks"] = 3
        args_cm["num_samples"] = 500
        args_cm["resblock_updown"] = True
        args_cm["use_fp16"] = True
        args_cm["weight_schedule"] = "uniform"

        if "consistency" in args_cm["training_mode"]:
            distillation = True
        else:
            distillation = False
        # cm_model, diffusion = create_model_and_diffusion(
        #     **args_to_dict(args_cm, model_and_diffusion_defaults().keys()),
        #     distillation=distillation,
        # )
        cm_model, diffusion = create_model_and_diffusion(
            **args_to_dict_(args_cm, model_and_diffusion_defaults().keys()),
            distillation=distillation,
        )

        cm_model.load_state_dict(torch.load(args_cm["model_path_"], map_location="cpu"))
        cm_model.cuda()
        if args_cm["use_fp16"]:
            cm_model.convert_to_fp16()
        cm_model.eval()

        if args_cm["sampler"] == "multistep":
            assert len(args_cm["ts"]) > 0
            ts = tuple(int(x) for x in args_cm["ts"].split(","))
        else:
            ts = None
        args_cm["ts_"] = ts
        generator = get_generator(args_cm["generator"], args_cm["num_samples"], args_cm["seed"])
        args_cm["generator_"] = generator
        args_cm["shape"] = (args_cm["batch_size"], 3, args_cm["image_size"], args_cm["image_size"])

        cur_model = (args_cm, cm_model, diffusion)

    elif model_type in ["sd","sd_unet"]:
        #model_id = "runwayml/stable-diffusion-v1-5"
        #model_id = "stabilityai/stable-diffusion-2"
        #model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        model_id = "stabilityai/stable-diffusion-2-base"
        #model_id = "OFA-Sys/small-stable-diffusion-v0"
        
        cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        #sd = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    return cur_model
