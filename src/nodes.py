#!../../.venv python3

import os
import sys
import base64
import torch
import asyncio
import grpc
import flatbuffers
import json
import numpy as np
from PIL import Image
from server import PromptServer
from aiohttp import web
from google.protobuf.json_format import MessageToJson

from .generated import imageService_pb2, imageService_pb2_grpc
from .credentials import credentials
from .data_types import *
from .config import build_config
from .image_handlers import prepare_callback, convert_response_image, decode_preview, convert_image_for_request, convert_mask_for_request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
show_preview = True

def get_channel(server, port, use_tls):
    if use_tls and credentials is not None:
        return grpc.secure_channel(f"{server}:{port}", credentials)
    return grpc.insecure_channel(f"{server}:{port}")


def get_aio_channel(server, port, use_tls):
    options = [["grpc.max_send_message_length", -1], ["grpc.max_receive_message_length", -1]]
    if use_tls and credentials is not None:
        return grpc.aio.secure_channel(f"{server}:{port}", credentials, options=options)
    return grpc.aio.insecure_channel(f"{server}:{port}", options=options)


def get_files(server, port, use_tls) -> ModelsInfo:
    with get_channel(server, port, use_tls) as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        response = stub.Echo(imageService_pb2.EchoRequest(name="ComfyUI"))
        response_json = json.loads(MessageToJson(response))
        DrawThingsSampler.files_list = response_json["files"]
        override = dict(response_json['override'])
        model_info = { k: json.loads(str(base64.b64decode(override[k]), 'utf8')) for k in override.keys() }

        if 'upscalers' not in model_info:
            official = [
                'realesrgan_x2plus_f16.ckpt',
                'realesrgan_x4plus_f16.ckpt',
                'realesrgan_x4plus_anime_6b_f16.ckpt',
                'esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt',
                'remacri_4x_f16.ckpt',
                '4x_ultrasharp_f16.ckpt',
            ]
            model_info['upscalers'] = [UpscalerInfo(file=f, name=f) for f in official if f in DrawThingsSampler.files_list]

        return model_info


routes = PromptServer.instance.routes
@routes.post('/dt_grpc_files_info')
async def handle_files_info_request(request):
    try:
        post = await request.post()
        server = post.get('server')
        port = post.get('port')
        use_tls = post.get('use_tls')

        if server is None or port is None:
            return web.json_response({"error": "Missing server or port parameter"}, status=400)
        all_files = get_files(server, port, use_tls)
        return web.json_response(all_files)
    except Exception as e:
        print(e)
        return web.json_response({"error": "Could not connect to Draw Things gRPC server. Please check the server address and port."}, status=500)


@routes.post('/dt_grpc_preview')
async def handle_preview_request(request):
    global show_preview
    try:
        post = await request.post()
        show_preview = False if post.get('preview') == "none" else True
        print('show preview:', show_preview)
        return web.json_response()
    except Exception as e:
        print(e)
        return web.json_response()

async def dt_sampler(inputs: dict):
    server, port, use_tls = inputs.get('server'), inputs.get('port'), inputs.get('use_tls')
    positive, negative = inputs.get('positive'), inputs.get('negative')
    image, mask = inputs.get('image'), inputs.get('mask')

    version = inputs["model"]["value"]["version"] if "value" in inputs["model"] and "version" in inputs["model"]["value"] else None
    inputs["version"] = version
    config = build_config(inputs)

    builder = flatbuffers.Builder(0)
    builder.Finish(config.Pack(builder))
    config_fbs = bytes(builder.Output())

    try:
        print(json.dumps(inputs, indent=4))
        print(json.dumps(config, indent=4))
    except Exception as e:
        pass

    contents = []
    img2img = None
    maskimg = None
    if image is not None:
        img2img = convert_image_for_request(image)
    if mask is not None:
        maskimg = convert_mask_for_request(mask, config.startWidth * 64, config.startHeight * 64)

    # override = imageService_pb2.MetadataOverride()

    hints = []
    cnets = inputs.get("control_net")
    if cnets is not None:
        for cnet in cnets:
            if cnet.get("image") is not None:
                c_input_slot = cnet["input_type"] if cnet["input_type"] in ["Custom", "Depth", "Scribble", "Pose", "Color"] else "Custom"
                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(cnet["image"], c_input_slot.lower())
                taw.weight = 1

                hnt = imageService_pb2.HintProto()
                hnt.hintType = c_input_slot.lower()
                hnt.tensors.append(taw)
                hints.append(hnt)

    lora = inputs.get("lora")
    if lora is not None:
        for lora_cfg in lora:
            if 'control_image' in lora_cfg:
                modifier = lora_cfg["model"]["modifier"]

                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(lora_cfg["control_image"], modifier)
                taw.weight = 1 # lora_cfg["weight"] if "weight" in lora_cfg else 1

                hnt = imageService_pb2.HintProto()
                hnt.hintType = modifier if modifier in ["custom", "depth", "scribble", "pose", "color"] else "custom"
                hnt.tensors.append(taw)
                hints.append(hnt)

    async with get_aio_channel(server, port, use_tls) as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        generate_stream = stub.GenerateImage(imageService_pb2.ImageGenerationRequest(
            image = img2img,
            scaleFactor = 1,
            mask = maskimg,
            hints = hints,
            prompt = positive,
            negativePrompt = negative,
            configuration = config_fbs,
            # override = override,
            user = "ComfyUI",
            device = "LAPTOP",
            contents = contents
        ))

        response_images = []
        print("show preview:", show_preview)
        while True:
            response = await generate_stream.read()
            if response == grpc.aio.EOF:
                break

            current_step = response.currentSignpost.sampling.step
            preview_image = response.previewImage
            generated_images = response.generatedImages

            if current_step:
                try:
                    x0 = None
                    if preview_image and version and show_preview:
                        x0 = decode_preview(preview_image,version)
                    prepare_callback(current_step, config.steps, x0)
                except Exception as e:
                    print('DrawThings-gRPC had an error decoding the preview image:', e)

            if generated_images:
                response_images.extend(generated_images)

        images = []
        for img_data in response_images:
            result = convert_response_image(img_data)
            if result is not None:
                data = result['data']
                width = result['width']
                height = result['height']
                channels = result['channels']
                mode = "RGB"
                if channels >= 4:
                    mode = "RGBA"
                img = Image.frombytes(mode, (width, height), data)
                image_np = np.array(img)
                tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
                images.append(tensor_image)

        return (torch.stack(images),)


class DrawThingsSampler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "settings": (["Basic", "Advanced", "All"], {"default": "Basic"}),
                "server": ("STRING", {"multiline": False, "default": DrawThingsLists.dtserver, "tooltip": "The IP address of the Draw Things gRPC Server."}),
                "port": ("STRING", {"multiline": False, "default": DrawThingsLists.dtport, "tooltip": "The port that the Draw Things gRPC Server is listening on."}),
                "use_tls": ("BOOLEAN", {"default": True}),

                "model": ("DT_MODEL", {"model_type": "models", "tooltip": "The model used for denoising the input latent."}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01, "tooltip": "When generating from an image, a high value allows more artistic freedom from the original. 1.0 means no influence from the existing image (a.k.a. text to image)."}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 4294967295, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "seed_mode": (DrawThingsLists.seed_mode, {"default": "ScaleAlike"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 8192, "step": 64}),
                # upscaler
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "tooltip": "The number of steps used in the denoising process."}),
                "num_frames": ("INT", {"default": 14, "min": 1, "max": 201, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 50.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),

                "speed_up": ("BOOLEAN", {"default": True}),
                "guidance_embed": ("FLOAT", {"default": 4.5, "min": 0, "max": 50, "step": 0.1}),

                "sampler_name": (DrawThingsLists.sampler_list, {"default": "DPM++ 2M AYS", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                # stochastic_sampling_gamma

                "res_dpt_shift": ("BOOLEAN", {"default": True}),

                "shift": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 8.00, "step": 0.01, "round": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                # refiner
                "fps": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
                "motion_scale": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "guiding_frame_noise": ("FLOAT", {"default": 0.02, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
                "start_frame_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 25.0, "step": 0.1, "round": 0.1}),
                "causal_inference": ("INT", {"default": 0, "min": 0, "max": 129, "step": 1, "tooltip": "Set to 0 to disable causal inference"}),

                # zero_negative_prompt
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 23, "step": 1}),
                "sharpness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.1}),
                "mask_blur": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 50.0, "step": 0.1, "round": 0.1}),
                "mask_blur_outset": ("INT", {"default": 0, "min": -100, "max": 1000, "step": 1}),
                "preserve_original": ("BOOLEAN", {"default": True}),
                # face_restoration

                "high_res_fix": ("BOOLEAN", {"default": False}),
                "high_res_fix_start_width": ("INT", {"default": 448, "min": 128, "max": 2048, "step": 64}),
                "high_res_fix_start_height": ("INT", {"default": 448, "min": 128, "max": 2048, "step": 64}),
                "high_res_fix_strength": ("FLOAT", {"default": 0.70, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),

                "tiled_decoding": ("BOOLEAN", {"default": False}),
                "decoding_tile_width": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 64}),
                "decoding_tile_height": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 64}),
                "decoding_tile_overlap": ("INT", {"default": 128, "min": 64, "max": 1024, "step": 64}),

                "tiled_diffusion": ("BOOLEAN", {"default": False}),
                "diffusion_tile_width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "diffusion_tile_height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "diffusion_tile_overlap": ("INT", {"default": 64, "min": 64, "max": 1024, "step": 64}),

                "tea_cache": ("BOOLEAN", {"default": False}),
                "tea_cache_start": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "tea_cache_end": ("INT", {"default": 2, "min": 0, "max": 1000, "step": 1}),
                "tea_cache_threshold": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
                "tea_cache_max_skip_steps": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),

                "separate_clip_l": ("BOOLEAN", {"default": False}),
                "clip_l_text": ("STRING", {"forceInput": False }),
                "separate_open_clip_g": ("BOOLEAN", {"default": False}),
                "open_clip_g_text": ("STRING", {"forceInput": False } ),

                # ti embed
                # image_guidance_scale
                # decode_with_attention
                # hires_fix_decode_with_attention
                # clip_weight
                # negative_prompt_for_image_prior
                # image_prior_steps

                # sdxl img size !!!

                # aesthetic_score
                # negative_aesthetic_score
                # name ???
                # cond_aug ???
                # stage_2_steps
                # stage_2_cfg
                # stage_2_shift
                # t5_text_encoder
                # separate_t5
                # t5_text

            },
            "hidden": {
                "scale_factor": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK", {"tooltip": "A black/white image where black areas will be kept and the rest will be regenerated according to your strength setting."}),
                "positive": ("STRING", {"forceInput": True, "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("STRING", {"forceInput": True, "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "lora": ("DT_LORA", ),
                "control_net": ("DT_CNET", ),
                "upscaler": ("DT_UPSCALER", ),
                "refiner": ("DT_REFINER", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    DESCRIPTION = ""
    FUNCTION = "sample"
    CATEGORY = "DrawThings"

    def sample(self, **kwargs):
        return asyncio.run(dt_sampler(kwargs))

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        PromptServer.instance.send_sync("dt-grpc-validate", dict({"hello": "js"}))
        return True


class DrawThingsRefiner:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner_model": ("DT_MODEL", {"model_type": "models" }),
                "refiner_start": ("FLOAT", {"default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("DT_REFINER",)
    RETURN_NAMES = ("REFINER",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, refiner_model, refiner_start):
        refiner = {"refiner_model": refiner_model, "refiner_start": refiner_start}
        return (refiner,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


class DrawThingsUpscaler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaler_model": ("DT_MODEL", {"model_type": "upscalers"}),
                "upscaler_scale_factor": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("DT_UPSCALER",)
    RETURN_NAMES = ("UPSCALER",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, upscaler_model, upscaler_scale_factor):
        upscaler = {"upscaler_model": upscaler_model, "upscaler_scale_factor": upscaler_scale_factor}
        return (upscaler,)


class DrawThingsPositive:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {
                    "multiline": True, "default": "a lovely cat", "tooltip": "The conditioning describing the attributes you want to include in the image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POSITIVE",)
    FUNCTION = "prompt"
    CATEGORY = "DrawThings"

    def prompt(self, positive):
        return (positive,)

class DrawThingsNegative:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "negative": ("STRING", {
                    "multiline": True, "default": "text, watermark", "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("NEGATIVE",)
    FUNCTION = "prompt"
    CATEGORY = "DrawThings"

    def prompt(self, negative):
        return (negative,)


class DrawThingsControlNet:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_name": ("DT_MODEL", {"model_type": "controlNets", "tooltip": "The model used."}),
                "control_input_type": (DrawThingsLists.control_input_type, {"default": "Custom"}),
                "control_mode": (DrawThingsLists.control_mode, {"default": "Balanced", "tooltip": ""}),
                "control_weight": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.50, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "control_start": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "invert_image": ("BOOLEAN", {"default": False, "tooltip": "Some Control Nets might need their image to be inverted."}),
            },
            "optional": {
                "control_net": ("DT_CNET", ),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("DT_CNET",)
    RETURN_NAMES = ("CONTROL_NET",)
    CATEGORY = "DrawThings"
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, control_name, control_input_type, control_mode, control_weight, control_start, control_end, control_net=None, image=None, invert_image=False) -> ControlNetInfo:
        if invert_image == True:
            image = 1.0 - image

        cnet_list: ControlStack = list()

        if control_net is not None:
            cnet_list.extend(control_net)

        cnet_info = ControlNetInfo(control_name["value"]) if 'value' in control_name else None

        if cnet_info is not None and 'file' in cnet_info:
            cnet_item = {
                "model": cnet_info,
                "input_type": control_input_type,
                "mode": control_mode,
                "weight": control_weight,
                "start": control_start,
                "end": control_end,
                "image": image
            }
            cnet_list.append(cnet_item)

        return (cnet_list,)

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

class DrawThingsLoRA:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ("DT_MODEL", {"model_type": "loras", "tooltip": "The model used."}),
                "lora_weight": ("FLOAT", {"default": 1.00, "min": -5.00, "max": 5.00, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            },
            "optional": {
                # "lora": ("DT_LORA",),
            }
        }

    RETURN_TYPES = ("DT_LORA",)
    RETURN_NAMES = ("LORA",)
    CATEGORY = "DrawThings"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, lora_name, lora_weight, **kwargs) -> LoraStack:
        prev_lora: LoraStack = kwargs.get("lora", None)
        control_image = kwargs.get("control_image", None)

        lora_list: LoraStack = list()
        if prev_lora is not None:
            lora_list.extend(prev_lora)

        lora_info = LoRAInfo(lora_name["value"]) if 'value' in lora_name else None

        if lora_info is not None and 'file' in lora_info:
            lora_item = { "model": lora_info, "weight": lora_weight }
            if control_image is not None:
                lora_item["control_image"] = control_image
            lora_list.append(lora_item)

        return (lora_list,)

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {
    "DrawThingsSampler": DrawThingsSampler,
    "DrawThingsControlNet": DrawThingsControlNet,
    "DrawThingsLoRA": DrawThingsLoRA,
    "DrawThingsPositive": DrawThingsPositive,
    "DrawThingsNegative": DrawThingsNegative,
    "DrawThingsRefiner": DrawThingsRefiner,
    "DrawThingsUpscaler": DrawThingsUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsSampler": "Draw Things Sampler",
    "DrawThingsControlNet": "Draw Things Control Net",
    "DrawThingsLoRA": "Draw Things LoRA",
    "DrawThingsPositive": "Draw Things Positive Prompt",
    "DrawThingsNegative": "Draw Things Negative Prompt",
    "DrawThingsRefiner": "Draw Things Refiner",
    "DrawThingsUpscaler": "Draw Things Upscaler",
}
