#!../../.venv python3

import os
import sys
import base64
import numpy as np
from PIL import Image, ImageOps
import io
from io import BytesIO
import torch
import torchvision
import tensorflow as tf
import asyncio
import logging
import grpc
import flatbuffers
import google.protobuf as pb
from . import imageService_pb2
from . import imageService_pb2_grpc
from . import Control
from . import LoRA
from . import GenerationConfiguration
import hashlib
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
from comfy.cli_args import args
import latent_preview
import comfy.latent_formats as latent_formats

MAX_RESOLUTION=16384
MAX_PREVIEW_RESOLUTION = args.preview_size

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def prepare_callback(step, total_steps, preview_image):
    pbar = comfy.utils.ProgressBar(step)
    def callback(step, total_steps):
        preview_bytes = None
        if preview_image is not None:
            preview_bytes = ("PNG", preview_image, MAX_PREVIEW_RESOLUTION)
        pbar.update_absolute(step, total_steps, preview_bytes)
    return callback(step, total_steps)

def image_to_base64(image_tensor: torch.Tensor):
    if image_tensor is not None:
        image_tensor = image_tensor.permute(3, 1, 2, 0).squeeze(3)
        transform = torchvision.transforms.ToPILImage()
        img = transform(image_tensor)

        # Save the image to a BytesIO object (in memory) rather than to a file
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # Encode the image as base64
        encoded_string = base64.b64encode(buffered.getvalue())
        return encoded_string
    return None

def convert_response_image(response_image: bytes):
    int_buffer = np.frombuffer(response_image, dtype=np.uint32, count=17)
    height, width, channels = int_buffer[6:9]

    offset = 68
    length = width * height * channels * 2

    print(f"Response image is {width}x{height} with {channels} channels")
    # print(f"Input size: {len(response_image)} (Expected: {length + 68})")

    f16rgb = np.frombuffer(response_image, dtype=np.float16, count=length // 2, offset=offset)
    u8c = np.clip((f16rgb + 1) * 127, 0, 255).astype(np.uint8)

    return {
        'data': u8c,
        'width': width,
        'height': height,
        'channels': channels,
    }

def convert_image_for_request(img: torch.Tensor):
    # C header + the Float16 blob of -1 to 1 values that represents the image (in RGB order and HWC format, meaning r(0, 0), g(0, 0), b(0, 0), r(1, 0), g(1, 0), b(1, 0) .... (r(x, y) represents the value of red at that particular coordinate). The actual header is a bit more complex, here is the reference: https://github.com/liuliu/s4nnc/blob/main/nnc/Tensor.swift#L1750 the ccv_nnc_tensor_param_t is here: https://github.com/liuliu/ccv/blob/unstable/lib/nnc/ccv_nnc_tfb.h#L79 The type is CCV_TENSOR_CPU_MEMORY, format is CCV_TENSOR_FORMAT_NHWC, datatype is CCV_16F (for Float16), dim is the dimension in N, H, W, C order (in the case it should be 1, actual height, actual width, 3).

    # ComfyUI: An IMAGE is a torch.Tensor with shape [B,H,W,C], C=3. If you are going to save or load images, you will need to convert to and from PIL.Image format - see the code snippets below! Note that some pytorch operations offer (or expect) [B,C,H,W], known as ‘channel first’, for reasons of computational efficiency. Just be careful.
    # A LATENT is a dict; the latent sample is referenced by the key samples and has shape [B,C,H,W], with C=4.

    width = img.size(dim=2)
    height = img.size(dim=1)
    channels = img.size(dim=3)

    offset = 68
    length = width * height * channels * 2

    print(f"Request image is {width}x{height} with {channels} channels")

    data = img.to(torch.float16)

    # Encode the image as base64
    encoded_string = base64.b64encode(tf.io.serialize_tensor(data))
    # encoded_string = base64.b64encode(data)
    return encoded_string

def get_files(server, port):
    with grpc.insecure_channel(f"{server}:{port}") as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        response = stub.Echo(imageService_pb2.EchoRequest(name="ComfyUI"))
        DrawThingsSampler.files_list = json.loads(pb.json_format.MessageToJson(response))["files"]
        print(response.message) # HELLO ComfyUI
        return json.loads(pb.json_format.MessageToJson(response))["files"]

async def dt_sampler(
                server, 
                port, 
                model, 
                seed, 
                steps, 
                cfg, 
                strength, 
                sampler_name, 
                positive, 
                negative, 
                width, 
                height, 
                batch_count, 
                scale_factor=1,
                image=None, 
                mask=None,
                control_net=None,
                lora=None
                ) -> None:

    builder = flatbuffers.Builder(0)

    loras = None
    if lora is not None:
        fin_loras = []
        for lora_cfg in lora["loras"]:
            lora_name = builder.CreateString(lora_cfg["lora_name"])
            LoRA.Start(builder)
            LoRA.AddFile(builder, lora_name)
            LoRA.AddWeight(builder, lora_cfg["lora_weight"])
            fin_lora = LoRA.End(builder)
            fin_loras.append(fin_lora)

        GenerationConfiguration.StartLorasVector(builder, len(lora["loras"]))
        for i, lora_cfg in enumerate(lora["loras"]):
            print(f"{i+1} loras loaded")
            builder.PrependUOffsetTRelative(fin_loras[i])
        loras = builder.EndVector()

    controls = None
    if control_net is not None:
        fin_controls = []
        for control_cfg in control_net["control_nets"]:
            control_name = builder.CreateString(control_cfg["control_name"])
            Control.Start(builder)
            Control.AddFile(builder, control_name)
            Control.AddInputOverride(builder, DrawThingsLists.control_input_type.index(control_cfg["control_input_type"]))
            Control.AddControlMode(builder, DrawThingsLists.control_mode.index(control_cfg["control_mode"]))
            Control.AddWeight(builder, control_cfg["control_weight"])
            Control.AddGuidanceStart(builder, control_cfg["control_start"])
            Control.AddGuidanceEnd(builder, control_cfg["control_end"])
            Control.AddNoPrompt(builder, False)
            Control.AddGlobalAveragePooling(builder, False)
            Control.AddDownSamplingRate(builder, 0)
            fin_control = Control.End(builder)
            fin_controls.append(fin_control)

        GenerationConfiguration.StartControlsVector(builder, len(control_net["control_nets"]))
        for i, control_cfg in enumerate(control_net["control_nets"]):
            print(f"{i+1} controlnets loaded")
            builder.PrependUOffsetTRelative(fin_controls[i])
        controls = builder.EndVector()

    start_width = width // 64 // scale_factor
    start_height = height // 64 // scale_factor
    model_name = builder.CreateString(model)
    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddModel(builder, model_name)
    GenerationConfiguration.AddStrength(builder, strength)
    GenerationConfiguration.AddSeed(builder, seed)
    GenerationConfiguration.AddSeedMode(builder, DrawThingsLists.seed_mode.index("ScaleAlike"))
    GenerationConfiguration.AddStartWidth(builder, start_width)
    GenerationConfiguration.AddStartHeight(builder, start_height)
    GenerationConfiguration.AddTargetImageWidth(builder, width)
    GenerationConfiguration.AddTargetImageHeight(builder, height)
    # upscaler
    GenerationConfiguration.AddSteps(builder, steps)
    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    # speed-up
    GenerationConfiguration.AddSampler(builder, DrawThingsLists.sampler_list.index(sampler_name))
    # res shift
    GenerationConfiguration.AddShift(builder, 2.33)
    GenerationConfiguration.AddBatchSize(builder, 1)
    # refiner
    # zero neg
    # sep clip
    GenerationConfiguration.AddClipSkip(builder, 1)
    GenerationConfiguration.AddSharpness(builder, 0.6)
    GenerationConfiguration.AddMaskBlur(builder, 5)
    GenerationConfiguration.AddMaskBlurOutset(builder, 4)
    GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, True)
    # face restore
    GenerationConfiguration.AddHiresFix(builder, False)
    GenerationConfiguration.AddTiledDecoding(builder, False)
    GenerationConfiguration.AddTiledDiffusion(builder, False)
    # ti embed
    GenerationConfiguration.AddBatchCount(builder, batch_count)
    if controls is not None:
        GenerationConfiguration.AddControls(builder, controls)
    if loras is not None:
        GenerationConfiguration.AddLoras(builder, loras)
    builder.Finish(GenerationConfiguration.End(builder))
    configuration = builder.Output()
    # generated = GenerationConfiguration.GenerationConfiguration.GetRootAs(configuration, 0)

    contents = []
    img2img = None
    maskimg = None
    if image is not None:
        img2img = bytes(convert_image_for_request(image))
    if mask is not None:
        maskimg = bytes(convert_image_for_request(mask))

    models_override = [{
        "default_scale": 8,
        "file": "sd_v1.5_f16.ckpt",
        "name": "Generic (Stable Diffusion v1.5)",
        "prefix": "",
        "upcast_attention": False,
        "version": "v1"}]
    override = imageService_pb2.MetadataOverride(
        models = bytes(f"{models_override}", encoding='utf-8'),
        loras = b'["hyper_sd_v1.x_4_step_lora_f16.ckpt"]',
        controlNets = b'["controlnet_depth_1.x_v1.1_f16.ckpt"]',
        textualInversions = b"[]",
        upscalers = b"[]"
    )

    tensor_and_weight = []
    if control_net is not None:
        for control_cfg in control_net["control_nets"]:
            image = control_cfg["image"]
            if image is not None:
                tensor_and_weight.append(
                    imageService_pb2.TensorAndWeight(
                        tensor = bytes(convert_image_for_request(image)),
                        weight = control_cfg["control_weight"]
                    )
                )

    hints = [
        imageService_pb2.HintProto(
            hintType = "depth",
            tensors = tensor_and_weight
        )
    ]

    async with grpc.aio.insecure_channel(f"{server}:{port}") as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        generate_stream = stub.GenerateImage(imageService_pb2.ImageGenerationRequest(
            image = img2img,                      # Image data as sha256 content.
            scaleFactor = scale_factor,
            mask = maskimg,                       # Optional  Mask data as sha256 content.
            hints = hints,                        # List of hints
            prompt = positive,                    # Optional prompt string
            negativePrompt = negative,            # Optional negative prompt string
            configuration = bytes(configuration), # Configuration data as bytes (FlatBuffer)
            # override = override,                  # Override the existing metadata on various Zoo objects.
            user = "ComfyUI",                     # The name of the client.
            device = "LAPTOP",                    # The type of the device uses.
            # contents = contents                   # The image data as array of bytes. It is addressed by its sha256 content. This is modeled as content-addressable storage.
        ))
        while True:
            response = await generate_stream.read()
            if response == grpc.aio.EOF:
                break

            current_step = response.currentSignpost.sampling.step
            preview_image = response.previewImage
            generated_images = response.generatedImages
            print(f"current_step: {current_step}")

            if current_step:
                img = None
                if preview_image:
                    result = convert_response_image(preview_image)
                    data = result['data']
                    width = result['width']
                    height = result['height']
                    channels = result['channels']
                    img = Image.frombytes('RGBA', (width, height), data)

                    x0 = torch.tensor(data, dtype=torch.float16)

                    latent_format = latent_formats.SD15

                    # img = latent_preview.Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias).decode_latent_to_preview(x0=preview_image)
                prepare_callback(current_step, steps, img)

            if generated_images:
                images = []
                for img_data in response.generatedImages:
                    # Convert the image data to a Pillow Image object
                    result = convert_response_image(img_data)
                    data = result['data']
                    width = result['width']
                    height = result['height']
                    channels = result['channels']
                    mode = "RGB"
                    if channels >= 4:
                        mode = "RGBA"
                    img = Image.frombytes(mode, (width, height), data)
                    print(f"size: {img.size}, mode: {img.mode}")
                    image_np = np.array(img)
                    # Convert to float32 tensor and normalize
                    tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
                    images.append(tensor_image)
                return (torch.stack(images),)

class DrawThingsLists:
    dtserver = "localhost"
    dtport = "7859"

    files_list = ["Press R to (re)load this list"]

    sampler_list = [
                "DPMPP 2M Karras",
                "Euler A",
                "DDIM",
                "PLMS",
                "DPMPP SDE Karras",
                "UniPC",
                "LCM",
                "Euler A Substep",
                "DPMPP SDE Substep",
                "TCD",
                "Euler A Trailing",
                "DPMPP SDE Trailing",
                "DPMPP 2MA YS",
                "Euler A AYS",
                "DPMPP SDE AYS",
                "DPMPP 2M Trailing",
                "DDIM Trailing",
            ]

    seed_mode = [
                "Legacy",
                "TorchCpuCompatible",
                "ScaleAlike",
                "NvidiaGpuCompatible",
            ]

    control_mode = [
                "Balanced",
                "Prompt",
                "Control",
            ]

    control_input_type = [
                "Unspecified",
                "Custom",
                "Depth",
                "Canny",
                "Scribble",
                "Pose",
                "Normalbae",
                "Color",
                "Lineart",
                "Softedge",
                "Seg",
                "Inpaint",
                "Ip2p",
                "Shuffle",
                "Mlsd",
                "Tile",
                "Blur",
                "Lowquality",
                "Gray",
            ]

class DrawThingsSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        def get_filtered_files():
            all_files = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
            filtered_files = [
                f for f in all_files
                if not any(exclude in f for exclude in ["vae", "lora", "clip", "encoder"])
            ]
            return filtered_files
        
        return {
            "required": {
                "server": ("STRING", {"multiline": False, "default": DrawThingsLists.dtserver, "tooltip": "The IP address of the Draw Things gRPC Server."}),
                "port": ("STRING", {"multiline": False, "default": DrawThingsLists.dtport, "tooltip": "The port that the Draw Things gRPC Server is listening on."}),
                "model": (get_filtered_files(), {"default": "Press R to (re)load this list", "tooltip": "The model used for denoising the input latent.\nPlease note that this lists all files, so be sure to pick the right one.\nPress R to (re)load this list."}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01, "tooltip": "When generating from an image, a high value allows more artistic freedom from the original. 1.0 means no influence from the existing image (a.k.a. text to image)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (DrawThingsLists.sampler_list, {"default": "DPMPP 2M Trailing", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "hidden": {
                "scale_factor": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "positive": ("STRING", {
                    "multiline": True, "default": "a lovely cat", "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("STRING", {
                    "multiline": True, "default": "text, watermark", "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "lora": ("dict", ),
                "control_net": ("dict", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    DESCRIPTION = ""
    FUNCTION = "sample"
    CATEGORY = "DrawThings"

    def sample(self, 
                server, 
                port, 
                model, 
                seed, 
                steps, 
                cfg, 
                strength, 
                sampler_name, 
                positive, 
                negative, 
                width, 
                height, 
                batch_count, 
                scale_factor=1,
                image=None, 
                mask=None,
                control_net=None,
                lora=None
                ):
        DrawThingsLists.dtserver = server
        DrawThingsLists.dtport = port
        DrawThingsLists.files_list = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
        return asyncio.run(dt_sampler(
                server, 
                port, 
                model, 
                seed, 
                steps, 
                cfg, 
                strength, 
                sampler_name, 
                positive, 
                negative, 
                width, 
                height, 
                batch_count=batch_count, 
                scale_factor=scale_factor,
                image=image, 
                mask=mask,
                control_net=control_net,
                lora=lora
                ))

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsControlNet:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        def get_filtered_files():
            all_files = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
            filtered_files = [
                f for f in all_files
                if not any(exclude in f for exclude in ["lora", "vae"])
            ]
            return filtered_files
        
        return {
            "required": { 
                "control_name": (get_filtered_files(), {"default": "Press R to (re)load this list", "tooltip": "The model used.\nPlease note that this lists all files, so be sure to pick the right one.\nPress R to (re)load this list."}),
                "control_input_type": (DrawThingsLists.control_input_type, {"default": "Unspecified"}),
                "control_mode": (DrawThingsLists.control_mode, {"default": "Balanced", "tooltip": ""}),
                "control_weight": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.50, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "control_start": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
            },
            "optional": {
                "control_net": ("dict", ),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("dict",)
    RETURN_NAMES = ("CONTROL_NET",)
    CATEGORY = "DrawThings"
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, control_name, control_input_type, control_mode, control_weight, control_start, control_end, control_net={}, image=None):
        # Check if 'control_nets' exists in the pipeline
        if "control_nets" not in control_net:
            # Create 'control_nets' as an empty list
            control_net["control_nets"] = []
        # Append the new entry as a dictionary to the list
        control_net["control_nets"].append({
            "control_name": control_name,
            "control_input_type": control_input_type,
            "control_mode": control_mode,
            "control_weight": control_weight,
            "control_start": control_start,
            "control_end": control_end,
            "image": image
        })
        return (control_net,)

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsLoRA:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        def get_lora_files():
            all_files = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
            lora_files = [f for f in all_files if "lora" in f]
            return lora_files
        
        return {
            "required": {
                "lora_name": (get_lora_files(), {"default": "Press R to (re)load this list", "tooltip": "The model used.\nPlease note that this lists all files, so be sure to pick the right one.\nPress R to (re)load this list."}),
                "lora_weight": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.50, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            },
            "optional": {
                "lora": ("dict", ),
            }
        }

    RETURN_TYPES = ("dict",)
    RETURN_NAMES = ("LORA",)
    CATEGORY = "DrawThings"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, lora_name, lora_weight, lora={}):
        # Check if 'loras' exists in the pipeline
        if "loras" not in lora:
            # Create 'loras' as an empty list
            lora["loras"] = []
        # Append the new entry as a dictionary to the list
        lora["loras"].append({"lora_name": lora_name, "lora_weight": lora_weight})
        # print(f"lora: {lora}")
        return (lora,)

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DrawThingsSampler": DrawThingsSampler,
    "DrawThingsControlNet": DrawThingsControlNet,
    "DrawThingsLoRA": DrawThingsLoRA,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsSampler": "Draw Things Sampler",
    "DrawThingsControlNet": "Draw Things ControlNet",
    "DrawThingsLoRA": "Draw Things LoRA",
}
