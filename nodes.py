#!../../.venv python3

import os
import sys
import base64
import numpy as np
from PIL import Image
import io
from io import BytesIO
import torch
import torchvision
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

MAX_RESOLUTION=16384
MAX_PREVIEW_RESOLUTION = args.preview_size

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def prepare_callback(step, total_steps, preview_image):
    pbar = comfy.utils.ProgressBar(step)
    def callback(step, total_steps):
        preview_bytes = None
        if preview_image is not None:
            preview_bytes = ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)
        pbar.update_absolute(step, total_steps, preview_bytes)
    return callback(step, total_steps)

def image_to_base64(image_tensor):
    if image_tensor is not None:
        image_tensor = image_tensor.permute(3, 1, 2, 0)
        image_tensor = image_tensor.squeeze(3)
        transform = torchvision.transforms.ToPILImage()
        img = transform(image_tensor)

        # Save the image to a BytesIO object (in memory) rather than to a file
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # Encode the image as base64
        encoded_string = base64.b64encode(buffered.getvalue())
        return encoded_string
    return None

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
            builder.PrependUOffsetTRelative(fin_loras[i])
        loras = builder.EndVector()

    controls = None
    if control_net is not None:
        fin_controls = []
        for control_cfg in control_net["control_nets"]:
            control_name = builder.CreateString(control_cfg["control_net_name"])
            Control.Start(builder)
            Control.AddFile(builder, control_name)
            Control.AddInputOverride(builder, DrawThingsLists.control_input_type.index(control_cfg["control_input_type"]))
            Control.AddControlMode(builder, DrawThingsLists.control_mode.index(control_cfg["control_mode"]))
            Control.AddWeight(builder, control_cfg["control_net_weight"])
            Control.AddGuidanceStart(builder, 0.0)
            Control.AddGuidanceEnd(builder, 1.0)
            fin_control = Control.End(builder)
            fin_controls.append(fin_control)

        GenerationConfiguration.StartControlsVector(builder, len(control_net["control_nets"]))
        for i, control_cfg in enumerate(control_net["control_nets"]):
            builder.PrependUOffsetTRelative(fin_controls[i])
        controls = builder.EndVector()

    start_width = width // 64 // scale_factor
    start_height = height // 64 // scale_factor
    model_name = builder.CreateString(model)
    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddModel(builder, model_name)
    GenerationConfiguration.AddBatchCount(builder, batch_count)
    GenerationConfiguration.AddBatchSize(builder, 1)
    GenerationConfiguration.AddSampler(builder, DrawThingsLists.sampler_list.index(sampler_name))
    GenerationConfiguration.AddSeedMode(builder, DrawThingsLists.seed_mode.index("ScaleAlike"))
    GenerationConfiguration.AddSteps(builder, steps)
    GenerationConfiguration.AddSeed(builder, seed)
    GenerationConfiguration.AddStartWidth(builder, start_width)
    GenerationConfiguration.AddStartHeight(builder, start_height)
    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    GenerationConfiguration.AddStrength(builder, strength)
    GenerationConfiguration.AddTargetImageWidth(builder, width)
    GenerationConfiguration.AddTargetImageHeight(builder, height)
    GenerationConfiguration.AddSharpness(builder, 0.6)
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
        img2img = bytes(hashlib.sha256(image_to_base64(image)).digest())
        contents.append(base64.b64decode(image_to_base64(image)))
    if mask is not None:
        maskimg = bytes(hashlib.sha256(image_to_base64(mask)).digest())
        contents.append(base64.b64decode(image_to_base64(mask)))

    models_override = [{
        "default_scale": 8,
        "file": "sd_v1.5_f16.ckpt",
        "name": "Generic (Stable Diffusion v1.5)",
        "prefix": "",
        "upcast_attention": False,
        "version": "v1"}]
    override = imageService_pb2.MetadataOverride(
        models = bytes(f"{models_override}", encoding='utf-8'),
        loras = b"[]",
        controlNets = b"[]",
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
                        tensor = bytes(hashlib.sha256(image_to_base64(image)).digest()),
                        weight = control_cfg["control_net_weight"]
                    )
                )
                contents.append(base64.b64decode(image_to_base64(image)))

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
            override = override,                  # Override the existing metadata on various Zoo objects.
            user = "ComfyUI",                     # The name of the client.
            device = "LAPTOP",                    # The type of the device uses.
            contents = contents                   # The image data as array of bytes. It is addressed by its sha256 content. This is modeled as content-addressable storage.
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
                    # Convert the image data to a Pillow Image object
                    img = Image.frombytes('RGB', (64, 64), response.previewImage, 'raw')
                prepare_callback(current_step, steps, img)

            if generated_images:
                images = []
                for img_data in response.generatedImages:
                    # Convert the image data to a Pillow Image object
                    img = Image.frombytes('RGB', (width, height), img_data, 'raw')
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
        DrawThingsLists.files_list = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
        return {
            "required": {
                "server": ("STRING", {"multiline": False, "default": DrawThingsLists.dtserver, "tooltip": "The IP address of the Draw Things gRPC Server."}),
                "port": ("STRING", {"multiline": False, "default": DrawThingsLists.dtport, "tooltip": "The port that the Draw Things gRPC Server is listening on."}),
                "model": (DrawThingsLists.files_list, {"default": "Press R to (re)load this list", "tooltip": "The model used for denoising the input latent. Please note that this lists all files, so be sure to pick the right one. Press R to (re)load this list."}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01, "tooltip": "When generating from an image, a high value allows more artistic freedom from the original. 1.0 means no influence from the existing image (a.k.a. text to image)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
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
    # def IS_CHANGED(s, control_net_name, **kwargs):
    #     DrawThingsSampler.files_list = get_files(DrawThingsSampler.dtserver, DrawThingsSampler.dtport)
    #     control_net_name = DrawThingsSampler.files_list
    #     print("IS_CHANGED")
    #     return float("NaN")
    # setattr(self.__class__, 'IS_CHANGED', IS_CHANGED)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsControlNet:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        DrawThingsLists.files_list = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
        return {
            "required": { 
                "control_net_name": (DrawThingsLists.files_list, {"default": "Press R to (re)load this list", "tooltip": "The model used. Please note that this lists all files, so be sure to pick the right one. Press R to (re)load this list."}),
                "control_input_type": (DrawThingsLists.control_input_type, {"default": "Unspecified"}),
                "control_mode": (DrawThingsLists.control_mode, {"default": "Balanced", "tooltip": ""}),
                "control_net_weight": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.50, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
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

    def add_to_pipeline(self, control_net_name, control_input_type, control_mode, control_net_weight, control_net={}, image=None ):
        # Check if 'control_nets' exists in the pipeline
        if "control_nets" not in control_net:
            # Create 'control_nets' as an empty list
            control_net["control_nets"] = []
        # Append the new entry as a dictionary to the list
        control_net["control_nets"].append({
            "control_net_name": control_net_name,
            "control_input_type": control_input_type,
            "control_mode": control_mode,
            "control_net_weight": control_net_weight,
            "image": image
        })
        return (control_net,)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsLoRA:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        DrawThingsLists.files_list = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
        return {
            "required": { 
                "lora_name": (DrawThingsLists.files_list, {"default": "Press R to (re)load this list", "tooltip": "The model used. Please note that this lists all files, so be sure to pick the right one. Press R to (re)load this list."}),
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
