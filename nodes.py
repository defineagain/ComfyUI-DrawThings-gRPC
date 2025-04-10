#!../../.venv python3

import os
import sys
import base64
import numpy as np
from PIL import Image, ImageOps
import io
from typing import TypedDict
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
import json
import struct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import latent_preview
import comfy.latent_formats as latent_formats
import comfy_execution.graph_utils as graph_utils
from comfy_execution.graph_utils import GraphBuilder

from server import PromptServer
from aiohttp import web

ModelInfo = TypedDict('ModelInfo', {
    'file': str,
    'name': str,
    'version': str,
    'prefix': str
})
ControlNetInfo = TypedDict('ControlNetInfo', {
    'file': str,
    'name': str,
    'version': str,
    'modifier': str,
    'type': str
})
LoRAInfo = TypedDict('LoRAInfo', {
    'file': str,
    'name': str,
    'version': str,
    'prefix': str
})
ModelsInfo = TypedDict('ModelsInfo', {
    'models': list[ModelInfo],
    'controlNets': list[ControlNetInfo],
    'loras': list[LoRAInfo]
})

MAX_RESOLUTION=16384
MAX_PREVIEW_RESOLUTION = args.preview_size

def prepare_callback(step, total_steps, x0: torch.Tensor, latent_format):
    previewer = None
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    if x0 is not None:
        previewer = latent_preview.get_previewer(x0.device, latent_format)

    pbar = comfy.utils.ProgressBar(step)
    def callback(step, x0: torch.Tensor, total_steps):

        preview_bytes = None
        if previewer is not None:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step, total_steps, preview_bytes)
    return callback(step, x0, total_steps)

def convert_response_image(response_image: bytes):
    int_buffer = np.frombuffer(response_image, dtype=np.uint32, count=17)
    height, width, channels = int_buffer[6:9]

    offset = 68
    length = width * height * channels * 2

    # print(f"Received image is {width}x{height} with {channels} channels")
    # print(f"Input size: {len(response_image)} (Expected: {length + 68})")

    data = np.frombuffer(response_image, dtype=np.float16, count=length // 2, offset=offset)
    if np.isnan(data[0]):
        print("NaN detected in data")
        return None
    data = np.clip((data + 1) * 127, 0, 255).astype(np.uint8)

    return {
        'data': data,
        'width': width,
        'height': height,
        'channels': channels,
    }

def convert_image_for_request(image_tensor: torch.Tensor, control_type=None):
# Draw Things: C header + the Float16 blob of -1 to 1 values that represents the image (in RGB order and HWC format, meaning r(0, 0), g(0, 0), b(0, 0), r(1, 0), g(1, 0), b(1, 0) .... (r(x, y) represents the value of red at that particular coordinate). The actual header is a bit more complex, here is the reference: https://github.com/liuliu/s4nnc/blob/main/nnc/Tensor.swift#L1750 the ccv_nnc_tensor_param_t is here: https://github.com/liuliu/ccv/blob/unstable/lib/nnc/ccv_nnc_tfb.h#L79 The type is CCV_TENSOR_CPU_MEMORY, format is CCV_TENSOR_FORMAT_NHWC, datatype is CCV_16F (for Float16), dim is the dimension in N, H, W, C order (in the case it should be 1, actual height, actual width, 3).

# ComfyUI: An IMAGE is a torch.Tensor with shape [B,H,W,C], C=3. If you are going to save or load images, you will need to convert to and from PIL.Image format - see the code snippets below! Note that some pytorch operations offer (or expect) [B,C,H,W], known as ‘channel first’, for reasons of computational efficiency. Just be careful.
# A LATENT is a dict; the latent sample is referenced by the key samples and has shape [B,C,H,W], with C=4.

    width = image_tensor.size(dim=2)
    height = image_tensor.size(dim=1)
    channels = image_tensor.size(dim=3)
    # print(f"Request image tensor is {width}x{height} with {channels} channels")

    image_tensor = image_tensor.to(torch.float16)

    image_tensor = image_tensor.permute(3, 1, 2, 0).squeeze(3)
    transform = torchvision.transforms.ToPILImage()
    pil_image = transform(image_tensor)

    match control_type:
        case "depth": # what else?
            transform = torchvision.transforms.Grayscale(num_output_channels=1)
            pil_image = transform(pil_image)
            # print(f"Converted request image is {pil_image.size}, {pil_image.mode}")
            channels = 1

    CCV_TENSOR_CPU_MEMORY = 0x1
    CCV_TENSOR_FORMAT_NHWC = 0x02
    CCV_16F = 0x20000

    image_bytes = bytearray(68 + width * height * channels * 2)
    struct.pack_into("<9I", image_bytes, 0, 0, CCV_TENSOR_CPU_MEMORY, CCV_TENSOR_FORMAT_NHWC, CCV_16F, 0, 1, height, width, channels)

    for y in range(height):
        for x in range(width):
            pixel = pil_image.getpixel((x, y))
            offset = 68 + (y * width + x) * (channels * 2)
            for c in range(channels):
                if channels == 1:
                    v = pixel / 255 * 2 - 1
                else:
                    v = pixel[c] / 255 * 2 - 1
                struct.pack_into("<e", image_bytes, offset + c * 2, v)

    return bytes(image_bytes)

def get_files(server, port) -> ModelsInfo:
    with grpc.insecure_channel(f"{server}:{port}") as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        response = stub.Echo(imageService_pb2.EchoRequest(name="ComfyUI"))
        response_json = json.loads(pb.json_format.MessageToJson(response))
        DrawThingsSampler.files_list = response_json["files"]
        override = dict(response_json['override'])
        model_info = { k: json.loads(str(base64.b64decode(override[k]), 'utf8')) for k in override.keys() }
        return model_info

routes = PromptServer.instance.routes
@routes.post('/dt_grpc_files_info')
async def handle_files_info_request(request):
    # if 'server' not in request.args or 'port' not in request.args:
    #     return web.json_response({"error": "Missing server or port parameter"}, status=400)
    # server = request.args['server']
    # port = request.args['port']
    try:
        post = await request.post()
        server = post.get('server')
        port = post.get('port')
        if server is None or port is None:
            return web.json_response({"error": "Missing server or port parameter"}, status=400)
        all_files = get_files(server, port)
        return web.json_response(all_files)
    except:
        return web.json_response({"error": "Could not connect to Draw Things gRPC server. Please check the server address and port."}, status=500)

async def dt_sampler(
                server,
                port,
                model,
                preview_type,
                seed,
                seed_mode,
                steps,
                cfg,
                strength,
                sampler_name,
                shift,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                positive,
                negative,
                width,
                height,
                batch_count=1,
                scale_factor=1,
                image=None,
                mask=None,
                control_net=None,
                lora=None,
                refiner=None,
                high_res_fix=None,
                video=None,
                upscaler=None,
                ) -> None:

    builder = flatbuffers.Builder(0)

    loras_out = None
    if lora is not None and len(lora):
        fin_loras = []
        for l in lora:
            print(l)
            lora_file = builder.CreateString(l['file'])
            LoRA.Start(builder)
            LoRA.AddFile(builder, lora_file)
            LoRA.AddWeight(builder, l['weight'])
            fin_lora = LoRA.End(builder)
            fin_loras.append(fin_lora)

        GenerationConfiguration.StartLorasVector(builder, len(fin_loras))
        for fl in fin_loras:
            builder.PrependUOffsetTRelative(fl)
        loras_out = builder.EndVector()

    controls_out = None
    if control_net is not None and len(control_net):
        fin_controls = []
        for c in control_net:
            control_name = builder.CreateString(c["file"])
            Control.Start(builder)
            Control.AddFile(builder, control_name)
            Control.AddInputOverride(builder, DrawThingsLists.control_input_type.index(c["input_type"]))
            Control.AddControlMode(builder, DrawThingsLists.control_mode.index(c["mode"]))
            Control.AddWeight(builder, c["weight"])
            Control.AddGuidanceStart(builder, c["start"])
            Control.AddGuidanceEnd(builder, c["end"])
            Control.AddNoPrompt(builder, False)
            Control.AddGlobalAveragePooling(builder, False)
            Control.AddDownSamplingRate(builder, 0)
            Control.AddTargetBlocks(builder, 0)
            fin_control = Control.End(builder)
            fin_controls.append(fin_control)

        GenerationConfiguration.StartControlsVector(builder, len(fin_controls))
        for fc in fin_controls:
            builder.PrependUOffsetTRelative(fc)
        controls_out = builder.EndVector()

    start_width = width // 64 // scale_factor
    start_height = height // 64 // scale_factor
    model_name = builder.CreateString(model)
    if upscaler is not None:
        upscaler_model = builder.CreateString(upscaler["upscaler_model"])
    if refiner is not None:
        refiner_model = builder.CreateString(refiner["refiner_model"])
    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddModel(builder, model_name)
    GenerationConfiguration.AddStrength(builder, strength)
    GenerationConfiguration.AddSeed(builder, seed)
    GenerationConfiguration.AddSeedMode(builder, DrawThingsLists.seed_mode.index(seed_mode))
    GenerationConfiguration.AddStartWidth(builder, start_width)
    GenerationConfiguration.AddStartHeight(builder, start_height)
    GenerationConfiguration.AddTargetImageWidth(builder, width)
    GenerationConfiguration.AddTargetImageHeight(builder, height)
    if upscaler is not None:
        GenerationConfiguration.AddUpscaler(builder, upscaler_model)
        GenerationConfiguration.AddUpscalerScaleFactor(builder, upscaler["scale_factor"])
    GenerationConfiguration.AddSteps(builder, steps)

    if video is not None:
        GenerationConfiguration.AddNumFrames(builder, video["num_frames"])

    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    # GenerationConfiguration.AddSpeedUpWithGuidanceEmbed(builder, True) # flux dev option
    GenerationConfiguration.AddSampler(builder, DrawThingsLists.sampler_list.index(sampler_name))
    # res shift # flux dev option
    GenerationConfiguration.AddShift(builder, shift)
    GenerationConfiguration.AddBatchSize(builder, 1)
    if refiner is not None:
        GenerationConfiguration.AddRefinerModel(builder, refiner_model)
        GenerationConfiguration.AddRefinerStart(builder, refiner["refiner_start"])
    # zero neg
    # sep clip
    GenerationConfiguration.AddClipSkip(builder, clip_skip)
    GenerationConfiguration.AddSharpness(builder, sharpness)
    GenerationConfiguration.AddMaskBlur(builder, mask_blur)
    GenerationConfiguration.AddMaskBlurOutset(builder, mask_blur_outset)
    GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, preserve_original)
    # face restore
    if high_res_fix is not None:
        GenerationConfiguration.AddHiresFix(builder, True)
        GenerationConfiguration.AddHiresFixStartWidth(builder, high_res_fix["high_res_fix_start_width"])
        GenerationConfiguration.AddHiresFixStartHeight(builder, high_res_fix["high_res_fix_start_height"])
        GenerationConfiguration.AddHiresFixStrength(builder, high_res_fix["high_res_fix_strength"])
    else:
        GenerationConfiguration.AddHiresFix(builder, False)

    GenerationConfiguration.AddTiledDecoding(builder, False)
    GenerationConfiguration.AddTiledDiffusion(builder, False)

    # if tea_cache is not None: # flux or video option
        # GenerationConfiguration.AddTeaCache(builder, False)
        # GenerationConfiguration.AddTeaCacheStart(builder, 5)
        # GenerationConfiguration.AddTeaCacheEnd(builder, -1)
        # GenerationConfiguration.AddTeaCacheThreshold(builder, 0.06)

    # ti embed
    GenerationConfiguration.AddBatchCount(builder, batch_count)
    if controls_out is not None:
        GenerationConfiguration.AddControls(builder, controls_out)
    if loras_out is not None:
        GenerationConfiguration.AddLoras(builder, loras_out)
    builder.Finish(GenerationConfiguration.End(builder))
    configuration = builder.Output()
    # generated = GenerationConfiguration.GenerationConfiguration.GetRootAs(configuration, 0)

    contents = []
    img2img = None
    maskimg = None
    if image is not None:
        img2img = convert_image_for_request(image)
    if mask is not None:
        maskimg = convert_image_for_request(mask)

    override = imageService_pb2.MetadataOverride()
    # models_override = [{
    #     "default_scale": 8,
    #     "file": "sd_v1.5_f16.ckpt",
    #     "name": "Generic (Stable Diffusion v1.5)",
    #     "prefix": "",
    #     "upcast_attention": False,
    #     "version": "v1"}]
    # override.models = bytes(f"{models_override}", encoding='utf-8')
    # override.loras = b'["hyper_sd_v1.x_4_step_lora_f16.ckpt"]'
    # override.controlNets = b'["controlnet_depth_1.x_v1.1_f16.ckpt"]'
    # override.textualInversions = b"[]"
    # override.upscalers = b"[]"

    hints = []
    if control_net is not None:
        for control_cfg in control_net:
            control_image = control_cfg["image"]
            if control_image is not None:
                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(control_image, control_cfg["input_type"].lower())
                taw.weight = control_cfg["weight"]

                hnt = imageService_pb2.HintProto()
                hnt.hintType = control_cfg["input_type"].lower()
                hnt.tensors.append(taw)
                hints.append(hnt)

    options = [["grpc.max_send_message_length", -1], ["grpc.max_receive_message_length", -1]]

    async with grpc.aio.insecure_channel(f"{server}:{port}", options) as channel:
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
            # print(f"current_step: {current_step}")

            if current_step:
                img = None
                if preview_image:
# ComfyUI: An IMAGE is a torch.Tensor with shape [B,H,W,C], C=3. If you are going to save or load images, you will need to convert to and from PIL.Image format - see the code snippets below! Note that some pytorch operations offer (or expect) [B,C,H,W], known as ‘channel first’, for reasons of computational efficiency. Just be careful.
# A LATENT is a dict; the latent sample is referenced by the key samples and has shape [B,C,H,W], with C=4.
                    x0 = None
                    latent_format = None
                    result = convert_response_image(preview_image)
                    if result is not None:
                        data = result['data']
                        width = result['width']
                        height = result['height']
                        channels = result['channels']

                        np_array = data.reshape(-1, channels, height, width)
                        x0 = torch.from_numpy(np_array).to(torch.float32)
                        # print(f"{x0.shape}")

                        match preview_type:
                            case "SD1.5":
                                latent_format = latent_formats.SD15(latent_formats.LatentFormat)
                            case "SD3":
                                latent_format = latent_formats.SD3(latent_formats.LatentFormat)
                            case "SDXL":
                                latent_format = latent_formats.SDXL(latent_formats.LatentFormat)
                            case "Flux":
                                latent_format = latent_formats.Flux()
                prepare_callback(current_step, steps, x0, latent_format)

            if generated_images:
                images = []
                for img_data in response.generatedImages:
                    # Convert the image data to a Pillow Image object
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
                        # print(f"size: {img.size}, mode: {img.mode}")
                        image_np = np.array(img)
                        # Convert to float32 tensor and normalize
                        tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
                        images.append(tensor_image)
                return (torch.stack(images),)

class DrawThingsLists:
    dtserver = "localhost"
    dtport = "7859"

    empty_models = ['No connection. Check server and try again', 'Click to rety']

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
        # NOTE: Draw Things currently only supports these input slots
        # Any other controlnet needs to use "Custom"
                # "Unspecified",
                "Custom",
                "Depth",
                # "Canny",
                "Scribble",
                "Pose",
                # "Normalbae",
                "Color",
                # "Lineart",
                # "Softedge",
                # "Seg",
                # "Inpaint",
                # "Ip2p",
                # "Shuffle",
                # "Mlsd",
                # "Tile",
                # "Blur",
                # "Lowquality",
                # "Gray",
            ]

    modeltype_list = [
                "SD1.5",
                "SD3",
                "SDXL",
                "Flux",
            ]

class DrawThingsSampler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "server": ("STRING", {"multiline": False, "default": DrawThingsLists.dtserver, "tooltip": "The IP address of the Draw Things gRPC Server."}),
                "port": ("STRING", {"multiline": False, "default": DrawThingsLists.dtport, "tooltip": "The port that the Draw Things gRPC Server is listening on."}),
                "model": ("DT_MODEL", {"model_type": "models", "tooltip": "The model used for denoising the input latent."}),

                # TODO: Remove the need to manually set preview type
                "preview_type": (DrawThingsLists.modeltype_list, {"default": "SD1.5"}),

                "strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01, "tooltip": "When generating from an image, a high value allows more artistic freedom from the original. 1.0 means no influence from the existing image (a.k.a. text to image)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "seed_mode": (DrawThingsLists.seed_mode, {"default": "ScaleAlike"}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                # speedup flux
                "sampler_name": (DrawThingsLists.sampler_list, {"default": "DPMPP 2M Trailing", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                # res shift
                "shift": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 8.00, "step": 0.01, "round": 0.01}),

                # zero neg
                # sep clip
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 23, "step": 1}),
                "sharpness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.1}),
                "mask_blur": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1, "round": 0.1}),
                "mask_blur_outset": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "preserve_original": ("BOOLEAN", {"default": True}),
                # face restore

                # ti embed
            },
            "hidden": {
                "scale_factor": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 1, "step": 1}),
            },
            "optional": {
                "positive": ("STRING", {
                    "multiline": True, "default": "a lovely cat", "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("STRING", {
                    "multiline": True, "default": "text, watermark", "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "lora": ("DT_LORA", ),
                "control_net": ("DT_CNET", ),
                "upscaler": ("DT_UPSCALER", ),
                "video": ("DT_VIDEO", ),
                "refiner": ("DT_REFINER", ),
                "high_res_fix": ("DT_HIGHRES", ),
                # "tiled_decoding": ("BOOLEAN", {"default": False}),
                # "tiled_diffusion": ("BOOLEAN", {"default": False}),
                # "tea_cache": ("DT_TEA", ),
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
                preview_type,
                seed,
                seed_mode,
                steps,
                cfg,
                strength,
                sampler_name,
                shift,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                positive,
                negative,
                width,
                height,
                batch_count=1,
                scale_factor=1,
                image=None,
                mask=None,
                control_net=None,
                lora=None,
                refiner=None,
                high_res_fix=None,
                video=None,
                upscaler=None,
                ):

        # need to replace model NAMES with model FILES

        all_files = get_files(server, port)
        model_file = next((m['file'] for m in all_files["models"] if m['name'] == model), None)

        if lora is not None:
            for lora_item in lora:
                lora_item['file'] = next((m['file'] for m in all_files['loras'] if m['name'] == lora_item['name']), None)

        if control_net is not None:
            for cnet in control_net:
                cnet['file'] = next((m['file'] for m in all_files['controlNets'] if m['name'] == cnet['name']), None)

        return asyncio.run(dt_sampler(
                server,
                port,
                model_file,
                preview_type,
                seed,
                seed_mode,
                steps,
                cfg,
                strength,
                sampler_name,
                shift,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                positive,
                negative,
                width,
                height,
                batch_count=batch_count,
                scale_factor=scale_factor,
                image=image,
                mask=mask,
                control_net=control_net,
                lora=lora,
                refiner=refiner,
                high_res_fix=high_res_fix,
                video=video,
                upscaler=upscaler,
                ))

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsRefiner:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        def get_filtered_files():
            file_list = ["Press R to (re)load this list"]
            try:
                all_files = get_files(DrawThingsLists.dtserver, DrawThingsLists.dtport)
            except:
                file_list.insert(0, "Could not connect to Draw Things gRPC server. Please check the server address and port.")
            else:
                file_list.extend([f['name'] for f in all_files['models']])
            return file_list

        return {
            "required": {
                "refiner_model": (get_filtered_files(), {"default": "Press R to (re)load this list", "tooltip": "The model used for denoising the input latent.\nPlease note that this lists all files, so be sure to pick the right one.\nPress R to (re)load this list."}),
                "refiner_start": ("FLOAT", {"default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("DT_REFINER",)
    RETURN_NAMES = ("refiner",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, refiner_model, refiner_start):
        refiner = {"refiner_model": refiner_model, "refiner_start": refiner_start}
        return (refiner,)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsHighResFix:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_res_fix_start_width": ("INT", {"default": 448, "min": 128, "max": 2048, "step": 64}),
                "high_res_fix_start_height": ("INT", {"default": 448, "min": 128, "max": 2048, "step": 64}),
                "high_res_fix_strength": ("FLOAT", {"default": 0.70, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("DT_HIGHRES",)
    RETURN_NAMES = ("high_res_fix",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, high_res_fix_start_width, high_res_fix_start_height, high_res_fix_strength):
        high_res_fix = {"high_res_fix_start_width": high_res_fix_start_width, "high_res_fix_start_height": high_res_fix_start_height, "high_res_fix_strength": high_res_fix_strength}
        return (high_res_fix,)

class DrawThingsVideo:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_frames": ("INT", {"default": 14, "min": 1, "max": 81, "step": 1}),
            }
        }

    RETURN_TYPES = ("DT_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, num_frames):
        # as dict in case more options are needed later
        video = {"num_frames": num_frames}
        return (video,)

class DrawThingsUpscaler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # TODO: Add modellist
                "upscaler_model": ("STRING", {"default": "Under construction"}),
                "scale_factor": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("DT_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, upscaler_model, scale_factor):
        upscaler = {"upscaler_model": upscaler_model, "scale_factor": scale_factor}
        # return (upscaler,)
        return (None,)

class DrawThingsPositive:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {
                    "multiline": True, "default": "a lovely cat", "tooltip": "The conditioning describing the attributes you want to include in the image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "prompt"
    CATEGORY = "DrawThings"

    def prompt(self, positive):
        return (positive,)

class DrawThingsNegative:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative": ("STRING", {
                    "multiline": True, "default": "text, watermark", "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "prompt"
    CATEGORY = "DrawThings"

    def prompt(self, negative):
        return (negative,)

class DrawThingsControlNet:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_name": ("DT_MODEL", {"model_type": "controlNets", "tooltip": "The model used."}),
                "control_input_type": (DrawThingsLists.control_input_type, {"default": "Unspecified", "tooltip": "Draw Things currently only supports these input slots, any other controlnet needs to use 'Custom'"}),
                "control_mode": (DrawThingsLists.control_mode, {"default": "Balanced", "tooltip": ""}),
                "control_weight": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.50, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "control_start": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "invert_image": ("BOOLEAN", {"default": False, "tooltip": "Some Control Nets (i.e. LineArt) need their image to be inverted."}),
            },
            "optional": {
                "control_net": ("DT_CNET",),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("DT_CNET",)
    RETURN_NAMES = ("control_net",)
    CATEGORY = "DrawThings"
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, control_name, control_input_type, control_mode, control_weight, control_start, control_end, control_net=None, image=None, invert_image=False):
        if invert_image == True:
            image = 1.0 - image

        cnet_list = list()

        if control_net is not None:
            cnet_list.extend(control_net)

        cnet_list.append({
            "name": control_name,
            "input_type": control_input_type,
            "mode": control_mode,
            "weight": control_weight,
            "start": control_start,
            "end": control_end,
            "image": image
        })

        return (cnet_list,)

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
        return {
            "required": {
                "lora_name": ("DT_MODEL", {"model_type": "loras", "tooltip": "The model used."}),
                "lora_weight": ("FLOAT", {"default": 1.00, "min": -3.00, "max": 3.00, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            },
            "optional": {
                "lora": ("DT_LORA",),
            }
        }

    RETURN_TYPES = ("DT_LORA",)
    RETURN_NAMES = ("lora",)
    CATEGORY = "DrawThings"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, lora_name, lora_weight, lora=None):
        lora_list = list()

        if lora is not None:
            lora_list.extend(lora)

        lora_list.append({"name": lora_name, "weight": lora_weight})

        return (lora_list,)

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
    "DrawThingsPositive": DrawThingsPositive,
    "DrawThingsNegative": DrawThingsNegative,
    "DrawThingsRefiner": DrawThingsRefiner,
    "DrawThingsHighResFix": DrawThingsHighResFix,
    "DrawThingsVideo": DrawThingsVideo,
    "DrawThingsUpscaler": DrawThingsUpscaler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawThingsSampler": "Draw Things Sampler",
    "DrawThingsControlNet": "Draw Things Control Net",
    "DrawThingsLoRA": "Draw Things LoRA",
    "DrawThingsPositive": "Draw Things Positive Prompt",
    "DrawThingsNegative": "Draw Things Negative Prompt",
    "DrawThingsRefiner": "Draw Things Refiner",
    "DrawThingsHighResFix": "Draw Things High Resolution Fix",
    "DrawThingsVideo": "Draw Things Video Options",
    "DrawThingsUpscaler": "Draw Things Upscaler",
}
