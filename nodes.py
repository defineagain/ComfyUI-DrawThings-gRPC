#!../../.venv python3

import os
import sys
import base64
import re
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
# from generated import imageService_pb2
from .generated import imageService_pb2, imageService_pb2_grpc
from .generated import Control
from .generated import LoRA
from .generated import GenerationConfiguration
import json
import struct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.utils
from comfy.cli_args import args
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
UpscalerInfo = TypedDict('UpscalerInfo', {
    'file': str,
    'name': str,
})
ModelsInfo = TypedDict('ModelsInfo', {
    'models': list[ModelInfo],
    'controlNets': list[ControlNetInfo],
    'loras': list[LoRAInfo],
    'upscalers': list[UpscalerInfo]
})

MAX_RESOLUTION=16384
MAX_PREVIEW_RESOLUTION = args.preview_size

def prepare_callback(step, total_steps, x0=None):
    pbar = comfy.utils.ProgressBar(step)
    def callback(step, total_steps, x0=None):

        preview_bytes = None
        if x0 is not None:
            preview_bytes = ("PNG", x0, MAX_PREVIEW_RESOLUTION)
        pbar.update_absolute(step, total_steps, preview_bytes)
    return callback(step, total_steps, x0)

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

def decode_preview(preview, version):
    int_buffer = np.frombuffer(preview, dtype=np.uint32, count=17)
    image_height, image_width, channels = int_buffer[6:9]

    if channels not in {3, 4, 16}:
        return None

    offset = 68
    length = image_width * image_height * channels * 2

    # print(f"Received image is {image_width}x{image_height} with {channels} channels")
    # print(f"Input size: {len(preview)} (Expected: {length + 68})")

    fp16 = np.frombuffer(preview, dtype=np.float16, offset=offset)

    print(f'version: {version}')

    image = None
    if version in ['v1', 'v2', 'svdI2v']:
        bytes_array = np.zeros((image_height, image_width, channels), dtype=np.uint8)
        for i in range(image_height * image_width):
            v0, v1, v2, v3 = fp16[i * 4:i * 4 + 4]
            r = 49.5210 * v0 + 29.0283 * v1 - 23.9673 * v2 - 39.4981 * v3 + 99.9368
            g = 41.1373 * v0 + 42.4951 * v1 + 24.7349 * v2 - 50.8279 * v3 + 99.8421
            b = 40.2919 * v0 + 18.9304 * v1 + 30.0236 * v2 - 81.9976 * v3 + 99.5384

            bytes_array[i // image_width, i % image_width] = [
                min(max(int(r if np.isfinite(r) else 0), 0), 255),
                min(max(int(g if np.isfinite(g) else 0), 0), 255),
                min(max(int(b if np.isfinite(b) else 0), 0), 255),
                255
            ]
        image = Image.fromarray(bytes_array, 'RGBA')

    if version[:3] == 'sd3':
        bytes_array = bytearray(image_width * image_height * 4)
        for i in range(image_height * image_width):
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = \
                fp16[i * 16:(i + 1) * 16]
            r = (-0.0922 * v0 + 0.0311 * v1 + 0.1994 * v2 + 0.0856 * v3 + 0.0587 * v4 - 0.0006 * v5 + 0.0978 * v6 - 0.0042 * v7 - 0.0194 * v8 - 0.0488 * v9 + 0.0922 * v10 - 0.0278 * v11 + 0.0332 * v12 - 0.0069 * v13 - 0.0596 * v14 - 0.1448 * v15 + 0.2394) * 127.5 + 127.5
            g = (-0.0175 * v0 + 0.0633 * v1 + 0.0927 * v2 + 0.0339 * v3 + 0.0272 * v4 + 0.1104 * v5 + 0.0306 * v6 + 0.1038 * v7 + 0.0020 * v8 + 0.0130 * v9 + 0.0988 * v10 + 0.0524 * v11 + 0.0456 * v12 - 0.0030 * v13 - 0.0465 * v14 - 0.1463 * v15 + 0.2135) * 127.5 + 127.5
            b = (0.0749 * v0 + 0.0954 * v1 + 0.0458 * v2 + 0.0902 * v3 - 0.0496 * v4 + 0.0309 * v5 + 0.0427 * v6 + 0.1358 * v7 + 0.0669 * v8 - 0.0268 * v9 + 0.0951 * v10 - 0.0542 * v11 + 0.0895 * v12 - 0.0810 * v13 - 0.0293 * v14 - 0.1189 * v15 + 0.1925) * 127.5 + 127.5

            bytes_array[i * 4] = max(min(int(r), 255), 0)
            bytes_array[i * 4 + 1] = max(min(int(g), 255), 0)
            bytes_array[i * 4 + 2] = max(min(int(b), 255), 0)
            bytes_array[i * 4 + 3] = 255
        image = Image.frombytes('RGBA', (image_width, image_height), bytes(bytes_array))

    if version[:4] == 'sdxl' or version in ['ssd1b', 'pixart', 'auraflow']:
        bytes_array = np.zeros((image_height * image_width * 4,), dtype=np.uint8)
        for i in range(image_height * image_width):
            v0, v1, v2, v3 = fp16[i * 4:(i + 1) * 4]
            r = 47.195 * v0 - 29.114 * v1 + 11.883 * v2 - 38.063 * v3 + 141.64
            g = 53.237 * v0 - 1.4623 * v1 + 12.991 * v2 - 28.043 * v3 + 127.46
            b = 58.182 * v0 + 4.3734 * v1 - 3.3735 * v2 - 26.722 * v3 + 114.5

            bytes_array[i * 4] = max(min(int(r), 255), 0)
            bytes_array[i * 4 + 1] = max(min(int(g), 255), 0)
            bytes_array[i * 4 + 2] = max(min(int(b), 255), 0)
            bytes_array[i * 4 + 3] = 255
        image = Image.frombytes('RGBA', (image_width, image_height), bytes_array)

    if version[:4] == 'flux':
        bytes_array = np.zeros((image_height * image_width * 4,), dtype=np.uint8)
        for i in range(image_height * image_width):
            v = fp16[i * 16:i * 16 + 16]
            r = (-0.0346 * v[0] + 0.0034 * v[1] + 0.0275 * v[2] - 0.0174 * v[3] +
                    0.0859 * v[4] + 0.0004 * v[5] + 0.0405 * v[6] - 0.0236 * v[7] -
                    0.0245 * v[8] + 0.1008 * v[9] - 0.0515 * v[10] + 0.0428 * v[11] +
                    0.0817 * v[12] - 0.1264 * v[13] - 0.0280 * v[14] - 0.1262 * v[15] - 0.0329) * 127.5 + 127.5
            g = (0.0244 * v[0] + 0.0210 * v[1] - 0.0668 * v[2] + 0.0160 * v[3] +
                    0.0721 * v[4] + 0.0383 * v[5] + 0.0861 * v[6] - 0.0185 * v[7] +
                    0.0250 * v[8] + 0.0755 * v[9] + 0.0201 * v[10] - 0.0012 * v[11] +
                    0.0765 * v[12] - 0.0522 * v[13] - 0.0881 * v[14] - 0.0982 * v[15] - 0.0718) * 127.5 + 127.5
            b = (0.0681 * v[0] + 0.0687 * v[1] - 0.0433 * v[2] + 0.0617 * v[3] +
                    0.0329 * v[4] + 0.0115 * v[5] + 0.0915 * v[6] - 0.0259 * v[7] +
                    0.1180 * v[8] - 0.0421 * v[9] + 0.0011 * v[10] - 0.0036 * v[11] +
                    0.0749 * v[12] - 0.1103 * v[13] - 0.0499 * v[14] - 0.0778 * v[15] - 0.0851) * 127.5 + 127.5

            bytes_array[i * 4] = min(max(int(r) if np.isfinite(r) else 0, 0), 255)
            bytes_array[i * 4 + 1] = min(max(int(g) if np.isfinite(g) else 0, 0), 255)
            bytes_array[i * 4 + 2] = min(max(int(b) if np.isfinite(b) else 0, 0), 255)
            bytes_array[i * 4 + 3] = 255
        image = Image.fromarray(bytes_array.reshape((image_height, image_width, 4)), 'RGBA')

    if version[:3] == 'wan':
        bytes_array = np.zeros((image_height * image_width * 4,), dtype=np.uint8)
        for i in range(image_height * image_width):
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fp16[i*16:(i+1)*16]
            r = (-0.1299 * v0 + 0.0671 * v1 + 0.3568 * v2 + 0.0372 * v3 + 0.0313 * v4 + 0.0296 * v5
                    - 0.3477 * v6 + 0.0166 * v7 - 0.0412 * v8 - 0.1293 * v9 + 0.0680 * v10 + 0.0032
                    * v11 - 0.1251 * v12 + 0.0060 * v13 + 0.3477 * v14 + 0.1984 * v15 - 0.1835) * 127.5 + 127.5
            g = (-0.1692 * v0 + 0.0406 * v1 + 0.2548 * v2 + 0.2344 * v3 + 0.0189 * v4 - 0.0956 * v5
                    - 0.4059 * v6 + 0.1902 * v7 + 0.0267 * v8 + 0.0740 * v9 + 0.3019 * v10 + 0.0581
                    * v11 + 0.0927 * v12 - 0.0633 * v13 + 0.2275 * v14 + 0.0913 * v15 - 0.0868) * 127.5 + 127.5
            b = (0.2932 * v0 + 0.0442 * v1 + 0.1747 * v2 + 0.1420 * v3 - 0.0328 * v4 - 0.0665 * v5
                    - 0.2925 * v6 + 0.1975 * v7 - 0.1364 * v8 + 0.1636 * v9 + 0.1128 * v10 + 0.0639
                    * v11 + 0.1699 * v12 + 0.0005 * v13 + 0.2950 * v14 + 0.1861 * v15 - 0.336) * 127.5 + 127.5

            bytes_array[i*4] = max(0, min(255, int(r)))
            bytes_array[i*4+1] = max(0, min(255, int(g)))
            bytes_array[i*4+2] = max(0, min(255, int(b)))
            bytes_array[i*4+3] = 255
        image = Image.frombytes('RGBA', (image_width, image_height), bytes_array)

    if version == 'hunyuanVideo':
        bytes_array = np.zeros((image_height * image_width * 4,), dtype=np.uint8)
        for i in range(image_height * image_width):
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = fp16[i*16:(i+1)*16]
            r = (-0.0395 * v0 + 0.0696 * v1 + 0.0135 * v2 + 0.0108 * v3 - 0.0209 * v4 - 0.0804 * v5
                    - 0.0991 * v6 - 0.0646 * v7 - 0.0696 * v8 - 0.0799 * v9 + 0.1166 * v10 + 0.1165
                    * v11 - 0.2315 * v12 - 0.0270 * v13 - 0.0616 * v14 + 0.0249 * v15 + 0.0249) * 127.5 + 127.5
            g = (-0.0331 * v0 + 0.0795 * v1 - 0.0945 * v2 - 0.0250 * v3 + 0.0032 * v4 - 0.0254 * v5
                    + 0.0271 * v6 - 0.0422 * v7 - 0.0595 * v8 - 0.0208 * v9 + 0.1627 * v10 + 0.0432
                    * v11 - 0.1920 * v12 + 0.0401 * v13 - 0.0997 * v14 - 0.0469 * v15 - 0.0192) * 127.5 + 127.5
            b = (0.0445 * v0 + 0.0518 * v1 - 0.0282 * v2 - 0.0765 * v3 + 0.0224 * v4 - 0.0639 * v5
                    - 0.0669 * v6 - 0.0400 * v7 - 0.0894 * v8 - 0.0375 * v9 + 0.0962 * v10 + 0.0407
                    * v11 - 0.1355 * v12 - 0.0821 * v13 - 0.0727 * v14 - 0.1703 * v15 - 0.0761) * 127.5 + 127.5

            bytes_array[i*4] = max(0, min(255, int(r)))
            bytes_array[i*4+1] = max(0, min(255, int(g)))
            bytes_array[i*4+2] = max(0, min(255, int(b)))
            bytes_array[i*4+3] = 255
        image = Image.frombytes('RGBA', (image_width, image_height), bytes_array)

    if version[:5] == 'wurst':
        bytes_array = np.zeros((image_height * image_width * 4,), dtype=np.uint8)
        if channels == 3:
            for i in range(image_height * image_width):
                r, g, b = fp16[i * 3], fp16[i * 3 + 1], fp16[i * 3 + 2]
                bytes_array[i * 4] = max(min(int(r * 255), 255), 0)
                bytes_array[i * 4 + 1] = max(min(int(g * 255), 255), 0)
                bytes_array[i * 4 + 2] = max(min(int(b * 255), 255), 0)
                bytes_array[i * 4 + 3] = 255
        else:
            for i in range(image_height * image_width):
                v0, v1, v2, v3 = fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3]
                r = max(min(int(10.175 * v0 - 20.807 * v1 - 27.834 * v2 - 2.0577 * v3 + 143.39), 255), 0)
                g = max(min(int(21.07 * v0 - 4.3022 * v1 - 11.258 * v2 - 18.8 * v3 + 131.53), 255), 0)
                b = max(min(int(7.8454 * v0 - 2.3713 * v1 - 0.45565 * v2 - 41.648 * v3 + 120.76), 255), 0)

                bytes_array[i * 4] = r
                bytes_array[i * 4 + 1] = g
                bytes_array[i * 4 + 2] = b
                bytes_array[i * 4 + 3] = 255
        image = Image.frombytes('RGBA', (image_width, image_height), bytes_array)

    return image

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
                flux=None,
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
            if c["input_type"] not in ["Custom", "Depth", "Scribble", "Pose", "Color"]:
                c_input_slot = "Custom"
            else:
                c_input_slot = c["input_type"]
            print(f"{c_input_slot}")
            control_name = builder.CreateString(c["file"])
            Control.Start(builder)
            Control.AddFile(builder, control_name)
            Control.AddInputOverride(builder, DrawThingsLists.control_input_type.index(c_input_slot))
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
        GenerationConfiguration.AddUpscalerScaleFactor(builder, upscaler["upscaler_scale_factor"])
    GenerationConfiguration.AddSteps(builder, steps)

    if video is not None:
        GenerationConfiguration.AddNumFrames(builder, video["num_frames"])

    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    GenerationConfiguration.AddSampler(builder, DrawThingsLists.sampler_list.index(sampler_name))
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

    if video is not None: # flux or video option
        if video["tea_cache"] is not None: # flux or video option
            GenerationConfiguration.AddTeaCache(builder, True)
            GenerationConfiguration.AddTeaCacheStart(builder, video["tea_cache"]["tea_cache_start"])
            GenerationConfiguration.AddTeaCacheEnd(builder, video["tea_cache"]["tea_cache_end"])
            GenerationConfiguration.AddTeaCacheThreshold(builder, video["tea_cache"]["tea_cache_threshold"])

    if flux is not None: # flux or video option
        if flux["tea_cache"] is not None: # flux or video option
            GenerationConfiguration.AddTeaCache(builder, True)
            GenerationConfiguration.AddTeaCacheStart(builder, flux["tea_cache"]["tea_cache_start"])
            GenerationConfiguration.AddTeaCacheEnd(builder, flux["tea_cache"]["tea_cache_end"])
            GenerationConfiguration.AddTeaCacheThreshold(builder, flux["tea_cache"]["tea_cache_threshold"])

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
                # NOTE: So apparantly other cnets don't work if it's not one of these 5
                # TODO: Fix Union cnets
                if control_cfg["input_type"] not in ["Custom", "Depth", "Scribble", "Pose", "Color"]:
                    c_input_slot = "Custom"
                else:
                    c_input_slot = control_cfg["input_type"]
                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(control_image, c_input_slot.lower())
                taw.weight = control_cfg["weight"]

                hnt = imageService_pb2.HintProto()
                hnt.hintType = c_input_slot.lower()
                hnt.tensors.append(taw)
                hints.append(hnt)
    if lora is not None:
        # Needed for loras like FLUX.1-Depth-dev-lora
        for lora_cfg in lora:
            lora_image = None
            if "image" in lora_cfg:
                lora_image = lora_cfg["image"]
            if lora_image is not None:
                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(lora_image, lora_cfg["input_type"].lower())
                taw.weight = lora_cfg["weight"]

                hnt = imageService_pb2.HintProto()
                hnt.hintType = lora_cfg["input_type"].lower()
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
                x0 = None
                if preview_image:
                    modelinfo_version = DrawThingsLists.modelinfo_list["version"]
                    if modelinfo_version != "":
                        x0 = decode_preview(preview_image, modelinfo_version)
                prepare_callback(current_step, steps, x0)

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
        # NOTE: Draw Things currently only supports these input slots: Custom, Depth, Scribble, Pose, Color
        # But in order to have Union cnets working, we still need the full list to set the hints-type, only input-override has to be set to one of the slots.
                "Custom",
                "Depth",
                "Canny", # -> Custom
                "Scribble",
                "Pose",
                "Normalbae", # -> Custom
                "Color",
                "Lineart", # -> Custom
                "Softedge", # -> Custom
                "Seg", # -> Custom
                "Inpaint", # -> Custom
                "Ip2p", # -> Custom
                "Shuffle", # -> Custom
                "Mlsd", # -> Custom
                "Tile", # -> Custom
                "Blur", # -> Custom
                "Lowquality", # -> Custom
                "Gray", # -> Custom
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
                "flux": ("DT_FLUX", ),
                "video": ("DT_VIDEO", ),
                "refiner": ("DT_REFINER", ),
                "high_res_fix": ("DT_HIGHRES", ),
                # "tiled_decoding": ("BOOLEAN", {"default": False}),
                # "tiled_diffusion": ("BOOLEAN", {"default": False}),
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
                flux=None,
                ):

        # need to replace model NAMES with model FILES

        all_files = get_files(server, port)

        def getModelInfo(item, models):
            item_name = item['name'] if 'name' in item else item
            matches = re.match(r"^(.*) \((\w|\.)+\)$", item_name)
            name = matches[1] if matches else item_name
            return next((m for m in models if m['name'] == name), None)

        model_info = getModelInfo(model, all_files["models"])
        DrawThingsLists.modelinfo_list = model_info

        if lora is not None:
            for lora_item in lora:
                lora_item['file'] = getModelInfo(lora_item, all_files['loras'])['file']

        if control_net is not None:
            for cnet in control_net:
                cnet['file'] = getModelInfo(cnet, all_files['controlNets'])['file']

        return asyncio.run(dt_sampler(
                server,
                port,
                model_info['file'],
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
                flux=flux,
                ))

    # @classmethod
    # def IS_CHANGED(s, **kwargs):
    #     return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

class DrawThingsFlux:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "speed_up": ("BOOLEAN", {"default": True}),
                "res_dpt_shift": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "tea_cache": ("DT_TEA", ),
            }
        }

    RETURN_TYPES = ("DT_FLUX",)
    RETURN_NAMES = ("flux",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, speed_up, res_dpt_shift, tea_cache=None):
        flux = {"speed_up": speed_up, "res_dpt_shift": res_dpt_shift, "tea_cache": tea_cache}
        return (flux,)

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
            },
            "optional": {
                "tea_cache": ("DT_TEA", ),
            }
        }

    RETURN_TYPES = ("DT_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, num_frames, tea_cache=None):
        video = {"num_frames": num_frames, "tea_cache": tea_cache}
        return (video,)

class DrawThingsUpscaler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscaler_model": ("DT_MODEL", {"model_type": "upscalers"}),
                "upscaler_scale_factor": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("DT_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, upscaler_model, upscaler_scale_factor):
        upscaler = {"upscaler_model": upscaler_model, "upscaler_scale_factor": upscaler_scale_factor}
        return (upscaler,)

class DrawThingsTeaCache:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tea_cache_start": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "tea_cache_end": ("INT", {"default": 2, "min": 0, "max": 81, "step": 1}),
                "tea_cache_threshold": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("DT_TEA",)
    RETURN_NAMES = ("tea_cache",)
    FUNCTION = "add_to_pipeline"
    DESCRIPTION = "Connect this via a supported node, like Flux or Video."
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, tea_cache_start, tea_cache_end, tea_cache_threshold):
        tea_cache = {"tea_cache_start": tea_cache_start, "tea_cache_end": tea_cache_end, "tea_cache_threshold": tea_cache_threshold}
        return (tea_cache,)

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
                "control_input_type": (DrawThingsLists.control_input_type, {"default": "Custom"}),
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

class DrawThingsLoRANet:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "lora_name": ("DT_MODEL", {"model_type": "loras", "tooltip": "The model used."}),
                "lora_weight": ("FLOAT", {"default": 1.00, "min": -3.00, "max": 3.00, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "lora_input_type": (DrawThingsLists.control_input_type, {"default": "Custom"}),
                "invert_image": ("BOOLEAN", {"default": False, "tooltip": "Some Control Nets (i.e. LineArt) need their image to be inverted."}),
            },
            "optional": {
                "lora": ("DT_LORA",),
            }
        }

    RETURN_TYPES = ("DT_LORA",)
    RETURN_NAMES = ("lora",)
    CATEGORY = "DrawThings"
    DESCRIPTION = "Some LoRAs (i.e. FLUX.1-Depth-dev-lora) perform the function of Control Net and need to be supplied with an image and type. Use this node in that case."
    FUNCTION = "add_to_pipeline"

    def add_to_pipeline(self, lora_name, lora_weight, image, lora_input_type, invert_image, lora=None):
        if invert_image == True:
            image = 1.0 - image

        lora_list = list()

        if lora is not None:
            lora_list.extend(lora)

        lora_list.append({
            "name": lora_name, 
            "weight": lora_weight, 
            "image": image, 
            "input_type": lora_input_type
        })

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
    "DrawThingsTeaCache": DrawThingsTeaCache,
    "DrawThingsFlux": DrawThingsFlux,
    "DrawThingsLoRANet": DrawThingsLoRANet,
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
    "DrawThingsTeaCache": "Draw Things Tea Cache",
    "DrawThingsFlux": "Draw Things Flux Options",
    "DrawThingsLoRANet": "Draw Things LoRA Net",
}
