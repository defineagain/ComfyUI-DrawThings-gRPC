import base64
from google.protobuf.json_format import MessageToJson
import json
import grpc
import grpc.aio
import torch
import flatbuffers
from .. import cancel_request, settings
from .config import build_config
import numpy as np
from PIL import Image
from .credentials import credentials
from .data_types import ModelsInfo, UpscalerInfo
from .generated import imageService_pb2, imageService_pb2_grpc

from .image_handlers import (
    convert_image_for_request,
    convert_mask_for_request,
    convert_response_image,
    decode_preview,
    prepare_callback,
)


async def get_files(server, port, use_tls) -> ModelsInfo:
    async with get_aio_channel(server, port, use_tls) as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        response = await stub.Echo(imageService_pb2.EchoRequest(name="ComfyUI"))
        response_json = json.loads(MessageToJson(response))
        override = dict(response_json["override"])
        model_info = {
            k: json.loads(str(base64.b64decode(override[k]), "utf8"))
            for k in override.keys()
        }

        if "upscalers" not in model_info:
            official = [
                "realesrgan_x2plus_f16.ckpt",
                "realesrgan_x4plus_f16.ckpt",
                "realesrgan_x4plus_anime_6b_f16.ckpt",
                "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt",
                "remacri_4x_f16.ckpt",
                "4x_ultrasharp_f16.ckpt",
            ]
            model_info["upscalers"] = [UpscalerInfo(file=f, name=f) for f in official]

        return ModelsInfo(
            models=model_info["models"],
            controlNets=model_info["controlNets"],
            loras=model_info["loras"],
            upscalers=model_info["upscalers"],
            textualInversions=model_info["textualInversions"],
        )


def get_aio_channel(server, port, use_tls):
    options = [
        ["grpc.max_send_message_length", -1],
        ["grpc.max_receive_message_length", -1],
    ]
    if use_tls and credentials is not None:
        return grpc.aio.secure_channel(f"{server}:{port}", credentials, options=options)
    return grpc.aio.insecure_channel(f"{server}:{port}", options=options)


async def dt_sampler(inputs: dict):
    server, port, use_tls = (
        inputs.get("server"),
        inputs.get("port"),
        inputs.get("use_tls"),
    )
    positive, negative = inputs.get("positive"), inputs.get("negative")
    image, mask = inputs.get("image"), inputs.get("mask")
    version = inputs.get("version")

    config = build_config(inputs)
    width = config.startWidth * 64
    height = config.startHeight * 64

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
        img2img = convert_image_for_request(image, width=width, height=height)
    if mask is not None:
        maskimg = convert_mask_for_request(mask, width=width, height=height)

    hints = []
    cnets = inputs.get("control_net")
    if cnets is not None:
        for cnet in cnets:
            if cnet.get("image") is not None and cnet.get("input_type") is not None:
                taws = []
                for i in range(cnet["image"].size(dim=0)):
                    hint_tensor = convert_image_for_request(
                        cnet["image"],
                        cnet["input_type"].lower(),
                        batch_index=i,
                        width=width,
                        height=height,
                    )
                    taw = imageService_pb2.TensorAndWeight()
                    taw.weight = 1
                    taw.tensor = hint_tensor
                    taws.append(taw)

                hnt = imageService_pb2.HintProto()
                hnt.hintType = cnet["input_type"].lower()
                hnt.tensors.extend(taws)
                hints.append(hnt)

    lora = inputs.get("lora")
    if lora is not None:
        for lora_cfg in lora:
            if "control_image" in lora_cfg:
                modifier = lora_cfg["model"]["modifier"]

                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(
                    lora_cfg["control_image"],
                    control_type=modifier,
                    width=width,
                    height=height,
                )
                taw.weight = 1  # lora_cfg["weight"] if "weight" in lora_cfg else 1

                hnt = imageService_pb2.HintProto()
                hnt.hintType = (
                    modifier
                    if modifier in ["custom", "depth", "scribble", "pose", "color"]
                    else "custom"
                )
                hnt.tensors.append(taw)
                hints.append(hnt)

    async with get_aio_channel(server, port, use_tls) as channel:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        generate_stream = stub.GenerateImage(
            imageService_pb2.ImageGenerationRequest(
                image=img2img,
                scaleFactor=1,
                mask=maskimg,
                hints=hints,
                prompt=positive,
                negativePrompt=negative,
                configuration=config_fbs,
                # override = override,
                user="ComfyUI",
                device="LAPTOP",
                contents=contents,
            )
        )

        cancel_request.reset()
        response_images = []

        while True:
            response = await generate_stream.read()
            if response == grpc.aio.EOF:
                break

            if cancel_request.should_cancel:
                await channel.close()
                raise Exception("canceled")

            current_step = response.currentSignpost.sampling.step
            preview_image = response.previewImage
            generated_images = response.generatedImages

            if current_step:
                try:
                    x0 = None
                    if preview_image and version and settings.show_preview:
                        x0 = decode_preview(preview_image, version)
                    prepare_callback(current_step, config.steps, x0)
                except Exception as e:
                    print("DrawThings-gRPC had an error decoding the preview image:", e)

            if generated_images:
                response_images.extend(generated_images)

        images = []
        for img_data in response_images:
            result = convert_response_image(img_data)
            if result is not None:
                data = result["data"]
                width = result["width"]
                height = result["height"]
                channels = result["channels"]
                mode = "RGB"
                if channels >= 4:
                    mode = "RGBA"
                img = Image.frombytes(mode, (width, height), data)
                image_np = np.array(img)
                tensor_image = torch.from_numpy(image_np.astype(np.float32) / 255.0)
                images.append(tensor_image)

        return (torch.stack(images),)
