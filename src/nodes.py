#!../../.venv python3

import os
import sys
import base64
import numpy as np
from PIL import Image
import torch
import asyncio
import grpc
import flatbuffers
import json
from google.protobuf.json_format import MessageToJson
from .generated import imageService_pb2, imageService_pb2_grpc
from .generated import Control
from .generated import LoRA
from .generated import GenerationConfiguration
from .data_types import *
from .image_handlers import prepare_callback, convert_response_image, decode_preview, convert_image_for_request, convert_mask_for_request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from server import PromptServer
from aiohttp import web

try:
    with open('./custom_nodes/ComfyUI-DrawThings-gRPC/resources/root_ca.crt', 'rb') as cert:
        credentials = grpc.ssl_channel_credentials(cert.read())
except:
    credentials = None

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
    # if 'server' not in request.args or 'port' not in request.args:
    #     return web.json_response({"error": "Missing server or port parameter"}, status=400)
    # server = request.args['server']
    # port = request.args['port']
    try:
        post = await request.post()
        server = post.get('server')
        port = post.get('port')
        use_tls = post.get('use_tls')

        print(server, port, use_tls)

        if server is None or port is None:
            return web.json_response({"error": "Missing server or port parameter"}, status=400)
        all_files = get_files(server, port, use_tls)
        return web.json_response(all_files)
    except Exception as e:
        print(e)
        return web.json_response({"error": "Could not connect to Draw Things gRPC server. Please check the server address and port."}, status=500)

async def dt_sampler(
                server,
                port,
                use_tls,
                model: ModelInfo,
                seed,
                seed_mode,
                steps,
                num_frames,
                cfg,
                strength,
                speed_up,
                guidance_embed,
                sampler_name,
                res_dpt_shift,
                shift,
                batch_size,
                fps,
                motion_scale,
                guiding_frame_noise,
                start_frame_guidance,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                positive,
                negative,
                width,
                height,
                high_res_fix,
                high_res_fix_start_width,
                high_res_fix_start_height,
                high_res_fix_strength,
                tiled_decoding,
                decoding_tile_width,
                decoding_tile_height,
                decoding_tile_overlap,
                tiled_diffusion,
                diffusion_tile_width,
                diffusion_tile_height,
                diffusion_tile_overlap,
                tea_cache,
                tea_cache_start,
                tea_cache_end,
                tea_cache_threshold,
                tea_cache_max_skip_steps,

                separate_clip_l: bool,
                clip_l_text: str,
                separate_open_clip_g : bool,
                open_clip_g_text: str,

                batch_count=1,
                scale_factor=1,
                image=None,
                mask=None,
                control_net: ControlStack=None,
                lora: LoraStack=None,
                refiner=None,
                upscaler=None,
            ) -> None:
    print(positive, negative)
    builder = flatbuffers.Builder(0)

    loras_out = None
    if lora is not None and len(lora):
        fin_loras = []
        for l in lora:
            print(l)
            lora_model = l['model']
            lora_file = builder.CreateString(lora_model['file'])
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
            print(f'{c["input_type"]}')
            cnet_model = c['model']
            control_name = builder.CreateString(cnet_model["file"])
            Control.Start(builder)
            Control.AddFile(builder, control_name)
            # NOTE: So apparantly THIS is where you set all the types, NOT via Hints as that's for which slot to use
            Control.AddInputOverride(builder, DrawThingsLists.control_input_type.index(c["input_type"]))
            Control.AddControlMode(builder, DrawThingsLists.control_mode.index(c["mode"]))
            Control.AddWeight(builder, c["weight"])
            Control.AddGuidanceStart(builder, c["start"])
            Control.AddGuidanceEnd(builder, c["end"])
            # Control.AddNoPrompt(builder, False)
            # Control.AddGlobalAveragePooling(builder, False)
            # Control.AddDownSamplingRate(builder, 0)
            # Control.AddTargetBlocks(builder, 0)
            # Control.StartTargetBlocksVector(builder, )
            fin_control = Control.End(builder)
            fin_controls.append(fin_control)

        GenerationConfiguration.StartControlsVector(builder, len(fin_controls))
        for fc in fin_controls:
            builder.PrependUOffsetTRelative(fc)
        controls_out = builder.EndVector()

    start_width = width // 64 // scale_factor
    start_height = height // 64 // scale_factor
    model_name = builder.CreateString(model['file'])
    if upscaler is not None:
        upscaler_model = builder.CreateString(upscaler["upscaler_model"])
    if refiner is not None:
        refiner_model = builder.CreateString(refiner["refiner_model"])

    clip_l_text_buf = builder.CreateString(clip_l_text or "") if separate_clip_l else None
    open_clip_g_text_buf = builder.CreateString(open_clip_g_text or "") if separate_open_clip_g else None

    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddModel(builder, model_name)
    GenerationConfiguration.AddStrength(builder, strength)
    GenerationConfiguration.AddSeed(builder, seed % 4294967295)
    GenerationConfiguration.AddSeedMode(builder, DrawThingsLists.seed_mode.index(seed_mode))
    GenerationConfiguration.AddStartWidth(builder, start_width)
    GenerationConfiguration.AddStartHeight(builder, start_height)
    GenerationConfiguration.AddTargetImageWidth(builder, width)
    GenerationConfiguration.AddTargetImageHeight(builder, height)
    if upscaler is not None:
        GenerationConfiguration.AddUpscaler(builder, upscaler_model)
        GenerationConfiguration.AddUpscalerScaleFactor(builder, upscaler["upscaler_scale_factor"])
    GenerationConfiguration.AddSteps(builder, steps)
    GenerationConfiguration.AddNumFrames(builder, num_frames)
    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    GenerationConfiguration.AddImageGuidanceScale(builder, cfg) # Don't know why again, but adding to be sure
    GenerationConfiguration.AddSpeedUpWithGuidanceEmbed(builder, speed_up)
    GenerationConfiguration.AddGuidanceEmbed(builder, guidance_embed)
    GenerationConfiguration.AddSampler(builder, DrawThingsLists.sampler_list.index(sampler_name))
    GenerationConfiguration.AddResolutionDependentShift(builder, res_dpt_shift)
    GenerationConfiguration.AddShift(builder, shift)
    GenerationConfiguration.AddFpsId(builder, fps)
    GenerationConfiguration.AddMotionBucketId(builder, motion_scale)
    GenerationConfiguration.AddCondAug(builder, guiding_frame_noise)
    GenerationConfiguration.AddStartFrameCfg(builder, start_frame_guidance)
    GenerationConfiguration.AddBatchSize(builder, batch_size)
    if refiner is not None:
        GenerationConfiguration.AddRefinerModel(builder, refiner_model)
        GenerationConfiguration.AddRefinerStart(builder, refiner["refiner_start"])
    # GenerationConfiguration.AddStage2Steps(builder, ) # wurst
    # GenerationConfiguration.AddStage2Cfg(builder, ) # wurst
    # GenerationConfiguration.AddStage2Shift(builder, ) # wurst
    # GenerationConfiguration.AddZeroNegativePrompt(builder, )
    # GenerationConfiguration.AddSeparateClipL(builder, )
    # GenerationConfiguration.AddClipLText(builder, )
    # GenerationConfiguration.AddSeparateOpenClipG(builder, )
    # GenerationConfiguration.AddOpenClipGText(builder, )
    GenerationConfiguration.AddClipSkip(builder, clip_skip)
    GenerationConfiguration.AddSharpness(builder, sharpness)
    GenerationConfiguration.AddMaskBlur(builder, mask_blur)
    GenerationConfiguration.AddMaskBlurOutset(builder, mask_blur_outset)
    GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, preserve_original)
    # GenerationConfiguration.AddFaceRestoration(builder, )
    GenerationConfiguration.AddHiresFix(builder, high_res_fix)
    if high_res_fix is True:
        GenerationConfiguration.AddHiresFixStartWidth(builder, high_res_fix_start_width)
        GenerationConfiguration.AddHiresFixStartHeight(builder, high_res_fix_start_height)
        GenerationConfiguration.AddHiresFixStrength(builder, high_res_fix_strength)

    GenerationConfiguration.AddTiledDecoding(builder, tiled_decoding)
    if tiled_decoding is True:
        GenerationConfiguration.AddDecodingTileWidth(builder, decoding_tile_width)
        GenerationConfiguration.AddDecodingTileHeight(builder, decoding_tile_height)
        GenerationConfiguration.AddDecodingTileOverlap(builder, decoding_tile_overlap)

    GenerationConfiguration.AddTiledDiffusion(builder, tiled_diffusion)
    if tiled_diffusion is True:
        GenerationConfiguration.AddDiffusionTileWidth(builder, diffusion_tile_width)
        GenerationConfiguration.AddDiffusionTileHeight(builder, diffusion_tile_height)
        GenerationConfiguration.AddDiffusionTileOverlap(builder, diffusion_tile_overlap)

    GenerationConfiguration.AddTeaCache(builder, tea_cache)
    if tea_cache is True: # flux or video option
        GenerationConfiguration.AddTeaCacheStart(builder, tea_cache_start)
        GenerationConfiguration.AddTeaCacheEnd(builder, tea_cache_end)
        GenerationConfiguration.AddTeaCacheThreshold(builder, tea_cache_threshold)
        GenerationConfiguration.AddTeaCacheMaxSkipSteps(builder, tea_cache_max_skip_steps)

    if separate_clip_l:
        GenerationConfiguration.GenerationConfigurationAddSeparateClipL(builder, True)
        GenerationConfiguration.GenerationConfigurationAddClipLText(builder, clip_l_text_buf)

    if separate_open_clip_g:
        GenerationConfiguration.GenerationConfigurationAddSeparateOpenClipG(builder, True)
        GenerationConfiguration.GenerationConfigurationAddOpenClipGText(builder, open_clip_g_text_buf)

    # ti embed

    # The rest, don't know where they go or which models use them
    # GenerationConfiguration.AddClipWeight(builder, )
    # GenerationConfiguration.AddNegativePromptForImagePrior(builder, )
    # GenerationConfiguration.AddImagePriorSteps(builder, )
    # GenerationConfiguration.AddCropTop(builder, )
    # GenerationConfiguration.AddCropLeft(builder, )
    # GenerationConfiguration.AddAestheticScore(builder, )
    # GenerationConfiguration.AddNegativeAestheticScore(builder, )
    # GenerationConfiguration.AddNegativeOriginalImageHeight(builder, )
    # GenerationConfiguration.AddNegativeOriginalImageWidth(builder, )
    # GenerationConfiguration.AddName(builder, )
    # GenerationConfiguration.AddStochasticSamplingGamma(builder, )
    # GenerationConfiguration.AddT5TextEncoder(builder, )

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
            maskimg = convert_mask_for_request(mask, width, height)

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
                # NOTE: So apparantly THIS is where you set which slot to use, NOT via InputOverride as that's for all the types
                # Invalid ControlHintType 'unspecified'
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
        print(lora)
        # Needed for loras like FLUX.1-Depth-dev-lora
        for lora_cfg in lora:
            if 'control_image' in lora_cfg:
                modifier = lora_cfg["model"]["modifier"]

                taw = imageService_pb2.TensorAndWeight()
                taw.tensor = convert_image_for_request(lora_cfg["control_image"], modifier)
                taw.weight = lora_cfg["weight"] if "weight" in lora_cfg else 1

                hnt = imageService_pb2.HintProto()
                hnt.hintType = modifier if modifier in ["custom", "depth", "scribble", "pose", "color"] else "custom"
                hnt.tensors.append(taw)
                hints.append(hnt)

    async with get_aio_channel(server, port, use_tls) as channel:
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

            if current_step:
                x0 = None
                if preview_image:
                    model_version = model["version"]
                    if model_version:
                        x0 = decode_preview(preview_image, model_version)
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

    sampler_list = [
                "DPM++ 2M Karras",
                "Euler A",
                "DDIM",
                "PLMS",
                "DPM++ SDE Karras",
                "UniPC",
                "LCM",
                "Euler A Substep",
                "DPM++ SDE Substep",
                "TCD",
                "Euler A Trailing",
                "DPM++ SDE Trailing",
                "DPM++ 2M AYS",
                "Euler A AYS",
                "DPM++ SDE AYS",
                "DPM++ 2M Trailing",
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
                "Custom", # -> Slot
                "Depth", # -> Slot
                "Canny",
                "Scribble", # -> Slot
                "Pose", # -> Slot
                "Normalbae",
                "Color", # -> Slot
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

                "res_dpt_shift": ("BOOLEAN", {"default": True}),

                "shift": ("FLOAT", {"default": 1.00, "min": 0.10, "max": 8.00, "step": 0.01, "round": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                # refiner
                "fps": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
                "motion_scale": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "guiding_frame_noise": ("FLOAT", {"default": 0.02, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
                "start_frame_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 25.0, "step": 0.1, "round": 0.1}),
                # zero neg
                # sep clip
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 23, "step": 1}),
                "sharpness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.1}),
                "mask_blur": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 50.0, "step": 0.1, "round": 0.1}),
                "mask_blur_outset": ("INT", {"default": 0, "min": -100, "max": 1000, "step": 1}),
                "preserve_original": ("BOOLEAN", {"default": True}),
                # face restore

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

    def sample(self,
                settings,
                server,
                port,
                use_tls,
                model,
                seed,
                seed_mode,
                steps,
                num_frames,
                cfg,
                strength,
                speed_up,
                guidance_embed,
                sampler_name,
                res_dpt_shift,
                shift,
                batch_size,
                fps,
                motion_scale,
                guiding_frame_noise,
                start_frame_guidance,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                width,
                height,
                high_res_fix,
                high_res_fix_start_width,
                high_res_fix_start_height,
                high_res_fix_strength,
                tiled_decoding,
                decoding_tile_width,
                decoding_tile_height,
                decoding_tile_overlap,
                tiled_diffusion,
                diffusion_tile_width,
                diffusion_tile_height,
                diffusion_tile_overlap,
                tea_cache,
                tea_cache_start,
                tea_cache_end,
                tea_cache_threshold,
                tea_cache_max_skip_steps,
                separate_clip_l,
                clip_l_text,
                separate_open_clip_g,
                open_clip_g_text,
                batch_count=1,
                scale_factor=1,
                image=None,
                mask=None,
                positive="",
                negative="",
                control_net: ControlStack=None,
                lora: LoraStack=None,
                refiner=None,
                upscaler=None,
                ):

        return asyncio.run(dt_sampler(
                server,
                port,
                use_tls,
                model["value"],
                seed,
                seed_mode,
                steps,
                num_frames,
                cfg,
                strength,
                speed_up,
                guidance_embed,
                sampler_name,
                res_dpt_shift,
                shift,
                batch_size,
                fps,
                motion_scale,
                guiding_frame_noise,
                start_frame_guidance,
                clip_skip,
                sharpness,
                mask_blur,
                mask_blur_outset,
                preserve_original,
                positive or "",
                negative or "",
                width,
                height,
                high_res_fix,
                high_res_fix_start_width,
                high_res_fix_start_height,
                high_res_fix_strength,
                tiled_decoding,
                decoding_tile_width,
                decoding_tile_height,
                decoding_tile_overlap,
                tiled_diffusion,
                diffusion_tile_width,
                diffusion_tile_height,
                diffusion_tile_overlap,
                tea_cache,
                tea_cache_start,
                tea_cache_end,
                tea_cache_threshold,
                tea_cache_max_skip_steps,
                separate_clip_l,
                clip_l_text,
                separate_open_clip_g,
                open_clip_g_text,
                batch_count=batch_count,
                scale_factor=scale_factor,
                image=image,
                mask=mask,
                control_net=control_net,
                lora=lora,
                refiner=refiner,
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
    RETURN_NAMES = ("REFINER",)
    FUNCTION = "add_to_pipeline"
    CATEGORY = "DrawThings"

    def add_to_pipeline(self, refiner_model, refiner_start):
        refiner = {"refiner_model": refiner_model, "refiner_start": refiner_start}
        return (refiner,)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

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
    def INPUT_TYPES(s):
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
    def INPUT_TYPES(s):
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
    def INPUT_TYPES(s):
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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DrawThingsSampler": DrawThingsSampler,
    "DrawThingsControlNet": DrawThingsControlNet,
    "DrawThingsLoRA": DrawThingsLoRA,
    "DrawThingsPositive": DrawThingsPositive,
    "DrawThingsNegative": DrawThingsNegative,
    "DrawThingsRefiner": DrawThingsRefiner,
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
    "DrawThingsUpscaler": "Draw Things Upscaler",
}
