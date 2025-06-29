# ComfyUI-DrawThings-gRPC

**ComfyUI-DrawThings-gRPC** is a bridge between [ComfyUI](https://comfyui.org/) and [Draw Things](https://github.com/drawthingsai/draw-things-community) via gRPC. It allows ComfyUI to build and send image generation requests to Draw Things giving you more control over inputs amd settings than Draw Things alone offers, amd bringing the Draw Things sampler into your ComfyUI workflows.

---

## Requirements

- [ComfyUI](https://docs.comfy.org/get_started)
- Draw Things app (with gRPC server enabled) **or** [gRPCServerCLI](https://github.com/drawthingsai/draw-things-community/tree/main?tab=readme-ov-file#self-host-grpcservercli-from-packaged-binaries)
- Python 3.10+ (for ComfyUI and this extension)

---

## Setup

### 1. Install ComfyUI

Follow the [official instructions](https://docs.comfy.org/get_started) to install ComfyUI.

### 2. Install This Extension

**Via ComfyUI-Manager (Recommended):**
- Search for `ComfyUI-DrawThings-gRPC` in the ComfyUI-Manager and install.

**Manual Installation:**
- Clone this repository into your `ComfyUI/custom_nodes` directory:
  ```sh
  git clone https://github.com/yourusername/ComfyUI-DrawThings-gRPC.git ComfyUI/custom_nodes/ComfyUI-DrawThings-gRPC
  ```

### 3. Restart ComfyUI

---

## Configuring Draw Things gRPC Server

### If using the Draw Things app:

Ensure the following settings are enabled:
- **API Server:** enabled
- **Protocol:** gRPC
- **Transport Layer Security:** Enabled
- **Enable Model Browser:** Enabled
- **Response Compression:** Disabled

### If using gRPCServerCLI:

Start the server with:
```sh
gRPCServerCLI-macOS [path to models] --no-response-compression --model-browser
```

---

## Usage

1. Launch ComfyUI.
2. Ensure your Draw Things gRPC server is running and accessible.
3. Use the provided nodes in ComfyUI to send images to Draw Things for ControlNet processing.
4. Images generated in ComfyUI will be automatically transferred to Draw Things.

---

## Troubleshooting

- **Connection Issues:** Ensure both ComfyUI and Draw Things gRPC server are running on the same network and the correct ports are open.
- **TLS Errors:** Make sure Transport Layer Security settings match between client and server.
- **Model Not Found:** Enable Model Browser in Draw Things settings or use `--model-browser` with gRPCServerCLI.

---

## Discussion

Join the conversation and get support on [Discord](https://discord.com/channels/1038516303666876436/1357377020299837464).

---

## Acknowledgements

- [draw-things-community](https://github.com/drawthingsai/draw-things-community)
- [ComfyUI-DrawThingsWrapper](https://github.com/JosephThomasParker/ComfyUI-DrawThingsWrapper)
- [dt-grpc-ts](https://github.com/kcjerrell/dt-grpc-ts)
- [ComfyUI_tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes)

> **Note:**
> This project was created with the [cookiecutter-comfy-extension](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template to simplify custom node development for ComfyUI.
