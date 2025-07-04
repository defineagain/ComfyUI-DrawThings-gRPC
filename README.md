# ComfyUI-DrawThings-gRPC

**ComfyUI-DrawThings-gRPC** is a bridge between [ComfyUI](https://comfy.org/) and [Draw Things](https://drawthings.ai/) via gRPC. It allows ComfyUI to send image generation requests to Draw Things, letting you bring the Draw Things sampler into your ComfyUI workflows.

---

## Requirements

- [ComfyUI](https://comfy.org)
- [Draw Things](https://drawthings.ai/) app (with gRPC server enabled) **or** [gRPCServerCLI](https://github.com/drawthingsai/draw-things-community/tree/main?tab=readme-ov-file#self-host-grpcservercli-from-packaged-binaries)

---

## Setup

### 1. Install ComfyUI

### 2. Install Draw Things

### 2. Install This Extension

**Via ComfyUI-Manager (Recommended):**
- Search for `ComfyUI-DrawThings-gRPC` in the ComfyUI-Manager and install.

**Manual Installation:**
- Clone this repository into your `ComfyUI/custom_nodes` directory:
  ```sh
  git clone https://github.com/yourusername/ComfyUI-DrawThings-gRPC.git
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
