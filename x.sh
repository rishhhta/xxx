#!/bin/bash
set -e

source /venv/main/bin/activate

WORKSPACE=${WORKSPACE:-/workspace}
COMFYUI_DIR=${WORKSPACE}/ComfyUI

echo "=== Vast.ai ComfyUI provisioning ==="

# ---------------------------------------------
# CONFIG — NODES
# ---------------------------------------------
NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/chflame163/ComfyUI_LayerStyle"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/ClownsharkBatwing/RES4LYF"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
    "https://github.com/Smirnov75/ComfyUI-mxToolkit"
    "https://github.com/TheLustriVA/ComfyUI-Image-Size-Tools"
    "https://github.com/ZhiHui6/zhihui_nodes_comfyui"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/crystian/ComfyUI-Crystools"
    "https://github.com/jnxmx/ComfyUI_HuggingFace_Downloader"
    "https://github.com/plugcrypt/CRT-Nodes"
    "https://github.com/EllangoK/ComfyUI-post-processing-nodes"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
)

# ---------------------------------------------
# CONFIG — MODELS
# ---------------------------------------------
CLIP_MODELS=(
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
)

CKPT_MODELS=(
    "https://huggingface.co/cyberdelia/CyberRealisticPony/resolve/main/CyberRealisticPony_V15.0_FP32.safetensors"
)

FUN_MODELS=(
    "https://huggingface.co/arhiteector/zimage/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
)

TEXT_ENCODERS=(
    "https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/refs%2Fpr%2F5/models/clip/umt5-xxl-encoder-fp8-e4m3fn-scaled.safetensors"
)

UNET_MODELS=(
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
)

VAE_MODELS=(
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
)

DIFFUSION_MODELS=(
    "https://huggingface.co/T5B/Z-Image-Turbo-FP8/resolve/main/z-image-turbo-fp8-e4m3fn.safetensors"
)

BBOX_MODELS=(
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/face_yolov8s.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/femaleBodyDetection_yolo26.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/female_breast-v4.2.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/nipples_yolov8s.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/vagina-v4.2.pt"
    "https://huggingface.co/gazsuv/xmode/resolve/main/assdetailer.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/Eyeful_v2-Paired.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/Eyes.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/FacesV1.pt"
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/hand_yolov8s.pt"
    "https://huggingface.co/AunyMoons/loras-pack/blob/main/foot-yolov8l.pt"
)

SAM_PTH=(
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth"
)

QWEN3VL_1=(
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/added_tokens.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/chat_template.jinja"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/config.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/generation_config.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/merges.txt"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/model.safetensors.index.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/preprocessor_config.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/special_tokens_map.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/tokenizer.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/tokenizer_config.json"
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/vocab.json"
)

QWEN3VL_2=(
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/model-00001-of-00002.safetensors"
)

QWEN3VL_3=(
    "https://huggingface.co/svjack/Qwen3-VL-4B-Instruct-heretic-7refusal/resolve/main/model-00002-of-00002.safetensors"
)

UPSCALER_MODELS=(
    "https://huggingface.co/gazsuv/pussydetectorv4/resolve/main/4xUltrasharp_4xUltrasharpV10.pt"
)

# ---------------------------------------------
# FUNCTIONS
# ---------------------------------------------
download_files() {
    local dir="$1"
    shift
    mkdir -p "$dir"

    for url in "$@"; do
        echo "> $url"
        local auth_header=""
        if [[ -n "$HF_TOKEN" && "$url" =~ huggingface\.co ]]; then
            auth_header="--header=Authorization: Bearer $HF_TOKEN"
        fi
        wget $auth_header -nc --content-disposition --show-progress -e dotbytes=4M -P "$dir" "$url" \
            || echo " [!] Download failed: $url"
    done
}

# ---------------------------------------------
# 1. Clone ComfyUI
# ---------------------------------------------
if [[ ! -d "${COMFYUI_DIR}" ]]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
fi

cd "${COMFYUI_DIR}"

# ---------------------------------------------
# 2. Install base requirements
# ---------------------------------------------
if [[ -f requirements.txt ]]; then
    pip install --no-cache-dir -r requirements.txt
fi

# ---------------------------------------------
# 3. Custom nodes
# ---------------------------------------------
mkdir -p custom_nodes

for repo in "${NODES[@]}"; do
    dir="${repo##*/}"
    path="custom_nodes/${dir}"

    if [[ -d "$path" ]]; then
        echo "Updating node: $dir"
        (cd "$path" && git pull --ff-only 2>/dev/null || { git fetch && git reset --hard origin/main; })
    else
        echo "Cloning node: $dir"
        git clone "$repo" "$path" --recursive || echo " [!] Clone failed: $repo"
    fi

    [[ -f "${path}/requirements.txt" ]] && pip install --no-cache-dir -r "${path}/requirements.txt" \
        || echo " [!] pip requirements failed for $dir"
done

# ---------------------------------------------
# 4. Download models
# ---------------------------------------------
download_files "models/clip"                                                        "${CLIP_MODELS[@]}"
download_files "models/text_encoders"                                               "${TEXT_ENCODERS[@]}"
download_files "models/unet"                                                        "${UNET_MODELS[@]}"
download_files "models/vae"                                                         "${VAE_MODELS[@]}"
download_files "models/checkpoints"                                                 "${CKPT_MODELS[@]}"
download_files "models/model_patches"                                               "${FUN_MODELS[@]}"
download_files "models/diffusion_models"                                            "${DIFFUSION_MODELS[@]}"
download_files "models/ultralytics/bbox"                                            "${BBOX_MODELS[@]}"
download_files "models/sams"                                                        "${SAM_PTH[@]}"
download_files "models/prompt_generator/Qwen3-VL-4B-Instruct-heretic-7refusal"     "${QWEN3VL_1[@]}"
download_files "models/prompt_generator/Qwen3-VL-4B-Instruct-heretic-7refusal"     "${QWEN3VL_2[@]}"
download_files "models/prompt_generator/Qwen3-VL-4B-Instruct-heretic-7refusal"     "${QWEN3VL_3[@]}"
download_files "models/upscale_models"                                              "${UPSCALER_MODELS[@]}"

# ---------------------------------------------
# 5. Launch
# ---------------------------------------------
echo "=== Starting ComfyUI ==="
python main.py --listen 0.0.0.0 --port 8188
