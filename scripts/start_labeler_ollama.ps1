# Labeler backend: second Ollama instance with the iGPU (Vulkan), port 11435.
# Run this after a reboot (or whenever the :11435 instance is down).
# Requires (once, see README): OLLAMA_IGPU_ENABLE=1, OLLAMA_MAX_LOADED_MODELS=2
# and the medgemma-labeler model (thinking OFF) created on that instance.
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_VULKAN = "true"
$env:OLLAMA_IGPU_ENABLE = "1"
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
