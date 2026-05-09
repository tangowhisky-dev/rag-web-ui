"""
Minimal entrypoint for the ReLiK FastAPI service.
Bypasses the relik CLI (broken: typer/click incompatibility causes
'Secondary flag is not valid for non-boolean flag' on startup).
Calls the library's main() directly instead.
"""
import os
from relik.inference.serve.backend.fastapi_be import main

model_name = os.getenv("RELIK_MODEL", "relik-ie/relik-cie-small")
device = os.getenv("RELIK_DEVICE", "cpu")

print(f"Loading ReLiK model: {model_name} on {device}")

main(
    relik_pretrained=model_name,
    device=device,
    host="0.0.0.0",
    port=8000,
    workers=1,
)
