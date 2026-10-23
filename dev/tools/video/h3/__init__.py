"""MiniMax-H3 on-device frontend: text encoder -> prep -> W8A8 VDN-H3 DiT (engine) -> video VAE (engine) + audio VAE
(MLX) -> mp4, with an optional TAEH3 preview. Run `python -m h3 --help`; request options and paths: h3/config.py;
engine binding: h3/engine.py over include/h3/h3.h."""
