"""
PEBBLE Real PyTorch Inference Server.

Loads checkpoint_step_1500.pt directly and serves a live HTTP API
for real model generation using actual trained PyTorch weights.

Usage:
    python ml/server.py --checkpoint ml/checkpoint_step_1500.pt --port 8000
"""

import argparse
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import torch

# Add ml directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel
from pebble.tokenizer import PebbleTokenizer
from generate import generate

# Global model state
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cpu")


def load_model(checkpoint_path: str, tokenizer_path: str):
    global MODEL, TOKENIZER
    print(f"[Pebble Server] Loading PyTorch model weights from {checkpoint_path}...")
    
    config = PebbleConfig()
    MODEL = PebbleLMHeadModel(config).to(DEVICE)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            MODEL.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            MODEL.load_state_dict(checkpoint["model"])
        else:
            MODEL.load_state_dict(checkpoint)
        print(f"[Pebble Server] PyTorch checkpoint loaded successfully ({MODEL.count_parameters()/1e6:.1f}M params)!")
    else:
        print(f"[Pebble Server] Warning: Checkpoint file {checkpoint_path} not found. Using initialized weights.")
        
    MODEL.eval()

    if Path(tokenizer_path).exists():
        TOKENIZER = PebbleTokenizer(tokenizer_path)
        print(f"[Pebble Server] Tokenizer loaded successfully (vocab_size={TOKENIZER.vocab_size})!")
    else:
        print(f"[Pebble Server] Warning: Tokenizer file {tokenizer_path} not found.")


class PebbleHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "healthy",
                "model": "Pebble 120M Mamba-2",
                "checkpoint": "checkpoint_step_1500.pt",
                "device": str(DEVICE)
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode("utf-8"))
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 40)
                temperature = data.get("temperature", 0.8)

                if not prompt:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Prompt required"}')
                    return

                print(f"[Pebble Server] Generating PyTorch output for prompt: '{prompt}'...")
                
                # Execute real PyTorch model.generate()
                output_text = generate(
                    model=MODEL,
                    tokenizer=TOKENIZER,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    device=DEVICE
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                response = {
                    "prompt": prompt,
                    "output": output_text,
                    "model": "Pebble 120M (checkpoint_step_1500.pt)"
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                print(f"[Pebble Server] Error: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, PebbleHandler)
    print(f"[Pebble Server] Server listening on http://localhost:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Pebble Server] Stopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pebble PyTorch Inference Server.")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoint_step_1500.pt")
    parser.add_argument("--tokenizer", type=str, default="ml/tokenizer.json")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.checkpoint, args.tokenizer)
    run_server(args.port)
