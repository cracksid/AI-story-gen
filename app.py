"""A small web server so the model can be driven from a browser.

Not part of the assignment: the six design questions are about the model, and
this answers none of them. It exists so the system can be demonstrated live,
and so the model's uncertainty is visible rather than only summarised as a
perplexity number.

Built on http.server from the standard library rather than FastAPI or Flask,
because the whole job is two routes and adding a web framework to a project
whose dependency list is "PyTorch" would be out of proportion.

Run:  python app.py      then open http://127.0.0.1:8100
"""

import json
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from generate import load
from tokenizer import GENRES, clean_text

WEB_DIR = Path(__file__).parent / "web"
PORT = 8100
TOP_SHOWN = 6      # how many candidate characters the browser draws


class Handler(BaseHTTPRequestHandler):
    """Two routes: the page itself, and the streaming generation endpoint."""

    def __init__(self, model, tokenizer, *args, **kwargs):
        # The model is loaded once at startup and shared by every request.
        # BaseHTTPRequestHandler is constructed per request, so it is passed
        # in via functools.partial rather than reloaded here.
        self.model, self.tokenizer = model, tokenizer
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        pass   # silence the default per-request logging

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/":
            self.send_file("index.html", "text/html")
        elif route.path == "/api/genres":
            self.send_json({"genres": GENRES})
        elif route.path == "/api/generate":
            self.stream_generation(parse_qs(route.query))
        else:
            self.send_error(404)

    def send_file(self, name: str, content_type: str) -> None:
        body = (WEB_DIR / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_generation(self, query: dict[str, list[str]]) -> None:
        """Server-Sent Events: hold the connection open, write as we sample.

        A normal response is request, wait, one complete reply. SSE keeps the
        socket open and writes "data: {...}" blocks as they become ready, and
        the browser's built-in EventSource fires an event per block. One-way
        and about ten lines, where a WebSocket would need a handshake protocol
        for a conversation that only ever goes one direction.
        """
        def arg(name, default, cast=str):
            return cast(query.get(name, [default])[0])

        genre = arg("genre", GENRES[0])
        if genre not in GENRES:
            self.send_error(400, "unknown genre")
            return
        # Clamped, because these arrive from a browser and a request for ten
        # million characters would tie up the server until it finished.
        tokens = max(1, min(2000, arg("tokens", "600", int)))
        temperature = max(0.1, min(2.0, arg("temperature", "0.8", float)))
        top_k = arg("top_k", "0", int) or None
        # Prompts are human-typed, so they go through the same cleaning the
        # corpus went through; otherwise encode raises KeyError on a character
        # the model has never seen.
        prompt = clean_text(arg("prompt", "")) or "\n"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        idx = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        gid = torch.tensor([GENRES.index(genre)], dtype=torch.long)
        try:
            self.send_event({"type": "prompt", "text": prompt})
            for next_id, probs in self.model.stream(idx, gid, tokens,
                                                    temperature, top_k):
                scores, ids = torch.topk(probs, TOP_SHOWN)
                self.send_event({
                    "type": "char",
                    "c": self.tokenizer.itos[next_id],
                    # What the model considered, not only what it picked.
                    "top": [[self.tokenizer.itos[i], round(s, 4)]
                            for i, s in zip(ids.tolist(), scores.tolist())],
                })
            self.send_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            # The user navigated away or pressed stop. Normal, not an error.
            pass

    def send_event(self, payload: dict) -> None:
        """Write one SSE block and flush, so the browser sees it immediately."""
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("loading checkpoint...")
    model, tokenizer = load()
    n = sum(p.numel() for p in model.parameters())
    print(f"{n:,} parameters, vocabulary {tokenizer.vocab_size}")

    # ThreadingHTTPServer, not HTTPServer: generation holds a connection open
    # for several seconds, and a single-threaded server would refuse to serve
    # the page itself while one story was streaming.
    handler = partial(Handler, model, tokenizer)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    print(f"\n  open  http://127.0.0.1:{PORT}\n  stop  Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
