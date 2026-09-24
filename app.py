from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import base64

from text_model import TextGenerator
from image_generator import ImageGenerator
from audio_generator import AudioGenerator

ROOT = Path(__file__).parent
app = Flask(__name__, static_folder="web", static_url_path="")

text_generator = TextGenerator()
image_generator = ImageGenerator()
audio_generator = AudioGenerator()

@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")

@app.post("/api/generate/text")
def generate_text():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "hello")).strip()
    return jsonify(text_generator.generate(prompt, max_words=200))

@app.post("/api/generate/image")
def generate_image():
    payload = request.get_json(silent=True) or {}
    seed = int(payload.get("seed", 0))
    result = image_generator.generate(seed=seed)
    result["png"] = "data:image/png;base64," + base64.b64encode(result["png"]).decode("ascii")
    return jsonify(result)

@app.post("/api/generate/audio")
def generate_audio():
    payload = request.get_json(silent=True) or {}
    seed = int(payload.get("seed", 0))
    result = audio_generator.generate(seed=seed)
    result["wav"] = "data:audio/wav;base64," + base64.b64encode(result["wav"]).decode("ascii")
    return jsonify(result)

@app.post("/api/generate/all")
def generate_all():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "hello")).strip()
    seed = int(payload.get("seed", 0))

    text = text_generator.generate(prompt, max_words=200)
    image = image_generator.generate(seed=seed)
    audio = audio_generator.generate(seed=seed)

    image_data = "data:image/png;base64," + base64.b64encode(image.pop("png")).decode("ascii")
    audio_data = "data:audio/wav;base64," + base64.b64encode(audio.pop("wav")).decode("ascii")

    return jsonify({
        "text": text,
        "image": {**image, "png": image_data},
        "audio": {**audio, "wav": audio_data},
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
