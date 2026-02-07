from flask import Flask, request, jsonify, send_from_directory, render_template
import os

# Audio processing
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf
import scipy.signal   # ✅ added for smoothing

app = Flask(__name__)

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# ---------------- HELPERS ----------------
def convert_to_wav(file_path):
    """Convert MP3 to WAV if needed"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        return wav_path
    return file_path


def get_average_pitch(audio_path):
    """Extract average pitch using librosa"""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) == 0:
        return 0
    return np.mean(pitch_values)


def pitch_shift_audio(input_path, output_path, n_steps):
    """Shift pitch of audio with smoothing"""
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Pitch shift
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    # Smooth metallic artifacts
    y_shifted = scipy.signal.medfilt(y_shifted, kernel_size=5)

    sf.write(output_path, y_shifted, sr)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "song" not in request.files or "target_voice" not in request.files:
        return jsonify({"error": "Song and target voice are required"}), 400

    song = request.files["song"]
    target_voice = request.files["target_voice"]

    if song.filename == "" or target_voice.filename == "":
        return jsonify({"error": "Empty file selected"}), 400

    song_path = os.path.join(UPLOAD_FOLDER, song.filename)
    voice_path = os.path.join(UPLOAD_FOLDER, target_voice.filename)

    song.save(song_path)
    target_voice.save(voice_path)

    song_wav = convert_to_wav(song_path)
    voice_wav = convert_to_wav(voice_path)

    return jsonify({
        "song_path": song_wav,
        "voice_path": voice_wav
    })


@app.route("/transform", methods=["POST"])
def transform():
    try:
        data = request.get_json()
        song_path = data.get("song_path")
        voice_path = data.get("voice_path")

        if not song_path or not voice_path:
            return jsonify({"error": "song_path and voice_path required"}), 400

        output_filename = os.path.basename(song_path).replace(".wav", "_converted.wav")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # 🎯 Pitch analysis
        song_pitch = get_average_pitch(song_path)
        target_pitch = get_average_pitch(voice_path)

        if song_pitch == 0 or target_pitch == 0:
            return jsonify({"error": "Unable to detect pitch"}), 400

        # 🎚 Gentler pitch difference + limit
        pitch_shift_steps = (target_pitch - song_pitch) / 40
        pitch_shift_steps = max(min(pitch_shift_steps, 3), -3)

        # 🔁 Apply pitch modulation
        pitch_shift_audio(song_path, output_path, pitch_shift_steps)

        return jsonify({
            "output_path": output_filename
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
