from voice_ai.model_loader import load_model
from voice_ai.ai_engine import process_audio_files
import os

# Load model
model = load_model()

# Example input files (replace with actual test audio paths)
song_path = os.path.join("uploads", "test_song.wav")
voice_path = os.path.join("uploads", "test_voice.wav")

# Make sure the files exist
if not os.path.exists(song_path) or not os.path.exists(voice_path):
    print("❌ Test audio files not found in uploads folder.")
else:
    print(f"Processing:\nSong: {song_path}\nVoice: {voice_path}")
    
    # Process files through your AI
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = process_audio_files(model, song_path, voice_path, output_folder)

    print(f"✅ Model processing complete! Output file saved at: {os.path.join(output_folder, output_file)}")
