import torch
import torchaudio
import torch.nn.functional as F
import json
import gradio as gr
from typing import Dict

# --- Configuration ---
# Set the device to CPU for the free tier
DEVICE = "cpu"
MODEL_PATH = "mini_wav2vec2_full_6000.pth"
IDX2CHAR_PATH = "idx2char.json"
TARGET_SR = 16000

# ==============================================================================
# === 1. MiniWav2Vec2 Model Class (Keep exactly as before) ===
# ==============================================================================
class MiniWav2Vec2(torch.nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=30):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        return self.fc(x)

# ==============================================================================
# === 2. Token mapping helpers (Keep exactly as before) ===
# ==============================================================================
def greedy_decode(log_probs, idx2char: Dict[int, str]):
    max_probs = torch.argmax(log_probs, dim=-1)
    decoded_texts = []
    for seq in max_probs:
        prev_id = None
        decoded_seq = []
        for idx in seq.cpu().numpy():
            if idx != 0 and idx != prev_id:
                decoded_seq.append(idx2char.get(idx, ""))
            prev_id = idx
        decoded_texts.append("".join(decoded_seq))
    return decoded_texts

# ==============================================================================
# === 3. ASR Inference Function (Adapted from the Flask handler) ===
# ==============================================================================

# Global model and artifacts, loaded once at startup
MODEL = None
IDX2CHAR = None
MFCC_TRANSFORM = None

def load_model_and_artifacts():
    """Loads all heavy assets once at application startup."""
    global MODEL, IDX2CHAR, MFCC_TRANSFORM
    
    if MODEL is not None:
        return # Already loaded

    print("Loading model and artifacts...")
    try:
        # Load char mapping
        with open(IDX2CHAR_PATH, "r", encoding="utf-8") as f:
            IDX2CHAR = {int(k): v for k, v in json.load(f).items()}

        # Load model (ensure safe globals for custom class)
        torch.serialization.add_safe_globals([MiniWav2Vec2])
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        MODEL.eval()
        MODEL.to(DEVICE)

        # Initialize MFCC transform
        MFCC_TRANSFORM = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SR, 
            n_mfcc=40
        ).to(DEVICE)
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        raise RuntimeError("Failed to load model dependencies.")

# The core transcription logic
def transcribe_audio_file(audio_file_path: str) -> str:
    """
    Transcribes an audio file path provided by Gradio.
    """
    load_model_and_artifacts() # Ensure model is loaded
    
    try:
        # 1) Load waveform
        waveform, sr = torchaudio.load(audio_file_path) # [channels, time]

        # 2) Resample if needed
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR).to(DEVICE)
            waveform = resampler(waveform.to(DEVICE))
        else:
            waveform = waveform.to(DEVICE)

        # 3) Convert to mono
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 4) Compute MFCCs
        mfcc_feats = MFCC_TRANSFORM(waveform) # [1, n_mfcc, frames]

        # 5) Prepare for model (batch, T, features)
        feats = mfcc_feats.permute(0, 2, 1) # [1, frames, n_mfcc]

        if feats.shape[1] == 0:
            return "Error: Audio file too short to generate MFCC features."
        
        lengths = torch.tensor([feats.shape[1]])

        # 6) Forward pass
        with torch.no_grad():
            logits = MODEL(feats, lengths) 
            log_probs = F.log_softmax(logits, dim=-1)
            preds = greedy_decode(log_probs, IDX2CHAR)
            
            return preds[0] if preds else "Transcription failed."

    except Exception as e:
        import traceback
        return f"Transcription Error: {str(e)} \nTraceback: {traceback.format_exc()}"


# ==============================================================================
# === 4. Gradio Interface ===
# ==============================================================================
gr.Interface(
    fn=transcribe_audio_file,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio (.wav, .mp3) or Record")
    ],
    outputs=[
        gr.Textbox(label="Transcription Result")
    ],
    title="Mini Wav2Vec2 ASR Demo",
    description="Upload an audio file (or record audio) to get a real-time speech transcription using a custom PyTorch model."
).launch(server_name="0.0.0.0", server_port=7860) # Gradio default port