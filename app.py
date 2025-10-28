import torch
import torchaudio
import torch.nn.functional as F
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
# -----------------------------
# 1. MiniWav2Vec2 Model Class
# -----------------------------
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

# -----------------------------
# 2. Token mapping helpers
# -----------------------------
def greedy_decode(log_probs, idx2char):
    max_probs = torch.argmax(log_probs, dim=-1)
    decoded_texts = []
    for seq in max_probs:
        prev_id = None
        decoded_seq = []
        for idx in seq.cpu().numpy():
            if idx != 0 and idx != prev_id:  # skip pad and duplicates
                decoded_seq.append(idx2char.get(idx, ""))
            prev_id = idx
        decoded_texts.append("".join(decoded_seq))
    return decoded_texts

# -----------------------------
# 3. ASR Model Handler
# -----------------------------
class ASRModel:
    def __init__(self, model_path, idx2char_path):
        print("Loading model...")
        # Load char mapping
        with open(idx2char_path, "r", encoding="utf-8") as f:
            self.idx2char = {int(k): v for k, v in json.load(f).items()}


        # Load model
        torch.serialization.add_safe_globals([MiniWav2Vec2])
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.eval()
        self.resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        print("Model loaded successfully!")
        
    def transcribe(self, audio_file_path, debug=True):
        """
        Robust transcription with extensive shape logging and safe resampling.
        Returns transcription string on success, or raises/returns a helpful error.
        """
        try:
            # 1) load
            waveform, sr = torchaudio.load(audio_file_path)  # waveform: [channels, time]
            if debug:
                print(f"[DEBUG] loaded waveform.shape={tuple(waveform.shape)}, sr={sr}")

            # 2) if sample rate differs from target, create a resampler using actual sr
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                if debug:
                    print(f"[DEBUG] resampled -> waveform.shape={tuple(waveform.shape)}, new_sr={target_sr}")
            else:
                if debug:
                    print(f"[DEBUG] sample rate already {target_sr}, skipping resample")

            # 3) convert to mono (average channels) if needed
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # [1, time]
                if debug:
                    print(f"[DEBUG] converted to mono -> waveform.shape={tuple(waveform.shape)}")
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # ensure [1, time]
                if debug:
                    print(f"[DEBUG] added batch dim -> waveform.shape={tuple(waveform.shape)}")

            # 4) optional amplitude normalization (helps MFCC stability)
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / (max_val + 1e-9)
                if debug:
                    print(f"[DEBUG] normalized waveform max value before: {max_val:.6f}")

            # 5) compute MFCCs
            mfcc_feats = self.mfcc(waveform)  # expected [1, n_mfcc, frames]
            if debug:
                print(f"[DEBUG] mfcc_feats.shape={tuple(mfcc_feats.shape)}")

            # Validate MFCC dims
            if mfcc_feats.dim() != 3:
                raise RuntimeError(f"MFCC output expected 3 dims (C, n_mfcc, frames) but got {mfcc_feats.dim()} dims")

            # 6) convert to model's expected shape (batch, T, features)
            # mfcc_feats: [1, n_mfcc, frames] -> feats: [1, frames, n_mfcc]
            feats = mfcc_feats.permute(0, 2, 1)  # safe and explicit
            if debug:
                print(f"[DEBUG] feats.shape (batch, T, features) = {tuple(feats.shape)}")

            # sanity: ensure feats is 3D and features dim equals model input_dim (40 by default)
            if feats.dim() != 3:
                raise RuntimeError(f"feats must be 3D, got shape {tuple(feats.shape)}")
            expected_feature_dim = feats.shape[2]
            if expected_feature_dim != 40:
                # not fatal, but warn if your model expects 40 MFCCs
                print(f"[WARN] feature dim = {expected_feature_dim}; model may expect 40 MFCC features.")

            lengths = torch.tensor([feats.shape[1]])
            if debug:
                print(f"[DEBUG] lengths={lengths.tolist()}")

            # 7) forward pass
            with torch.no_grad():
                logits = self.model(feats, lengths)  # shape -> [batch, T, num_classes]
                if debug:
                    print(f"[DEBUG] logits.shape={tuple(logits.shape)}")

                log_probs = F.log_softmax(logits, dim=-1)
                if debug:
                    print(f"[DEBUG] log_probs.shape={tuple(log_probs.shape)}")

                preds = greedy_decode(log_probs, self.idx2char)
            return preds[0]

        except Exception as e:
            # Attach traceback / helpful debug info
            import traceback
            tb = traceback.format_exc()
            print("[ERROR] Exception in transcribe():", str(e))
            print(tb)
            # Re-raise so Flask error handler can capture it (or return an error string if you prefer)
            raise



# -----------------------------
# 4. Flask App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

print("Starting server and loading the model...")
asr_model = ASRModel(
    model_path="mini_wav2vec2_full_6000.pth",   # your saved model file
    idx2char_path="idx2char.json"           # saved mapping
)
print("Server is ready to accept API requests.")

# -----------------------------
# 5. API Endpoint
# -----------------------------
@app.route('/transcribe', methods=['POST'])
def handle_transcription():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": 'No selected file'}), 400

        temp_path = "temp_audio_file.wav"
        file.save(temp_path)

        transcription = asr_model.transcribe(temp_path)
        os.remove(temp_path)
        return jsonify({"text": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 6. Run the App
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
