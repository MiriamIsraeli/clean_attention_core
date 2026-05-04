"""End-to-end test: upload WAV via HTTP, verify noise cancellation."""
import requests, json, numpy as np

WAV_PATH = r"C:\Users\ישראלי\OneDrive\Desktop\miriam - project\0b136f.wav"

with open(WAV_PATH, "rb") as f:
    resp = requests.post("http://localhost:5000/stream", files={"audio": f}, stream=True)

print(f"Status: {resp.status_code}")
chunks = []
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode()
    if not line.startswith("data: "):
        continue
    data = json.loads(line[6:])
    
    if data["type"] == "meta":
        sr = data["sampleRate"]
        total = data["totalSamples"]
        chunk_sz = data["chunkSize"]
        print(f"Meta: sr={sr}, total={total}, chunkSize={chunk_sz}")
    
    elif data["type"] == "chunk":
        chunks.append(data)
        cid = data["id"]
        rms_o = data["rmsOriginal"]
        rms_c = data["rmsCleaned"]
        rms_a = data["rmsAntiNoise"]
        red = 20 * np.log10(rms_c / rms_o) if rms_o > 1e-10 else -99
        if cid < 5 or cid % 30 == 0:
            print(f"  Chunk {cid:3d}: orig={rms_o:.4f} clean={rms_c:.4f} anti={rms_a:.4f} {red:+.1f}dB")
    
    elif data["type"] == "done":
        print(f"\nDone! Total chunks: {len(chunks)}")
        all_samples = sum(len(c["cleaned"]) for c in chunks)
        print(f"Total samples: {all_samples} ({all_samples/sr:.2f}s)")
        
        # First chunk
        c0 = np.array(chunks[0]["cleaned"])
        print(f"First chunk: {len(c0)} samples, max={np.max(np.abs(c0)):.4f}, rms={np.sqrt(np.mean(c0**2)):.4f}")
        
        # Last chunk
        cl = np.array(chunks[-1]["cleaned"])
        print(f"Last chunk: {len(cl)} samples, max={np.max(np.abs(cl)):.6f}, rms={np.sqrt(np.mean(cl**2)):.6f}")
        
        # Overall reduction
        all_clean = np.concatenate([np.array(c["cleaned"]) for c in chunks])
        skip = int(0.3 * sr)
        rms_total = np.sqrt(np.mean(all_clean[skip:]**2))
        print(f"\nOverall cleaned RMS (after 0.3s): {rms_total:.4f}")
        print(f"PASS!" if rms_total < 0.05 else "FAIL - noise not cancelled!")
