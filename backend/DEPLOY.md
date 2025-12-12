# Clara Deployment Guide

This guide covers deploying Clara to Modal with models hosted on HuggingFace.

## Prerequisites

1. **HuggingFace Account** - https://huggingface.co
2. **Modal Account** - https://modal.com (free tier available)
3. **Models uploaded to HuggingFace** (see Step 1)

## Step 1: Upload Models to HuggingFace

Use the Colab notebook `upload_models_to_hf.ipynb`:

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `upload_models_to_hf.ipynb`
3. Update the `DRIVE_BASE` path to your Google Drive model location
4. Run all cells (this uploads ~7GB for knowledge brain + ~150MB for LoRAs)

After upload, your models will be at:
- `ChrisHartline/clara-knowledge` - Phi-3 knowledge brain
- `ChrisHartline/clara-warmth` - Warmth personality LoRA
- `ChrisHartline/clara-playful` - Playful personality LoRA
- `ChrisHartline/clara-encouragement` - Encouragement personality LoRA

## Step 2: Set Up Modal

```bash
# Install Modal CLI
pip install modal

# Login to Modal (opens browser)
modal setup

# Optional: Add HuggingFace token as secret
modal secret create huggingface-token HUGGINGFACE_TOKEN=<your-token>
```

## Step 3: Deploy to Modal

```bash
cd backend

# Test locally first (uses Modal's local runner)
modal serve modal_app.py

# Deploy to production
modal deploy modal_app.py
```

After deployment, Modal will show your URL:
```
https://<username>--clara-api-fastapi-app.modal.run
```

## Step 4: Update Frontend Config

Edit `config.ts` in the frontend:

```typescript
// Update this with your Modal URL
const MODAL_URL = 'https://<username>--clara-api-fastapi-app.modal.run';
```

## Step 5: Build & Deploy Frontend

For Vercel:
```bash
npm run build
vercel --prod
```

For local testing:
```bash
npm run dev
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/tools` | GET | List available tools |
| `/api/chat` | POST | REST chat endpoint |
| `/ws/chat` | WS | WebSocket chat |

## Monitoring

View logs and metrics in Modal dashboard:
https://modal.com/apps

## Costs

Modal pricing (as of 2024):
- **T4 GPU**: ~$0.000164/sec ($0.59/hr)
- **A10G GPU**: ~$0.000306/sec ($1.10/hr)
- **Container idle**: Free for first 5 min, then charged

With `container_idle_timeout=300` (5 min), you only pay when actively chatting.

## Upgrading GPU

To use a more powerful GPU, edit `modal_app.py`:

```python
@app.cls(
    gpu=gpu.A10G(),  # Change from T4 to A10G
    # ...
)
```

## Troubleshooting

### "Model not found"
- Verify models are uploaded to HuggingFace
- Check the HF username matches in `modal_app.py`
- Models are public (or add HF token as Modal secret)

### "Out of memory"
- Enable `load_in_4bit=True` (already set)
- Upgrade to A10G GPU (24GB VRAM)
- Reduce `max_new_tokens` in generation

### WebSocket connection fails
- Check CORS settings in Modal app
- Verify frontend URL matches Modal deployment
- Check browser console for specific errors

## Migration to GCP

When ready to migrate from Modal to GCP:

1. The FastAPI code is fully portable
2. Use Cloud Run or GKE for container hosting
3. Use Vertex AI for GPU inference (or self-managed)
4. Keep models on HuggingFace or move to GCS

The core `modal_app.py` FastAPI code can be extracted and deployed anywhere.
