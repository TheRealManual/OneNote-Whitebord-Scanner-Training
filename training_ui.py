"""
Training UI - Web interface for ML model training
Provides dataset visualization, training controls, and model management
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import os
import threading
import time
from datetime import datetime
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__, 
            static_folder='training_ui/static',
            template_folder='training_ui/templates')
CORS(app)

# Global training state
training_state = {
    'active': False,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'pixel_acc': 0.0,
    'progress': 0,
    'logs': [],
    'start_time': None
}

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset'
MODELS_DIR = BASE_DIR / 'models'
TRAINING_HISTORY_FILE = MODELS_DIR / 'training_history.json'

# Ensure directories exist
DATASET_DIR.mkdir(exist_ok=True)
(DATASET_DIR / 'images').mkdir(exist_ok=True)
(DATASET_DIR / 'masks').mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_training_history():
    """Load training history from JSON file"""
    if TRAINING_HISTORY_FILE.exists():
        with open(TRAINING_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_training_history(entry):
    """Save training run to history"""
    history = load_training_history()
    history.append(entry)
    with open(TRAINING_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_dataset_stats():
    """Get statistics about the dataset"""
    images_dir = DATASET_DIR / 'images'
    masks_dir = DATASET_DIR / 'masks'
    
    images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    masks = list(masks_dir.glob('*.png'))
    
    # Check for paired images
    paired = 0
    unpaired_images = []
    unpaired_masks = []
    
    image_stems = {img.stem for img in images}
    mask_stems = {mask.stem for mask in masks}
    
    paired = len(image_stems & mask_stems)
    unpaired_images = list(image_stems - mask_stems)
    unpaired_masks = list(mask_stems - image_stems)
    
    return {
        'total_images': len(images),
        'total_masks': len(masks),
        'paired': paired,
        'unpaired_images': unpaired_images,
        'unpaired_masks': unpaired_masks,
        'ready_for_training': paired >= 10
    }


def get_image_preview(image_path, mask_path=None):
    """Generate base64 encoded preview of image and mask"""
    try:
        # Load image
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        result = {'image': f'data:image/jpeg;base64,{img_str}'}
        
        # Load mask if available
        if mask_path and mask_path.exists():
            mask = Image.open(mask_path)
            mask.thumbnail((300, 300))
            
            # Colorize mask for visualization
            mask_array = np.array(mask)
            colored = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
            colored[mask_array == 1] = [0, 255, 0]    # Stroke = green
            colored[mask_array == 2] = [255, 255, 0]  # Smudge = yellow
            colored[mask_array == 3] = [255, 0, 0]    # Shadow = red
            
            colored_img = Image.fromarray(colored)
            buffered = io.BytesIO()
            colored_img.save(buffered, format="PNG")
            mask_str = base64.b64encode(buffered.getvalue()).decode()
            result['mask'] = f'data:image/png;base64,{mask_str}'
        
        return result
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    """Serve the training UI"""
    return render_template('index.html')


@app.route('/api/dataset/stats')
def dataset_stats():
    """Get dataset statistics"""
    stats = get_dataset_stats()
    return jsonify(stats)


@app.route('/api/dataset/images')
def list_images():
    """List all images in dataset"""
    images_dir = DATASET_DIR / 'images'
    masks_dir = DATASET_DIR / 'masks'
    
    images = []
    for img_path in sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png')):
        mask_path = masks_dir / f"{img_path.stem}.png"
        
        images.append({
            'filename': img_path.name,
            'stem': img_path.stem,
            'has_mask': mask_path.exists(),
            'size': img_path.stat().st_size,
            'modified': datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
        })
    
    return jsonify(images)


@app.route('/api/dataset/preview/<filename>')
def preview_image(filename):
    """Get preview of image and mask"""
    images_dir = DATASET_DIR / 'images'
    masks_dir = DATASET_DIR / 'masks'
    
    img_path = images_dir / filename
    stem = Path(filename).stem
    mask_path = masks_dir / f"{stem}.png"
    
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    preview = get_image_preview(img_path, mask_path if mask_path.exists() else None)
    return jsonify(preview)


@app.route('/api/dataset/upload', methods=['POST'])
def upload_files():
    """Upload images or masks"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    file_type = request.form.get('type', 'image')  # 'image' or 'mask'
    
    target_dir = DATASET_DIR / ('images' if file_type == 'image' else 'masks')
    uploaded = []
    
    for file in files:
        if file.filename:
            filepath = target_dir / file.filename
            file.save(filepath)
            uploaded.append(file.filename)
    
    return jsonify({
        'success': True,
        'uploaded': uploaded,
        'count': len(uploaded)
    })


@app.route('/api/dataset/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete an image and its mask"""
    images_dir = DATASET_DIR / 'images'
    masks_dir = DATASET_DIR / 'masks'
    
    img_path = images_dir / filename
    stem = Path(filename).stem
    mask_path = masks_dir / f"{stem}.png"
    
    deleted = []
    if img_path.exists():
        img_path.unlink()
        deleted.append(filename)
    
    if mask_path.exists():
        mask_path.unlink()
        deleted.append(f"{stem}.png")
    
    return jsonify({
        'success': True,
        'deleted': deleted
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training in background thread"""
    if training_state['active']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    config = request.json
    epochs = config.get('epochs', 25)
    batch_size = config.get('batch_size', 4)
    lr = config.get('lr', 0.001)
    
    # Reset state
    training_state.update({
        'active': True,
        'epoch': 0,
        'total_epochs': epochs,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'pixel_acc': 0.0,
        'progress': 0,
        'logs': [],
        'start_time': datetime.now().isoformat()
    })
    
    # Start training in background
    thread = threading.Thread(
        target=run_training,
        args=(epochs, batch_size, lr)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/training/status')
def training_status():
    """Get current training status"""
    return jsonify(training_state)


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training (not implemented - placeholder)"""
    training_state['active'] = False
    training_state['logs'].append('Training stopped by user')
    return jsonify({'success': True})


@app.route('/api/models/list')
def list_models():
    """List all trained models"""
    models = []
    
    for model_path in MODELS_DIR.glob('*.onnx'):
        models.append({
            'filename': model_path.name,
            'size': model_path.stat().st_size,
            'size_mb': round(model_path.stat().st_size / (1024 * 1024), 2),
            'modified': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
            'type': 'INT8' if 'int8' in model_path.name else 'Float32'
        })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify(models)


@app.route('/api/models/history')
def training_history():
    """Get training history"""
    history = load_training_history()
    return jsonify(history)


def run_training(epochs, batch_size, lr):
    """Run training process (called in background thread)"""
    import sys
    import subprocess
    
    try:
        training_state['logs'].append(f'Starting training: {epochs} epochs, batch size {batch_size}, lr {lr}')
        
        # Run training script (use absolute path)
        train_script = BASE_DIR / 'train_segmentation.py'
        cmd = [
            sys.executable,
            str(train_script),
            '--data-dir', str(DATASET_DIR),
            '--output-dir', str(MODELS_DIR),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--lr', str(lr)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line:
                training_state['logs'].append(line)
                
                # Parse progress from output
                if 'Epoch' in line:
                    try:
                        # Extract epoch number
                        parts = line.split('/')
                        if len(parts) >= 2:
                            current = int(parts[0].split()[-1])
                            training_state['epoch'] = current
                            training_state['progress'] = int((current / epochs) * 100)
                    except:
                        pass
                
                if 'Train Loss:' in line:
                    try:
                        # Extract losses
                        for part in line.split('-'):
                            if 'Train Loss:' in part:
                                training_state['train_loss'] = float(part.split(':')[1].strip())
                            if 'Val Loss:' in part:
                                training_state['val_loss'] = float(part.split(':')[1].strip())
                            if 'Pixel Acc:' in part:
                                training_state['pixel_acc'] = float(part.split(':')[1].strip())
                    except:
                        pass
        
        process.wait()
        
        # Training complete
        if process.returncode == 0:
            training_state['logs'].append('Training completed successfully!')
            training_state['progress'] = 100
            
            # Save to history
            history_entry = {
                'timestamp': training_state['start_time'],
                'duration': (datetime.now() - datetime.fromisoformat(training_state['start_time'])).total_seconds(),
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'final_train_loss': training_state['train_loss'],
                'final_val_loss': training_state['val_loss'],
                'final_pixel_acc': training_state['pixel_acc'],
                'model_file': 'whiteboard_seg_int8.onnx'
            }
            save_training_history(history_entry)
        else:
            training_state['logs'].append(f'Training failed with exit code {process.returncode}')
        
    except Exception as e:
        training_state['logs'].append(f'Error: {str(e)}')
    finally:
        training_state['active'] = False


if __name__ == '__main__':
    print("=" * 60)
    print("ML Training UI Server")
    print("=" * 60)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print("")
    print("Starting server at http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
