def analyze_new_sample(image_input, model, embedding_model, pca, kmeans,
                       entropy_threshold, style_emnist, pred_emnist, entropy_emnist, y_test_emnist):
    """
    Returns a dictionary with predicted label, confidence, predictive entropy,
    style cluster, historical cluster accuracy, historical cluster entropy,
    and a rejection flag for a single new sample.
    
    image_input: path (str), bytes, or numpy array.
    """
    import tensorflow as tf
    import numpy as np
    import io
    from PIL import Image

    # --- Robust Preprocessing for MNIST/EMNIST ---
    # 1. Decode & Normalize to 0-1 Float
    # This handles both png/jpg bytes and existing numpy arrays
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            img = tf.io.read_file(image_input)
            img = tf.image.decode_png(img, channels=1)
        else:
            img = tf.image.decode_image(image_input, channels=1) # May return 3D or 4D
            
        img = tf.cast(img, tf.float32)
        if tf.reduce_max(img) > 1.0:
            img = img / 255.0
            
    elif isinstance(image_input, np.ndarray):
        img = tf.convert_to_tensor(image_input, dtype=tf.float32)
        # Assuming if user passed array, they might not be normalized
        if tf.reduce_max(img) > 1.0:
            img = img / 255.0
        if len(img.shape) == 2:
             img = tf.expand_dims(img, axis=-1)
    else:
        raise ValueError("Unsupported image input type")

    # Ensure we are working with 3D (H, W, C)
    if len(img.shape) == 4:
        img = img[0] # Take first if batch

    # 2. Invert if necessary (Standardize to White-on-Black)
    img_np = img.numpy()
    if len(img_np.shape) == 3: img_np = img_np[:, :, 0]
    
    # Check corners or mean to detect background
    is_dark_bg = np.mean(img_np) < 0.5
    if is_dark_bg:
        img_np = 1.0 - img_np

    # 3. Crop to Bounding Box
    rows, cols = np.where(img_np < 0.5)
    if len(rows) > 0:
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        cropped = img_np[y_min:y_max+1, x_min:x_max+1]
        
        # 4. Resize longest side to 20 pixels
        h, w = cropped.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize using TF
        cropped_tf = tf.expand_dims(tf.expand_dims(cropped, -1), 0) # 1, H, W, 1
        resized_tf = tf.image.resize(cropped_tf, (new_h, new_w))
        resized = resized_tf.numpy()[0, :, :, 0]
        
        # 5. Pad to 28x28 (Center it)
        final_img = np.ones((28, 28), dtype=np.float32)
        pad_y = (28 - new_h) // 2
        pad_x = (28 - new_w) // 2
        final_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        img = tf.convert_to_tensor(final_img, dtype=tf.float32)
    else:
        # Blank image processing
        img = tf.image.resize(img, (28, 28)) # Just resize raw
        img = img[:, :, 0] # flattening to 2D for consistency logic below

    # Shape Guarantees
    # img is now (28, 28) float32 0-1
    img = tf.reshape(img, (28, 28, 1))

    # Debug Image Generation
    debug_np = (img.numpy()[:, :, 0] * 255).astype(np.uint8)
    debug_pil = Image.fromarray(debug_np)
    buf = io.BytesIO()
    debug_pil.save(buf, format="PNG")
    import base64
    debug_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Batch Dimension for Model
    img = tf.expand_dims(img, axis=0) # (1, 28, 28, 1)
    
    print(f"DEBUG: Input shape to model: {img.shape}")

    # MC Dropout prediction
    T = 50
    preds = np.array([model(img, training=True).numpy() for _ in range(T)])
    mean_pred = preds.mean(axis=0)
    # var_pred = preds.var(axis=0) # Unused
    entropy_val = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)[0]
    pred_label = np.argmax(mean_pred)
    pred_confidence = mean_pred.max()

    # Style embedding + PCA + cluster
    embedding = embedding_model.predict(img)
    embedding_pca = pca.transform(embedding)
    style_cluster = kmeans.predict(embedding_pca)[0]

    # Historical cluster info
    if len(y_test_emnist) > 0 and len(pred_emnist) == len(y_test_emnist):
        cluster_idx = style_emnist == style_cluster
        historical_acc = (pred_emnist[cluster_idx] == y_test_emnist[cluster_idx]).mean()
    else:
        historical_acc = -1.0 # Indicator for missing data

    historical_entropy = entropy_emnist[style_emnist == style_cluster].mean()

    # Rejection flag
    reject_flag = entropy_val > entropy_threshold

    return {
        "pred_label": int(pred_label),
        "confidence": float(pred_confidence),
        "entropy": float(entropy_val),
        "style_cluster": int(style_cluster),
        "historical_cluster_acc": float(historical_acc),
        "historical_cluster_entropy": float(historical_entropy),
        "rejected": bool(reject_flag),
        "debug_image": debug_b64
    }

# --- Load models ---
from tensorflow.keras.models import load_model
model = load_model("char_model.h5")
embedding_model = load_model("embedding_model.h5")

# --- Load PCA and KMeans ---
import joblib
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")

# --- Load metadata ---
import json
import numpy as np
import pandas as pd

with open("app_meta.json") as f:
    app_meta = json.load(f)

entropy_threshold = app_meta["entropy_threshold"]
style_emnist = np.array(app_meta["style_emnist"])
pred_emnist = np.array(app_meta["pred_emnist"])
entropy_emnist = np.array(app_meta["entropy_emnist"])

# --- Load y_test from CSV ---
try:
    print("Loading dataset labels from CSV...")
    # Assuming first column is the label. 
    # Reading only the first column to save memory if we only need labels.
    df = pd.read_csv("A_Z Handwritten Data.csv", usecols=[0], header=None) 
    # Note: Using header=None because typically these datasets don't have headers, 
    # but if it does, this might include the header string. 
    # 'A_Z Handwritten Data.csv' usually has no header and 0th col is label.
    y_test_emnist = df.iloc[:, 0].values
    
    # Defensive check: Length mismatch handling
    if len(y_test_emnist) != len(pred_emnist):
        print(f"Warning: Dataset length ({len(y_test_emnist)}) matches metadata ({len(pred_emnist)})? {len(y_test_emnist) == len(pred_emnist)}")
        # We might need to slice if one is larger, or if user tested on a subset.
        # For now, let's take the minimum length to avoid crash
        min_len = min(len(y_test_emnist), len(pred_emnist))
        y_test_emnist = y_test_emnist[:min_len]
        # We also need to slice the others to match so indices work
        # but changing global 'pred_emnist' might be risky if 'style_emnist' relies on it. 
        # Better constraint: Ensure analyzing sample uses safe indices or just warns.
        # But for 'historical_acc', boolean indexing requires same shape.
        # Let's truncate everything to min_len just to be safe if they mostly align.
        style_emnist = style_emnist[:min_len]
        pred_emnist = pred_emnist[:min_len]
        entropy_emnist = entropy_emnist[:min_len]
        
    print(f"Loaded y_test with {len(y_test_emnist)} samples.")

except Exception as e:
    print(f"Error loading CSV: {e}")
    y_test_emnist = np.array([])

print("All models, PCA, KMeans, and metadata loaded successfully!")