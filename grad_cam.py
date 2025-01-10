import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def compute_gradcam(model, image, class_index, layer_name=None):
    """
    Compute Grad-CAM heatmap for a single image and a given class index.

    Arguments:
      model: Keras model.
      image: Preprocessed image tensor of shape (H, W, C).
      class_index: Target class index for which to compute Grad-CAM.
      layer_name: Name of the convolutional (or attention) layer from which
                  to compute gradients. If None, the last conv layer is used for CNN,
                  for transformer-based models the final transformer block output is picked before pooling.
                  (recommended to specify layer for ResNet50)
    """
    if layer_name is None:
        # Attempt to pick last convolutional layer for CNN/ResNet
        # For transformer models, the penultimate LayerNormalization is picked.
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            # If no Conv2D found, pick the last normalization layer (transformers)
            for layer in reversed(model.layers):
                if isinstance(layer, layers.LayerNormalization):
                    layer_name = layer.name
                    break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # Add a batch dimension and track gradient tape
    with tf.GradientTape() as tape:
        inputs = tf.expand_dims(image, axis=0)
        feature_maps, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    # Compute gradients of loss wrt feature maps
    grads = tape.gradient(loss, feature_maps)

    # If it's a CNN-like feature map: shape [1, h, w, c]
    # If it's a ViT patch embedding: shape [1, num_patches, dim]
    if len(feature_maps.shape) == 4:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        feature_maps = feature_maps[0]  # shape: (h, w, c)
        heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    else:
        # transformers
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        feature_maps = feature_maps[0]  # shape: (num_patches, dim)
        heatmap = tf.tensordot(feature_maps, pooled_grads, axes=(1, 0))
        # Reshape heatmap back to patch grid
        patch_dim = input_shape[0] // patch_size
        heatmap = tf.reshape(heatmap, (patch_dim, patch_dim))

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, cmap='jet'):
    # Rescale heatmap to 0-1
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(cmap)
    colored_map = cmap(np.arange(256))[:,:3]
    heatmap_color = colored_map[heatmap]

    heatmap_color = np.uint8(255 * heatmap_color)
    # Resize heatmap_color to match img size
    heatmap_color = tf.image.resize(
        tf.constant(heatmap_color), (img.shape[0], img.shape[1])
    )
    heatmap_color = heatmap_color.numpy().astype(np.uint8)

    overlayed_img = heatmap_color * alpha + img*255 * (1 - alpha)
    overlayed_img = np.uint8(overlayed_img)
    return overlayed_img