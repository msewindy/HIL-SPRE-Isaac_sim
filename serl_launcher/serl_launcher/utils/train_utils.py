import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict
from flax.training import checkpoints

def ask_for_frame(images_dict):    
    # Create a new figure
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i, (idx, img) in enumerate(images_dict.items()):
        # Display the image
        axes[i].imshow(img)
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Overlay the index number
        axes[i].text(10, 30, str(idx), color='white', fontsize=12, 
                     bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.show(block=False)

    while True:
        try:
            first_success = int(input("First success frame number: "))
            assert first_success in images_dict.keys()
            break
        except:
            continue

    plt.close(fig)
    
    return first_success

def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if not isinstance(offline_batch, dict):
        offline_batch = offline_batch.unfreeze()

    if not isinstance(online_batch, dict):
        online_batch = online_batch.unfreeze()

    for k, v in offline_batch.items():
        if type(v) is dict:
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            batch[k] = jnp.concatenate((offline_batch[k], online_batch[k]), axis=axis)

    return frozen_dict.freeze(batch)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def load_resnet10_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        # Handle JAX version incompatibility when loading pickle files
        # JAX 0.8+ removed 'named_shape' from ShapedArray, but old pickles may contain it
        # Patch JAX's _reconstruct_array to filter out named_shape
        import jax._src.array as jax_array
        import jax._src.core as jax_core
        original_reconstruct = jax_array._reconstruct_array
        original_shaped_array_update = jax_core.ShapedArray.update
        
        def patched_reconstruct(fun, args, arr_state, aval_state):
            # Filter out 'named_shape' if present (JAX 0.8+ doesn't support it)
            # JAX 0.8.3 signature: (fun, args, arr_state, aval_state)
            if isinstance(aval_state, dict) and 'named_shape' in aval_state:
                aval_state = {k: v for k, v in aval_state.items() if k != 'named_shape'}
            return original_reconstruct(fun, args, arr_state, aval_state)
        
        def patched_shaped_array_update(self, **kwargs):
            # Filter out 'named_shape' if present
            kwargs = {k: v for k, v in kwargs.items() if k != 'named_shape'}
            return original_shaped_array_update(self, **kwargs)
        
        # Temporarily patch the functions
        jax_array._reconstruct_array = patched_reconstruct
        jax_core.ShapedArray.update = patched_shaped_array_update
        
        try:
            with open(file_path, "rb") as f:
                encoder_params = pkl.load(f)
        except (TypeError, AttributeError) as e:
            # If loading fails, delete the file and re-download
            import warnings
            warnings.warn(f"Pickle loading failed due to JAX version incompatibility: {e}")
            if os.path.exists(file_path):
                print(f"Removing incompatible pickle file: {file_path}")
                os.remove(file_path)
                # Re-download
                url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
                print(f"Re-downloading from {url}")
                try:
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 1024
                    t = tqdm(total=total_size, unit="iB", unit_scale=True)
                    with open(file_path, "wb") as f_dl:
                        for data in response.iter_content(block_size):
                            t.update(len(data))
                            f_dl.write(data)
                    t.close()
                    print("Re-downloaded. Retrying load...")
                    with open(file_path, "rb") as f_retry:
                        encoder_params = pkl.load(f_retry)
                except Exception as download_error:
                    print(f"Failed to re-download: {download_error}")
                    raise
            else:
                raise
        finally:
            # Restore original functions
            jax_array._reconstruct_array = original_reconstruct
            jax_core.ShapedArray.update = original_shaped_array_update

    param_count = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = agent.state.params

    for image_key in image_keys:
        new_encoder_params = new_params["modules_actor"]["encoder"][
            f"encoder_{image_key}"
        ]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for k in new_encoder_params:
            if k in encoder_params:
                new_encoder_params[k] = encoder_params[k]
                print(f"replaced {k} in pretrained_encoder")

    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent
