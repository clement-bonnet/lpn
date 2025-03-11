import os
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from huggingface_hub import hf_hub_download


DATASETS_BASE_PATH = "src/datasets"


def load_datasets(dataset_dirs: list[str], use_hf: bool) -> list[tuple[chex.Array, chex.Array, chex.Array]]:
    """Load datasets from the given directories.

    Args:
        dataset_dirs: List of directories containing the datasets.
        use_hf: Whether to use the HF hub to download the datasets.

    Returns:
        List of tuples containing the grids and shapes of the datasets.
    """

    datasets = []

    if use_hf:
        for dataset_dir in dataset_dirs:
            grids = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "grids.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            shapes = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "shapes.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            try:
                program_ids = np.load(
                    hf_hub_download(
                        repo_id="arcenv/arc_datasets",
                        filename=os.path.join(dataset_dir, "program_ids.npy"),
                        repo_type="dataset",
                    )
                ).astype("uint32")
            except:
                program_ids = jnp.zeros(grids.shape[0], dtype=jnp.uint32)

            datasets.append((grids, shapes, program_ids))

    else:
        for dataset_dir in dataset_dirs:
            grids = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "grids.npy")).astype("uint8")
            shapes = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "shapes.npy")).astype("uint8")

            try:
                program_ids = np.load(
                    os.path.join(DATASETS_BASE_PATH, dataset_dir, "program_ids.npy")
                ).astype("uint32")
            except:
                program_ids = jnp.zeros(grids.shape[0], dtype=jnp.uint32)

            datasets.append((grids, shapes, program_ids))
    return datasets


def shuffle_dataset_into_batches(
    dataset_grids: chex.Array, dataset_shapes: chex.Array, batch_size: int, key: chex.PRNGKey
) -> tuple[chex.Array, chex.Array]:
    if dataset_grids.shape[0] != dataset_shapes.shape[0]:
        raise ValueError("Dataset grids and shapes must have the same length.")

    # Shuffle the dataset.
    shuffled_indices = jax.random.permutation(key, len(dataset_grids))
    shuffled_grids = dataset_grids[shuffled_indices]
    shuffled_shapes = dataset_shapes[shuffled_indices]

    # Determine the number of batches.
    num_batches = len(dataset_grids) // batch_size
    if num_batches < 1:
        raise ValueError(f"Got dataset size: {len(dataset_grids)} < batch size: {batch_size}.")

    # Reshape the dataset into batches and crop the last batch if necessary.
    batched_grids = shuffled_grids[: num_batches * batch_size].reshape(
        (num_batches, batch_size, *dataset_grids.shape[1:])
    )
    batched_shapes = shuffled_shapes[: num_batches * batch_size].reshape(
        (num_batches, batch_size, *dataset_shapes.shape[1:])
    )

    return batched_grids, batched_shapes


def _apply_rotation(grid: chex.Array, grid_shape: chex.Array, k: int) -> tuple[chex.Array, chex.Array]:
    assert grid.ndim == 2 and grid_shape.ndim == 1

    def rotate_once(_, carry: tuple[chex.Array, chex.Array]) -> tuple[chex.Array, chex.Array]:
        grid, grid_shape = carry
        grid = jnp.rot90(grid, k=-1)
        # Roll the columns to the left until the first non-padded column is at the first position.
        num_rows = grid_shape[0].astype(jnp.int32)
        grid = jax.lax.fori_loop(
            0, grid.shape[0] - num_rows, lambda _, x: jnp.roll(x, shift=-1, axis=-1), grid
        )
        # Swap the rows and cols.
        grid_shape = grid_shape[::-1]
        return grid, grid_shape

    grid, grid_shape = jax.lax.fori_loop(0, k, rotate_once, (grid, grid_shape))
    return grid, grid_shape


def _apply_color_permutation(grid: chex.Array, key: chex.PRNGKey) -> chex.Array:
    # Exempt black (0).
    non_exempt = jnp.array([i for i in range(10) if i not in [0]], dtype=grid.dtype)
    permutation = jax.random.permutation(key, np.arange(len(non_exempt), dtype=grid.dtype))
    permuted_non_exempt = jnp.array([non_exempt[i] for i in permutation], dtype=grid.dtype)
    color_map = jnp.arange(10, dtype=grid.dtype).at[non_exempt].set(permuted_non_exempt)
    return color_map[grid]


def data_augmentation_fn(
    grids: chex.Array, shapes: chex.Array, key: chex.PRNGKey
) -> tuple[chex.Array, chex.Array]:
    """Apply data augmentation to the grids and shapes.

    Args:
        grids: The input grids. Shape (*B, N, R, C, 2).
        shapes: The shapes of the grids. Shape (*B, N, 2, 2).
        key: The random key.

    Returns:
        The augmented grids and shapes.
    """
    rotation_key, color_key = jax.random.split(key, 2)

    rotation_indices = jax.random.randint(rotation_key, grids.shape[:-4], 0, 4)
    apply_rotation_fn = _apply_rotation
    # vmap over the input/output channel (use same rotation).
    apply_rotation_fn = jax.vmap(apply_rotation_fn, in_axes=(-1, -1, None), out_axes=-1)
    # vmap over the pairs from the same task (use same rotation).
    apply_rotation_fn = jax.vmap(apply_rotation_fn, in_axes=(0, 0, None))
    # vmap over the batch dims if any (use different rotations).
    for _ in range(rotation_indices.ndim):
        apply_rotation_fn = jax.vmap(apply_rotation_fn)

    grids, shapes = apply_rotation_fn(grids, shapes, rotation_indices)

    color_keys = jax.random.split(color_key, grids.shape[:-4])
    apply_color_permutation_fn = _apply_color_permutation
    # vmap over the batch dims if any (use different color permutations).
    for _ in range(color_keys.ndim - 1):
        apply_color_permutation_fn = jax.vmap(apply_color_permutation_fn)

    grids = apply_color_permutation_fn(grids, color_keys)

    return grids, shapes


def make_leave_one_out(array: chex.Array, axis: int) -> chex.Array:
    """
    Args:
        array: shape (*B, N, *H).
        axis: The axis where N appears.

    Returns:
        Array of shape (*B, N, N-1, *H).
    """
    axis = axis % array.ndim
    output = []
    for i in range(array.shape[axis]):
        array_before = jax.lax.slice_in_dim(array, 0, i, axis=axis)
        array_after = jax.lax.slice_in_dim(array, i + 1, array.shape[axis], axis=axis)
        output.append(jnp.concatenate([array_before, array_after], axis=axis))
    output = jnp.stack(output, axis=axis)
    return output