from typing import Optional
import chex
import jax.numpy as jnp
import jax
from flax import linen as nn

from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig, TransformerLayer


class EncoderTransformer(nn.Module):
    config: EncoderTransformerConfig

    @nn.compact
    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
    ) -> tuple[chex.Array, Optional[chex.Array]]:
        """Applies Transformer Encoder on the (input, output) pairs.

        Args:
            pairs: input data as tokens. Shape (*B, R, C, 2).
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
                Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            dropout_eval: if false dropout is applied otherwise it is not.

        Returns:
            latent_mu: output of shape (*B, H) representing the mean latent embeddings of the (input, output)
                pairs.
            latent_logvar: output of shape (*B, H) representing the log-variance of the latent embeddings of
                the (input, output) pairs.
        """

        x = self.embed_grids(pairs, grid_shapes, dropout_eval)

        # Transformer block.
        pad_mask = self.make_pad_mask(grid_shapes)
        for _ in range(self.config.num_layers):
            x = TransformerLayer(self.config.transformer_layer)(
                embeddings=x,
                dropout_eval=dropout_eval,
                pad_mask=pad_mask,
            )

        # Extract the CLS embedding.
        cls_embed = x[..., 0, :]
        # Project the cls embedding to the program space.
        cls_embed = nn.LayerNorm(
            dtype=self.config.dtype,
            use_bias=self.config.transformer_layer.use_bias,
            use_scale=False,
            name="cls_layer_norm",
        )(cls_embed)
        latent_mu = nn.Dense(
            self.config.latent_dim, use_bias=self.config.latent_projection_bias, dtype=self.config.dtype
        )(cls_embed).astype(jnp.float32)
        if self.config.variational:
            latent_logvar = nn.Dense(
                self.config.latent_dim, use_bias=self.config.latent_projection_bias, dtype=self.config.dtype
            )(cls_embed).astype(jnp.float32)
        else:
            latent_logvar = None

        return latent_mu, latent_logvar

    def embed_grids(self, pairs: chex.Array, grid_shapes: chex.Array, dropout_eval: bool) -> chex.Array:
        config = self.config

        # Position embedding block.
        if self.config.scaled_position_embeddings:
            pos_row_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.zeros(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.zeros(config.max_cols, dtype=jnp.uint8))
            pos_row_embeds = jnp.arange(1, config.max_rows + 1)[:, None] * pos_row_embed
            pos_col_embeds = jnp.arange(1, config.max_cols + 1)[:, None] * pos_col_embed
            pos_embed = pos_row_embeds[:, None, None, :] + pos_col_embeds[None, :, None, :]
        else:
            pos_row_embed = nn.Embed(
                num_embeddings=config.max_rows,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.arange(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=config.max_cols,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.arange(config.max_cols, dtype=jnp.uint8))
            pos_embed = pos_row_embed[:, None, None, :] + pos_col_embed[None, :, None, :]

        # Colors embedding block.
        colors_embed = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            name="colors_embed",
        )(pairs)

        # Channels embedding block.
        channels_embed = nn.Embed(
            num_embeddings=2,
            features=config.emb_dim,
            dtype=config.dtype,
            name="channels_embed",
        )(jnp.arange(2, dtype=jnp.uint8))

        # Combine all the embeddings into a sequence x of shape (*B, 1+2*(R*C), H)
        x = colors_embed + pos_embed + channels_embed
        # Flatten the rows, columns and channels.
        x = jnp.reshape(x, (*x.shape[:-4], -1, x.shape[-1]))  # (*B, 2*R*C, H)

        # Embed the grid shape tokens.
        # TODO: potentially switch grid_shapes embeddings to linear embedding for better interpolation
        grid_shapes_row_embed = nn.Embed(
            num_embeddings=config.max_rows,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_row_embed",
        )(grid_shapes[..., 0, :] - 1)
        grid_shapes_row_embed += channels_embed
        grid_shapes_col_embed = nn.Embed(
            num_embeddings=config.max_cols,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_col_embed",
        )(grid_shapes[..., 1, :] - 1)
        grid_shapes_col_embed += channels_embed
        grid_shapes_embed = jnp.concatenate([grid_shapes_row_embed, grid_shapes_col_embed], axis=-2)
        x = jnp.concatenate([grid_shapes_embed, x], axis=-2)  # (*B, 4+2*R*C, H)

        # Add the cls token.
        cls_token = nn.Embed(
            num_embeddings=1,
            features=config.emb_dim,
            dtype=config.dtype,
            name="cls_token",
        )(jnp.zeros_like(x[..., 0:1, 0], jnp.uint8))
        x = jnp.concatenate([cls_token, x], axis=-2)  # (*B, 1+4+2*R*C, H)
        assert x.shape[-2] == 1 + 4 + 2 * config.max_len  # 1805
        x = nn.Dropout(rate=config.transformer_layer.dropout_rate, name="embed_dropout")(x, dropout_eval)
        return x

    def make_pad_mask(self, grid_shapes: chex.Array) -> chex.Array:
        """Make the pad mask False outside of the grid shapes and True inside.

        Args:
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].

        Returns:
            pad mask of shape (*B, 1, T, T) with T = 1 + 4 + 2 * max_rows * max_cols.
        """
        batch_ndims = len(grid_shapes.shape[:-2])
        row_arange_broadcast = jnp.arange(self.config.max_rows).reshape(
            (*batch_ndims * (1,), self.config.max_rows, 1)
        )
        row_mask = row_arange_broadcast < grid_shapes[..., 0:1, :]
        col_arange_broadcast = jnp.arange(self.config.max_cols).reshape(
            (*batch_ndims * (1,), self.config.max_cols, 1)
        )
        col_mask = col_arange_broadcast < grid_shapes[..., 1:2, :]
        pad_mask = row_mask[..., :, None, :] & col_mask[..., None, :, :]
        # Flatten the rows, columns and channels.
        pad_mask = jnp.reshape(pad_mask, (*pad_mask.shape[:-3], 1, -1))
        # Add the masks corresponding to the cls token and grid shapes tokens.
        pad_mask = jnp.concatenate([jnp.ones((*pad_mask.shape[:-1], 1 + 4), bool), pad_mask], axis=-1)
        # Outer product to make the self-attention mask.
        pad_mask = pad_mask[..., :, None] & pad_mask[..., None, :]
        return pad_mask


class DecoderTransformer(nn.Module):
    config: DecoderTransformerConfig

    @nn.compact
    def __call__(
        self,
        input_seq: chex.Array,
        output_seq: chex.Array,
        context: chex.Array,
        dropout_eval: bool,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Applies Transformer Decoder on the task outputs to reconstruct them given a context latent.

        Args:
            input_seq: flattened task input grid as tokens. Shape (*B, 2+R*C).
                - 2 for the grid shapes. Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
                - R: max number of rows.
                - C: max number of columns.
            output_seq: flattened task output grid as tokens. Shape (*B, 2+R*C).
                - 2 for the grid shapes. Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
                - R: max number of rows.
                - C: max number of columns.
            context: latent program of the task. Shape (*B, H).
            dropout_eval: if false dropout is applied otherwise it is not.

        Returns:
            grid_shape_row_logits of shape (*B, R) representing the logits for the grid shape row.
            grid_shape_col_logits of shape (*B, C) representing the logits for the grid shape column.
            output_grid_logits of shape (*B, R*C, V) representing the logits of the next-token predictions.
        """
        x = self.embed_inputs(input_seq, output_seq, context, dropout_eval)

        # Transformer block.
        causal_pad_mask = self.make_causal_pad_mask(
            input_grid_shape=input_seq[..., :2], output_grid_shape=output_seq[..., :2]
        )
        for _ in range(self.config.num_layers):
            x = TransformerLayer(self.config.transformer_layer)(
                embeddings=x,
                dropout_eval=dropout_eval,
                pad_mask=causal_pad_mask,
            )

        grid_shape_row_logits, grid_shape_col_logits, output_grid_logits = self.extract_logits(
            x, input_seq.shape[-1]
        )
        # In case of mixed precision, we need to cast the output back to float32.
        grid_shape_row_logits = grid_shape_row_logits.astype(jnp.float32)
        grid_shape_col_logits = grid_shape_col_logits.astype(jnp.float32)
        output_grid_logits = output_grid_logits.astype(jnp.float32)

        return grid_shape_row_logits, grid_shape_col_logits, output_grid_logits

    def embed_inputs(
        self, input_seq: chex.Array, output_seq: chex.Array, context: chex.Array, dropout_eval: bool
    ) -> chex.Array:
        config = self.config

        # Context embedding block.
        context_embed = nn.Dense(
            config.emb_dim, config.transformer_layer.use_bias, config.dtype, name="context_embed"
        )(context)

        # Position embedding block.
        if self.config.scaled_position_embeddings:
            pos_row_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.zeros(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.zeros(config.max_cols, dtype=jnp.uint8))
            pos_row_embeds = jnp.arange(1, config.max_rows + 1)[:, None] * pos_row_embed
            pos_col_embeds = jnp.arange(1, config.max_cols + 1)[:, None] * pos_col_embed
            pos_embed = pos_row_embeds[:, None] + pos_col_embeds[None, :]
        else:
            pos_row_embed = nn.Embed(
                num_embeddings=config.max_rows,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.arange(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=config.max_cols,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.arange(config.max_cols, dtype=jnp.uint8))
            pos_embed = pos_row_embed[:, None] + pos_col_embed[None, :]

        if self.config.next_position_embeddings:
            input_num_cols, output_num_cols = input_seq[..., 1], output_seq[..., 1]
            shifted_left_pos_embed = jnp.roll(pos_embed, shift=-1, axis=-2)
            first_col_embed = pos_embed[:, 0, :]
            shifted_up_first_col_embed = jnp.roll(first_col_embed, shift=-1, axis=-2)
            batch_ndims = len(input_num_cols.shape)
            arange_broadcast = jnp.arange(config.max_cols).reshape((*batch_ndims * (1,), config.max_cols))
            if self.config.next_position_embeddings_new_input_embeds:
                # Generate new postion embeddings for the input tokens only.
                if self.config.scaled_position_embeddings:
                    input_pos_row_embed = nn.Embed(
                        num_embeddings=1,
                        features=config.emb_dim,
                        dtype=config.dtype,
                        name="input_pos_row_embed",
                    )(jnp.zeros(config.max_rows, dtype=jnp.uint8))
                    input_pos_col_embed = nn.Embed(
                        num_embeddings=1,
                        features=config.emb_dim,
                        dtype=config.dtype,
                        name="input_pos_col_embed",
                    )(jnp.zeros(config.max_cols, dtype=jnp.uint8))
                    input_pos_row_embeds = jnp.arange(1, config.max_rows + 1)[:, None] * input_pos_row_embed
                    input_pos_col_embeds = jnp.arange(1, config.max_cols + 1)[:, None] * input_pos_col_embed
                    input_pos_embeds = input_pos_row_embeds[:, None] + input_pos_col_embeds[None, :]
                else:
                    input_pos_row_embed = nn.Embed(
                        num_embeddings=config.max_rows,
                        features=config.emb_dim,
                        dtype=config.dtype,
                        name="input_pos_row_embed",
                    )(jnp.arange(config.max_rows, dtype=jnp.uint8))
                    input_pos_col_embed = nn.Embed(
                        num_embeddings=config.max_cols,
                        features=config.emb_dim,
                        dtype=config.dtype,
                        name="input_pos_col_embed",
                    )(jnp.arange(config.max_cols, dtype=jnp.uint8))
                    input_pos_embeds = input_pos_row_embed[:, None] + input_pos_col_embed[None, :]
            else:
                # Reuse the positon embeddings for the input tokens.
                input_pos_embeds = pos_embed

            output_pos_embeds = jnp.where(
                jnp.expand_dims(arange_broadcast == output_num_cols[..., None] - 1, axis=(-3, -1)),
                shifted_up_first_col_embed[:, None],
                shifted_left_pos_embed,
            )
            input_pos_embeds = input_pos_embeds.reshape((*input_pos_embeds.shape[:-3], -1, config.emb_dim))
            output_pos_embeds = output_pos_embeds.reshape((*output_pos_embeds.shape[:-3], -1, config.emb_dim))
        else:
            pos_embeds = jnp.reshape(pos_embed, (-1, config.emb_dim))
            input_pos_embeds, output_pos_embeds = pos_embeds, pos_embeds

        # Grid shapes embedding block.
        # TODO: potentially switch grid_shapes embeddings to linear embedding for better interpolation
        grid_shapes_row_embed_layer = nn.Embed(
            num_embeddings=config.max_rows,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_row_embed",
        )
        input_grid_shapes_row_embed = grid_shapes_row_embed_layer(input_seq[..., 0] - 1)
        output_grid_shapes_row_embed = grid_shapes_row_embed_layer(output_seq[..., 0] - 1)
        grid_shapes_col_embed_layer = nn.Embed(
            num_embeddings=config.max_cols,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_col_embed",
        )
        input_grid_shapes_col_embed = grid_shapes_col_embed_layer(input_seq[..., 1] - 1)
        output_grid_shapes_col_embed = grid_shapes_col_embed_layer(output_seq[..., 1] - 1)

        # Colors embedding block.
        colors_embed_layer = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            name="colors_embed",
        )
        input_colors_embed = colors_embed_layer(input_seq[..., 2:])
        output_colors_embed = colors_embed_layer(output_seq[..., 2:])
        input_embed, output_embed = nn.Embed(
            num_embeddings=2,
            features=config.emb_dim,
            dtype=config.dtype,
            name="input_output_embed",
        )(jnp.arange(2, dtype=jnp.uint8))

        # Combining all the embeddings into a sequence x of shape (*B, 1+2*(2+R*C), H)
        x_input_shape_row = jnp.expand_dims(input_grid_shapes_row_embed + input_embed, axis=-2)
        x_input_shape_col = jnp.expand_dims(input_grid_shapes_col_embed + input_embed, axis=-2)
        x_input_colors = input_colors_embed + input_pos_embeds + input_embed
        x_context = jnp.expand_dims(context_embed, axis=-2)
        x_output_shape_row = jnp.expand_dims(output_grid_shapes_row_embed + output_embed, axis=-2)
        x_output_shape_col = jnp.expand_dims(output_grid_shapes_col_embed + output_embed, axis=-2)
        x_output_colors = output_colors_embed + output_pos_embeds + output_embed
        x = jnp.concatenate(
            [
                x_input_shape_row,
                x_input_shape_col,
                x_input_colors,
                x_context,
                x_output_shape_row,
                x_output_shape_col,
                x_output_colors,
            ],
            axis=-2,
        )
        x = nn.Dropout(rate=config.transformer_layer.dropout_rate, name="embed_dropout")(x, dropout_eval)
        assert x.shape[-2] == 1 + 2 * (2 + config.max_len)  # 1805
        return x

    def make_causal_pad_mask(self, input_grid_shape: chex.Array, output_grid_shape: chex.Array) -> chex.Array:
        """Make a mask that is:
        - input/input: True in the input grid shapes
        - input/output: False (to respect causality)
        - output/input: True in the input and output grid shapes
        - output/output: True in the output grid shapes and causal

        Args:
            input_grid_shape: shape of the input grid. Shape (*B, 2). Number of rows and columns of the
                input grid. Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            output_grid_shape: shape of the output grid. Shape (*B, 2). Number of rows and columns of the
                output grid. Expects grid shapes values to be in [1, max_rows] and [1, max_cols].

        Returns:
            causal pad mask of shape (*B, 1, T, T) with T = 1 + 2 * (2 + max_rows * max_cols) = 1805.
        """
        batch_ndims = len(input_grid_shape.shape[:-1])
        row_arange_broadcast = jnp.arange(self.config.max_rows).reshape(
            (*batch_ndims * (1,), self.config.max_rows)
        )
        col_arange_broadcast = jnp.arange(self.config.max_cols).reshape(
            (*batch_ndims * (1,), self.config.max_cols)
        )

        # Input pad mask
        input_row_mask = row_arange_broadcast < input_grid_shape[..., :1]
        input_col_mask = col_arange_broadcast < input_grid_shape[..., 1:]
        input_pad_mask = input_row_mask[..., None] & input_col_mask[..., None, :]
        # Flatten the rows and columns.
        input_pad_mask = jnp.reshape(input_pad_mask, (*input_pad_mask.shape[:-2], 1, -1))
        # Add the masks corresponding to the input grid shapes tokens.
        input_pad_mask = jnp.concat(
            [jnp.ones((*input_pad_mask.shape[:-1], 2), bool), input_pad_mask], axis=-1
        )
        # Outer product to make the self-attention mask.
        input_input_pad_mask = input_pad_mask[..., None] & input_pad_mask[..., None, :]

        # Output pad mask
        output_row_mask = row_arange_broadcast < output_grid_shape[..., :1]
        output_col_mask = col_arange_broadcast < output_grid_shape[..., 1:]
        output_pad_mask = output_row_mask[..., None] & output_col_mask[..., None, :]
        # Flatten the rows and columns.
        output_pad_mask = jnp.reshape(output_pad_mask, (*output_pad_mask.shape[:-2], 1, -1))
        # Add the masks corresponding to the output grid shapes tokens and the context.
        output_pad_mask = jnp.concat(
            [jnp.ones((*output_pad_mask.shape[:-1], 1 + 2), bool), output_pad_mask], axis=-1
        )
        # Outer product to make the self-attention mask.
        output_output_pad_mask = output_pad_mask[..., None] & output_pad_mask[..., None, :]

        # Output causal mask
        output_output_causal_mask = jnp.tril(
            jnp.ones((*batch_ndims * (1,), 1, *output_output_pad_mask.shape[-2:]), bool)
        )

        # Putting all masks together
        input_input_mask = input_input_pad_mask
        output_output_mask = output_output_pad_mask & output_output_causal_mask
        input_output_mask = jnp.zeros_like(output_output_mask)[..., :-1, :]
        # make the input see the first token of the output (i.e. the context)
        input_output_mask = input_output_mask.at[..., 0].set(input_pad_mask)
        output_input_mask = output_pad_mask[..., None] & input_pad_mask[..., None, :]
        causal_pad_mask = jnp.concat(
            [
                jnp.concat([input_input_mask, input_output_mask], axis=-1),
                jnp.concat([output_input_mask, output_output_mask], axis=-1),
            ],
            axis=-2,
        )
        return causal_pad_mask

    def extract_logits(
        self, x: chex.Array, input_seq_length: tuple
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        config = self.config

        # Keep the second half of the sequence (the output part) and remove the last token. Apply layer norm.
        shape_row_embeds = nn.LayerNorm(
            dtype=config.dtype,
            use_bias=config.transformer_layer.use_bias,
            use_scale=False,
            name="row_logits_layer_norm",
        )(x[..., input_seq_length, :])
        shape_col_embeds = nn.LayerNorm(
            dtype=config.dtype,
            use_bias=config.transformer_layer.use_bias,
            use_scale=False,
            name="col_logits_layer_norm",
        )(x[..., input_seq_length + 1, :])
        grid_embeds = nn.LayerNorm(
            dtype=config.dtype,
            use_bias=config.transformer_layer.use_bias,
            use_scale=False,
            name="grid_logits_layer_norm",
        )(x[..., input_seq_length + 2 : -1, :])

        # Last projection to the different logits vocab sizes.
        shape_row_logits = nn.Dense(
            config.max_rows, config.logits_projection_bias, config.dtype, name="shape_row_logits_proj"
        )(shape_row_embeds)
        shape_col_logits = nn.Dense(
            config.max_cols, config.logits_projection_bias, config.dtype, name="shape_col_logits_proj"
        )(shape_col_embeds)
        grid_logits = nn.Dense(
            config.output_vocab_size, config.logits_projection_bias, config.dtype, name="grid_logits_proj"
        )(grid_embeds)
        return shape_row_logits, shape_col_logits, grid_logits


if __name__ == "__main__":
    import jax

    batch_size = 4
    mini_batch_size = 3
    max_rows = 30
    max_cols = 30
    vocab_size = 10

    # Transformer Encoder.
    encoder_config = EncoderTransformerConfig(
        vocab_size=vocab_size, max_rows=max_rows, max_cols=max_cols, variational=True
    )
    encoder = EncoderTransformer(encoder_config)

    pairs = jax.random.randint(
        jax.random.PRNGKey(0),
        (batch_size, mini_batch_size, max_rows, max_cols, 2),
        minval=0,
        maxval=vocab_size,
    )
    grid_shapes = jnp.full((batch_size, mini_batch_size, 2, 2), 15, jnp.int32)
    variables = encoder.init(jax.random.PRNGKey(0), pairs, grid_shapes, dropout_eval=False)
    num_parameters = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Encoder -> number of parameters: {num_parameters:,}")
    apply_fn = jax.jit(encoder.apply, static_argnames="dropout_eval")
    rngs = {"dropout": jax.random.PRNGKey(0)}
    print("Input shape:", pairs.shape, grid_shapes.shape)
    latent_mu, latent_logvar = apply_fn(variables, pairs, grid_shapes, dropout_eval=False, rngs=rngs)
    assert latent_mu.shape == (batch_size, mini_batch_size, encoder_config.latent_dim)
    if latent_logvar is not None:
        print("Output shape (latent_mu):", latent_mu.shape)
        print("Output shape (latent_logvar):", latent_logvar.shape)
        assert latent_logvar.shape == (batch_size, mini_batch_size, encoder_config.latent_dim)
    else:
        print("Output shape:", latent_mu.shape)

    # Transformer Decoder.
    decoder_config = DecoderTransformerConfig(
        vocab_size=vocab_size, output_vocab_size=vocab_size, max_rows=max_rows, max_cols=max_cols
    )
    decoder = DecoderTransformer(decoder_config)

    inputs = jax.random.randint(
        jax.random.PRNGKey(0),
        (batch_size, max_rows, max_cols),
        minval=0,
        maxval=vocab_size,
    )
    inputs_grid_shapes = jnp.full((batch_size, 2), 15, jnp.int32)
    flattened_input = jnp.reshape(inputs, (*inputs.shape[:-2], -1))
    input_seq = jnp.concatenate([inputs_grid_shapes, flattened_input], axis=-1)
    output_seq = jnp.zeros_like(input_seq).at[..., :2].set(1)  # Initialize the grid shape tokens to 1.
    context = jax.random.normal(jax.random.PRNGKey(0), (batch_size, encoder_config.latent_dim))
    variables = decoder.init(jax.random.PRNGKey(0), input_seq, output_seq, context, dropout_eval=False)
    num_parameters = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Decoder -> number of parameters: {num_parameters:,}")
    apply_fn = jax.jit(decoder.apply, static_argnames="dropout_eval")
    rngs = {"dropout": jax.random.PRNGKey(0)}
    print("Input shape:", inputs.shape)
    row_logits, col_logits, logits = apply_fn(
        variables, input_seq, output_seq, context, dropout_eval=False, rngs=rngs
    )
    print("Output shape:", row_logits.shape, col_logits.shape, logits.shape)
    assert row_logits.shape == (batch_size, max_rows)
    assert col_logits.shape == (batch_size, max_cols)
    assert logits.shape == (batch_size, max_rows * max_cols, vocab_size)
