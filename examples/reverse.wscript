# A sequence reverser in one attention head, written by hand.
# Residual stream layout (d_model 24):  0-6 token one-hot | 7-16 position one-hot | 17-23 output one-hot
for t in range(n_vocab):
    embed.W_E[t, t] = 1                      # token t      -> dim t
    blocks[0].attn.W_V[0, t, t] = 1          # value: carry the token identity
    blocks[0].attn.W_O[0, t, 17 + t] = 1     # ... and write it into the output subspace
    unembed.W_U[17 + t, t] = 10              # output dim t -> logit of token t
for p in range(n_ctx):
    pos_embed.W_pos[p, 7 + p] = 1            # position p   -> dim 7 + p
    blocks[0].attn.W_K[0, 7 + p, p] = 10     # key: "I am position p"
for p in range(5, 9):
    blocks[0].attn.W_Q[0, 7 + p, 9 - p] = 10 # query at output position p: "I want position 9 - p"
