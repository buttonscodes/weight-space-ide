# (a + b) mod 7 as two attention heads and a 49-neuron lookup table, written by hand.
# Residual layout (d_model 32):  0-8 token one-hot | 9-12 position one-hot | 13-19 a | 20-26 b
# At the `=` position dims 0-6 are free, so the MLP writes the answer there and W_U reads it.
p = 7
for t in range(n_vocab):
    embed.W_E[t, t] = 1
for i in range(n_ctx):
    pos_embed.W_pos[i, 9 + i] = 1
for h, src, slot in [(0, 0, 13), (1, 2, 20)]:
    blocks[0].attn.W_Q[h, 9 + 3, 0] = 10         # the `=` position asks ...
    blocks[0].attn.W_K[h, 9 + src, 0] = 10       # ... for position 0 (head 0) or position 2 (head 1)
    for d in range(p):
        blocks[0].attn.W_V[h, d, d] = 1          # carry the digit
        blocks[0].attn.W_O[h, d, slot + d] = 1   # into the a slot or the b slot
for a in range(p):
    for b in range(p):
        n = a * p + b                            # neuron n fires only when it sees this a AND this b
        blocks[0].mlp.W_in[13 + a, n] = 1
        blocks[0].mlp.W_in[20 + b, n] = 1
        blocks[0].mlp.b_in[n] = -1
        blocks[0].mlp.W_out[n, (a + b) % p] = 1  # and writes the answer
for d in range(p):
    unembed.W_U[d, d] = 10
