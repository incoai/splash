
### decode

| shape | type | tensors in file | rows | bits/elem GGUF (streamed) | theoretical | measured | efficiency | best kernel |
|---|---|---|---|---|---|---|---|---|
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 8 | 4.5 (4.5) | 1.00 | 0.87 | 0.87 | sga_q4k_m8_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 16 | 4.5 (4.5) | 1.00 | 1.09 | 1.09 | sga_q4k_m16_c32_sg4_k32_b1_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 24 | 4.5 (4.5) | 1.00 | 1.05 | 1.05 | sga_q4k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 32 | 4.5 (4.5) | 1.00 | 1.12 | 1.12 | sga_q4k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 8 | 4.25 (4.25) | 1.06 | 0.82 | 0.78 | sga_iq4xsT_m8_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 16 | 4.25 (4.25) | 1.06 | 1.08 | 1.02 | sga_iq4xsT_m16_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 24 | 4.25 (4.25) | 1.06 | 1.12 | 1.05 | sga_iq4xsT_m32_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 32 | 4.25 (4.25) | 1.06 | 1.18 | 1.11 | sga_iq4xsT_m32_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 8 | 4.5 (4.5) | 1.00 | 0.76 | 0.76 | sga_iq4nl_m8_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 16 | 4.5 (4.5) | 1.00 | 0.97 | 0.97 | sga_iq4nl_m16_c32_sg2_k32_b2_p2 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 24 | 4.5 (4.5) | 1.00 | 1.04 | 1.04 | sga_iq4nlB_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 32 | 4.5 (4.5) | 1.00 | 1.09 | 1.09 | sga_iq4nlB_m32_c32_sg4_k32_b2_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 8 | 5.5 (5.5) | 0.82 | 0.78 | 0.95 | sga_q5k_m8_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 16 | 5.5 (5.5) | 0.82 | 0.99 | 1.21 | sga_q5k_m16_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 24 | 5.5 (5.5) | 0.82 | 0.91 | 1.11 | sga_q5k_m32_c32_sg2_k32_b2_p2 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 32 | 5.5 (5.5) | 0.82 | 0.94 | 1.14 | sga_q5k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 8 | 6.562 (6.625) | 0.69 | 0.68 | 0.99 | sga_q6k_m8_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 16 | 6.562 (6.625) | 0.69 | 0.84 | 1.22 | sga_q6k_m16_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 24 | 6.562 (6.625) | 0.69 | 0.81 | 1.19 | sga_q6k_m32_c32_sg2_k32_b2_p2 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 32 | 6.562 (6.625) | 0.69 | 0.84 | 1.22 | sga_q6k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 8 | 3.438 (3.5) | 1.31 | 0.87 | 0.66 | sga_q3k_m8_c32_sg4_k32_b2_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 16 | 3.438 (3.5) | 1.31 | 1.12 | 0.85 | sga_q3k_m16_c32_sg4_k32_b2_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 24 | 3.438 (3.5) | 1.31 | 1.12 | 0.86 | sga_q3k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 32 | 3.438 (3.5) | 1.31 | 1.21 | 0.92 | sga_q3k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 8 | 3.438 (4.062) | 1.31 | 0.84 | 0.64 | sga_iq3s_m8_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 16 | 3.438 (4.062) | 1.31 | 1.09 | 0.84 | sga_iq3s_m16_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 24 | 3.438 (4.062) | 1.31 | 1.10 | 0.84 | sga_iq3s_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 32 | 3.438 (4.062) | 1.31 | 1.16 | 0.89 | sga_iq3s_m32_c32_sg2_k32_b2_p1 |
| down 5120x17408 | q4k | ffn_down x10 | 8 | 4.5 (4.5) | 1.00 | 1.10 | 1.10 | sgka_q4k_m8_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | q4k | ffn_down x10 | 16 | 4.5 (4.5) | 1.00 | 1.30 | 1.30 | sgka_q4k_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q4k | ffn_down x10 | 24 | 4.5 (4.5) | 1.00 | 1.60 | 1.60 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q4k | ffn_down x10 | 32 | 4.5 (4.5) | 1.00 | 1.56 | 1.56 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq4xs | ffn_down x24 | 8 | 4.25 (4.25) | 1.06 | 1.00 | 0.94 | sgka_iq4xsT_m8_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | iq4xs | ffn_down x24 | 16 | 4.25 (4.25) | 1.06 | 1.26 | 1.19 | sgka_iq4xsT_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq4xs | ffn_down x24 | 24 | 4.25 (4.25) | 1.06 | 1.65 | 1.56 | sgka_iq4xsT_m32_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | iq4xs | ffn_down x24 | 32 | 4.25 (4.25) | 1.06 | 1.63 | 1.54 | sgka_iq4xsT_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq4nl | ffn_down x3 | 8 | 4.5 (4.5) | 1.00 | 0.94 | 0.94 | sgka_iq4nl_m8_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq4nl | ffn_down x3 | 16 | 4.5 (4.5) | 1.00 | 1.18 | 1.18 | sgka_iq4nl_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq4nl | ffn_down x3 | 24 | 4.5 (4.5) | 1.00 | 1.53 | 1.53 | sgka_iq4nlB_m32_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | iq4nl | ffn_down x3 | 32 | 4.5 (4.5) | 1.00 | 1.52 | 1.52 | sgka_iq4nlB_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q5k | ffn_down x22 | 8 | 5.5 (5.5) | 0.82 | 0.93 | 1.13 | sgka_q5k_m8_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q5k | ffn_down x22 | 16 | 5.5 (5.5) | 0.82 | 1.11 | 1.36 | sgka_q5k_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q5k | ffn_down x22 | 24 | 5.5 (5.5) | 0.82 | 1.46 | 1.78 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q5k | ffn_down x22 | 32 | 5.5 (5.5) | 0.82 | 1.40 | 1.72 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q6k | ffn_down x1 | 8 | 6.562 (6.625) | 0.69 | 0.83 | 1.21 | sgka_q6k_m8_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q6k | ffn_down x1 | 16 | 6.562 (6.625) | 0.69 | 0.93 | 1.36 | sgka_q6k_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q6k | ffn_down x1 | 24 | 6.562 (6.625) | 0.69 | 1.28 | 1.86 | sgka_q6k_m32_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | q6k | ffn_down x1 | 32 | 6.562 (6.625) | 0.69 | 1.26 | 1.84 | sgka_q6k_m32_c32_sg4_k32_b2_p1 +splitK |
| down 5120x17408 | q3k | ffn_down x1 | 8 | 3.438 (3.5) | 1.31 | 1.12 | 0.85 | sgka_q3k_m8_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q3k | ffn_down x1 | 16 | 3.438 (3.5) | 1.31 | 1.37 | 1.05 | sgka_q3k_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q3k | ffn_down x1 | 24 | 3.438 (3.5) | 1.31 | 1.69 | 1.29 | sgka_q3k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | q3k | ffn_down x1 | 32 | 3.438 (3.5) | 1.31 | 1.68 | 1.28 | sgka_q3k_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq3s | ffn_down x3 | 8 | 3.438 (4.062) | 1.31 | 1.02 | 0.78 | sgka_iq3s_m8_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq3s | ffn_down x3 | 16 | 3.438 (4.062) | 1.31 | 1.33 | 1.02 | sgka_iq3s_m16_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq3s | ffn_down x3 | 24 | 3.438 (4.062) | 1.31 | 1.63 | 1.25 | sgka_iq3s_m32_c32_sg2_k32_b2_p1 +splitK |
| down 5120x17408 | iq3s | ffn_down x3 | 32 | 3.438 (4.062) | 1.31 | 1.64 | 1.25 | sgka_iq3s_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 8 | 4.5 (4.5) | 1.00 | 0.96 | 0.96 | sgka_q4k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 16 | 4.5 (4.5) | 1.00 | 1.06 | 1.06 | sgka_q4k_m16_c32_sg4_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 24 | 4.5 (4.5) | 1.00 | 1.06 | 1.06 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 32 | 4.5 (4.5) | 1.00 | 1.38 | 1.38 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 8 | 4.25 (4.25) | 1.06 | 0.87 | 0.82 | sgka_iq4xsT_m8_c32_sg4_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 16 | 4.25 (4.25) | 1.06 | 0.97 | 0.92 | sgka_iq4xsT_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 24 | 4.25 (4.25) | 1.06 | 1.00 | 0.94 | sgka_iq4xsT_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 32 | 4.25 (4.25) | 1.06 | 1.30 | 1.23 | sgka_iq4xsB_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 8 | 4.5 (4.5) | 1.00 | 0.83 | 0.83 | sgka_iq4nl_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 16 | 4.5 (4.5) | 1.00 | 0.94 | 0.94 | sgka_iq4nlB_m16_c32_sg4_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 24 | 4.5 (4.5) | 1.00 | 0.97 | 0.97 | sga_iq4nlB_m32_c32_sg2_k32_b2_p1 |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 32 | 4.5 (4.5) | 1.00 | 1.26 | 1.26 | sgka_iq4nlB_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 8 | 5.5 (5.5) | 0.82 | 0.86 | 1.05 | sgka_q5k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 16 | 5.5 (5.5) | 0.82 | 0.93 | 1.13 | sga_q5k_m16_c32_sg2_k32_b2_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 24 | 5.5 (5.5) | 0.82 | 0.94 | 1.15 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 32 | 5.5 (5.5) | 0.82 | 1.18 | 1.45 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 8 | 4.5 (4.5) | 1.00 | 0.93 | 0.93 | sgka_q4k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 16 | 4.5 (4.5) | 1.00 | 1.12 | 1.12 | sgka_q4k_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 24 | 4.5 (4.5) | 1.00 | 1.28 | 1.28 | sgka_q4k_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 32 | 4.5 (4.5) | 1.00 | 1.35 | 1.35 | sgka_q4k_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 8 | 4.25 (4.25) | 1.06 | 0.78 | 0.74 | sgka_iq4xsT_m8_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 16 | 4.25 (4.25) | 1.06 | 0.91 | 0.86 | sgka_iq4xsB_m16_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 24 | 4.25 (4.25) | 1.06 | 1.20 | 1.13 | sgka_iq4xsB_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 32 | 4.25 (4.25) | 1.06 | 1.27 | 1.20 | sgka_iq4xsB_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 8 | 5.5 (5.5) | 0.82 | 0.82 | 1.01 | sgka_q5k_m8_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 16 | 5.5 (5.5) | 0.82 | 0.95 | 1.17 | sgka_q5k_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 24 | 5.5 (5.5) | 0.82 | 1.15 | 1.40 | sgka_q5k_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 32 | 5.5 (5.5) | 0.82 | 1.24 | 1.52 | sgka_q5k_m32_c32_sg4_k32_b2_p1 +splitK |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 8 | 8.5 (8.5) | 0.53 | 3.64 | 6.87 | sgka_q80_m8_c16_sg8_k64_b1_p1 +splitK |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 16 | 8.5 (8.5) | 0.53 | 4.00 | 7.56 | sgka_q80_m16_c16_sg8_k64_b1_p1 +splitK |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 24 | 8.5 (8.5) | 0.53 | 5.64 | 10.66 | sgka_q80_m32_c16_sg8_k64_b1_p1 +splitK |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 32 | 8.5 (8.5) | 0.53 | 5.21 | 9.85 | sgka_q80_m32_c16_sg8_k64_b1_p1 +splitK |
| out 5120x6144 | q4k | ssm_out x5 | 8 | 4.5 (4.5) | 1.00 | 1.09 | 1.09 | sgka_q4k_m8_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | q4k | ssm_out x5 | 16 | 4.5 (4.5) | 1.00 | 1.36 | 1.36 | sgka_q4k_m16_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q4k | ssm_out x5 | 24 | 4.5 (4.5) | 1.00 | 1.51 | 1.51 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q4k | ssm_out x5 | 32 | 4.5 (4.5) | 1.00 | 1.68 | 1.68 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 8 | 4.25 (4.25) | 1.06 | 0.93 | 0.87 | sgka_iq4xsT_m8_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 16 | 4.25 (4.25) | 1.06 | 1.11 | 1.05 | sgka_iq4xsT_m16_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 24 | 4.25 (4.25) | 1.06 | 1.43 | 1.35 | sgka_iq4xsT_m32_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 32 | 4.25 (4.25) | 1.06 | 1.55 | 1.47 | sgka_iq4xsB_m32_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 8 | 5.5 (5.5) | 0.82 | 0.97 | 1.18 | sgka_q5k_m8_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 16 | 5.5 (5.5) | 0.82 | 1.16 | 1.42 | sgka_q5k_m16_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 24 | 5.5 (5.5) | 0.82 | 1.41 | 1.72 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 32 | 5.5 (5.5) | 0.82 | 1.53 | 1.87 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 8 | 6.562 (6.625) | 0.69 | 0.88 | 1.28 | sgka_q6k_m8_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 16 | 6.562 (6.625) | 0.69 | 0.93 | 1.35 | sgka_q6k_m16_c32_sg2_k32_b2_p1 +splitK |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 24 | 6.562 (6.625) | 0.69 | 1.26 | 1.83 | sgka_q6k_m32_c32_sg4_k32_b2_p1 +splitK |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 32 | 6.562 (6.625) | 0.69 | 1.38 | 2.01 | sgka_q6k_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_q 12288x5120 | q4k | attn_q x6 | 8 | 4.5 (4.5) | 1.00 | 0.93 | 0.93 | sgka_q4k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_q 12288x5120 | q4k | attn_q x6 | 16 | 4.5 (4.5) | 1.00 | 1.14 | 1.14 | sgka_q4k_m16_c32_sg4_k32_b2_p1 +splitK |
| attn_q 12288x5120 | q4k | attn_q x6 | 24 | 4.5 (4.5) | 1.00 | 1.28 | 1.28 | sga_q4k_m32_c32_sg4_k32_b1_p1 |
| attn_q 12288x5120 | q4k | attn_q x6 | 32 | 4.5 (4.5) | 1.00 | 1.15 | 1.15 | sgka_q4k_m32_c32_sg4_k32_b2_p1 +splitK |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 8 | 4.25 (4.25) | 1.06 | 0.83 | 0.79 | sgka_iq4xsT_m8_c32_sg4_k32_b2_p1 +splitK |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 16 | 4.25 (4.25) | 1.06 | 1.14 | 1.07 | sga_iq4xsT_m16_c64_sg1_k32_b2_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 24 | 4.25 (4.25) | 1.06 | 1.28 | 1.21 | sga_iq4xsT_m32_c64_sg1_k32_b2_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 32 | 4.25 (4.25) | 1.06 | 1.13 | 1.06 | sga_iq4xsT_m32_c64_sg1_k32_b2_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 8 | 5.5 (5.5) | 0.82 | 0.79 | 0.97 | sgka_q5k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_q 12288x5120 | q5k | attn_q x2 | 16 | 5.5 (5.5) | 0.82 | 0.96 | 1.17 | sgka_q5k_m16_c32_sg4_k32_b2_p1 +splitK |
| attn_q 12288x5120 | q5k | attn_q x2 | 24 | 5.5 (5.5) | 0.82 | 1.12 | 1.37 | sga_q5k_m32_c32_sg1_k32_b2_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 32 | 5.5 (5.5) | 0.82 | 1.00 | 1.22 | sga_q5k_m32_c32_sg2_k32_b2_p1 |
| attn_kv 1024x5120 | q4k | attn_k x1 | 8 | 4.5 (4.5) | 1.00 | 2.41 | 2.41 | sgka_q4k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q4k | attn_k x1 | 16 | 4.5 (4.5) | 1.00 | 3.12 | 3.12 | sgka_q4k_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q4k | attn_k x1 | 24 | 4.5 (4.5) | 1.00 | 3.86 | 3.86 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q4k | attn_k x1 | 32 | 4.5 (4.5) | 1.00 | 3.62 | 3.62 | sgka_q4k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 8 | 5.5 (5.5) | 0.82 | 2.16 | 2.64 | sgka_q5k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 16 | 5.5 (5.5) | 0.82 | 2.78 | 3.40 | sgka_q5k_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 24 | 5.5 (5.5) | 0.82 | 3.52 | 4.30 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 32 | 5.5 (5.5) | 0.82 | 3.30 | 4.04 | sgka_q5k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 8 | 6.562 (6.625) | 0.69 | 2.28 | 3.32 | sgka_q6k_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 16 | 6.562 (6.625) | 0.69 | 3.12 | 4.56 | sgka_q6k_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 24 | 6.562 (6.625) | 0.69 | 3.68 | 5.37 | sgka_q6k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 32 | 6.562 (6.625) | 0.69 | 3.45 | 5.04 | sgka_q6k_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 8 | 8.5 (8.5) | 0.53 | 2.16 | 4.08 | sgka_q80_m8_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 16 | 8.5 (8.5) | 0.53 | 2.78 | 5.25 | sgka_q80_m16_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 24 | 8.5 (8.5) | 0.53 | 3.52 | 6.65 | sgka_q80_m32_c32_sg2_k32_b2_p1 +splitK |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 32 | 8.5 (8.5) | 0.53 | 3.30 | 6.24 | sgka_q80_m32_c32_sg2_k32_b2_p1 +splitK |
| lm_head 248320x5120 | q6k | output.weight x1 | 8 | 6.562 (6.625) | 0.69 | 0.68 | 0.99 | sga_q6k_m8_c32_sg4_k32_b1_p1 |
| lm_head 248320x5120 | q6k | output.weight x1 | 16 | 6.562 (6.625) | 0.69 | 0.83 | 1.22 | sga_q6k_m16_c32_sg2_k32_b2_p1 |
| lm_head 248320x5120 | q6k | output.weight x1 | 24 | 6.562 (6.625) | 0.69 | 0.92 | 1.35 | sga_q6k_m32_c32_sg4_k32_b2_p1 |
| lm_head 248320x5120 | q6k | output.weight x1 | 32 | 6.562 (6.625) | 0.69 | 0.97 | 1.42 | sga_q6k_m32_c32_sg1_k32_b2_p1 |

### prefill

| shape | type | tensors in file | rows | bits/elem GGUF (streamed) | theoretical | measured | best kernel |
|---|---|---|---|---|---|---|---|
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 17 | 4.5 (4.5) | 1.00 | 1.28 | sga_q4k_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 128 | 4.5 (4.5) | 1.00 | 1.07 | pfa_q4k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 512 | 4.5 (4.5) | 1.00 | 1.02 | pfa_q4k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q4k | ffn_gate x14, ffn_up x14 | 2048 | 4.5 (4.5) | 1.00 | 1.03 | pfa_q4k_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 17 | 4.25 (4.25) | 1.00 | 1.26 | sga_iq4xsT_m32_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 128 | 4.25 (4.25) | 1.00 | 1.08 | pfa_iq4xsT_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 512 | 4.25 (4.25) | 1.00 | 1.03 | pfa_iq4xsT_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | iq4xs | ffn_gate x31, ffn_up x30 | 2048 | 4.25 (4.25) | 1.00 | 1.05 | pfa_iq4xsB_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 17 | 4.5 (4.5) | 1.00 | 1.19 | sga_iq4nlB_m32_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 128 | 4.5 (4.5) | 1.00 | 1.08 | pfa_iq4nl_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 512 | 4.5 (4.5) | 1.00 | 1.05 | pfa_iq4nl_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq4nl | ffn_gate x2, ffn_up x1 | 2048 | 4.5 (4.5) | 1.00 | 1.05 | pfa_iq4nl_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 17 | 5.5 (5.5) | 1.00 | 1.01 | sga_q5k_m32_c32_sg2_k32_b2_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 128 | 5.5 (5.5) | 1.00 | 1.07 | pfa_q5k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 512 | 5.5 (5.5) | 1.00 | 1.00 | pfa_q5k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q5k | ffn_gate x12, ffn_up x16 | 2048 | 5.5 (5.5) | 1.00 | 0.97 | pfa_q5k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 17 | 6.562 (6.625) | 1.00 | 0.92 | sga_q6k_m32_c32_sg4_k32_b1_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 128 | 6.562 (6.625) | 1.00 | 1.09 | pfa_q6k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 512 | 6.562 (6.625) | 1.00 | 1.00 | pfa_q6k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q6k | ffn_up x1 | 2048 | 6.562 (6.625) | 1.00 | 0.97 | pfa_q6k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 17 | 3.438 (3.5) | 1.00 | 1.31 | sga_q3k_m32_c32_sg2_k32_b2_p2 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 128 | 3.438 (3.5) | 1.00 | 1.06 | pfa_q3k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 512 | 3.438 (3.5) | 1.00 | 0.98 | pfa_q3k_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | q3k | ffn_gate x4, ffn_up x2 | 2048 | 3.438 (3.5) | 1.00 | 0.97 | pfa_q3k_r32_sg4_n64_k32_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 17 | 3.438 (4.062) | 1.00 | 1.27 | sga_iq3s_m32_c32_sg1_k32_b2_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 128 | 3.438 (4.062) | 1.00 | 1.06 | pfa_iq3s_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 512 | 3.438 (4.062) | 1.00 | 0.99 | pfa_iq3s_r32_sg4_n64_k64_p1 |
| gate/up 17408x5120 | iq3s | ffn_gate x1 | 2048 | 3.438 (4.062) | 1.00 | 0.95 | pfa_iq3s_r32_sg4_n64_k32_p1 |
| down 5120x17408 | q4k | ffn_down x10 | 17 | 4.5 (4.5) | 1.00 | 1.51 | sga_q4k_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | q4k | ffn_down x10 | 128 | 4.5 (4.5) | 1.00 | 1.38 | pfa_q4k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q4k | ffn_down x10 | 512 | 4.5 (4.5) | 1.00 | 1.12 | pfa_q4k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q4k | ffn_down x10 | 2048 | 4.5 (4.5) | 1.00 | 1.00 | pfa_q4k_r32_sg8_n64_k64_p1 |
| down 5120x17408 | iq4xs | ffn_down x24 | 17 | 4.25 (4.25) | 1.00 | 1.65 | sga_iq4xsT_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | iq4xs | ffn_down x24 | 128 | 4.25 (4.25) | 1.00 | 1.42 | pfa_iq4xs_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq4xs | ffn_down x24 | 512 | 4.25 (4.25) | 1.00 | 1.13 | pfa_iq4xs_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq4xs | ffn_down x24 | 2048 | 4.25 (4.25) | 1.00 | 1.00 | pfa_iq4xs_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq4nl | ffn_down x3 | 17 | 4.5 (4.5) | 1.00 | 1.42 | sga_iq4nl_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | iq4nl | ffn_down x3 | 128 | 4.5 (4.5) | 1.00 | 1.40 | pfa_iq4nlB_r32_sg4_n64_k32_p1 |
| down 5120x17408 | iq4nl | ffn_down x3 | 512 | 4.5 (4.5) | 1.00 | 1.07 | pfa_iq4nlB_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq4nl | ffn_down x3 | 2048 | 4.5 (4.5) | 1.00 | 0.99 | pfa_iq4nl_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q5k | ffn_down x22 | 17 | 5.5 (5.5) | 1.00 | 1.38 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | q5k | ffn_down x22 | 128 | 5.5 (5.5) | 1.00 | 1.40 | pfa_q5k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q5k | ffn_down x22 | 512 | 5.5 (5.5) | 1.00 | 1.06 | pfa_q5k_r32_sg8_n64_k64_p1 |
| down 5120x17408 | q5k | ffn_down x22 | 2048 | 5.5 (5.5) | 1.00 | 1.00 | pfa_q5k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q6k | ffn_down x1 | 17 | 6.562 (6.625) | 1.00 | 1.46 | sga_q6k_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | q6k | ffn_down x1 | 128 | 6.562 (6.625) | 1.00 | 1.38 | pfa_q6k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q6k | ffn_down x1 | 512 | 6.562 (6.625) | 1.00 | 1.09 | pfa_q6k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q6k | ffn_down x1 | 2048 | 6.562 (6.625) | 1.00 | 0.96 | pfa_q6k_r16_sg8_n64_k64_p1 |
| down 5120x17408 | q3k | ffn_down x1 | 17 | 3.438 (3.5) | 1.00 | 1.45 | sga_q3k_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | q3k | ffn_down x1 | 128 | 3.438 (3.5) | 1.00 | 1.34 | pfa_q3k_r32_sg4_n64_k32_p1 |
| down 5120x17408 | q3k | ffn_down x1 | 512 | 3.438 (3.5) | 1.00 | 1.11 | pfa_q3k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | q3k | ffn_down x1 | 2048 | 3.438 (3.5) | 1.00 | 0.99 | pfa_q3k_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq3s | ffn_down x3 | 17 | 3.438 (4.062) | 1.00 | 1.46 | sga_iq3s_m32_c32_sg4_k32_b1_p1 |
| down 5120x17408 | iq3s | ffn_down x3 | 128 | 3.438 (4.062) | 1.00 | 1.39 | pfa_iq3s_r32_sg4_n64_k32_p1 |
| down 5120x17408 | iq3s | ffn_down x3 | 512 | 3.438 (4.062) | 1.00 | 1.09 | pfa_iq3s_r32_sg4_n64_k64_p1 |
| down 5120x17408 | iq3s | ffn_down x3 | 2048 | 3.438 (4.062) | 1.00 | 0.99 | pfa_iq3s_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 17 | 4.5 (4.5) | 1.00 | 1.35 | sga_q4k_m32_c32_sg4_k32_b1_p1 |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 128 | 4.5 (4.5) | 1.00 | 1.18 | pfa_q4k_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 512 | 4.5 (4.5) | 1.00 | 1.06 | pfa_q4k_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | q4k | attn_qkv x29 | 2048 | 4.5 (4.5) | 1.00 | 1.05 | pfa_q4k_r32_sg4_n64_k32_p1 |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 17 | 4.25 (4.25) | 1.00 | 1.28 | sga_iq4xsT_m32_c32_sg2_k32_b2_p1 |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 128 | 4.25 (4.25) | 1.00 | 1.18 | pfa_iq4xsT_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 512 | 4.25 (4.25) | 1.00 | 1.07 | pfa_iq4xsB_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | iq4xs | attn_qkv x6 | 2048 | 4.25 (4.25) | 1.00 | 1.08 | pfa_iq4xsT_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 17 | 4.5 (4.5) | 1.00 | 1.26 | sga_iq4nlB_m32_c32_sg4_k32_b2_p1 |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 128 | 4.5 (4.5) | 1.00 | 1.20 | pfa_iq4nl_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 512 | 4.5 (4.5) | 1.00 | 1.08 | pfa_iq4nl_r32_sg4_n64_k32_p1 |
| attn_qkv 10240x5120 | iq4nl | attn_qkv x1 | 2048 | 4.5 (4.5) | 1.00 | 1.08 | pfa_iq4nlB_r32_sg4_n64_k32_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 17 | 5.5 (5.5) | 1.00 | 1.20 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 128 | 5.5 (5.5) | 1.00 | 1.16 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 512 | 5.5 (5.5) | 1.00 | 1.06 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_qkv 10240x5120 | q5k | attn_qkv x12 | 2048 | 5.5 (5.5) | 1.00 | 1.06 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 17 | 4.5 (4.5) | 1.00 | 1.48 | sga_q4k_m32_c32_sg4_k32_b1_p1 |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 128 | 4.5 (4.5) | 1.00 | 1.20 | pfa_q4k_r16_sg8_n64_k64_p1 |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 512 | 4.5 (4.5) | 1.00 | 1.05 | pfa_q4k_r32_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | q4k | attn_gate x24 | 2048 | 4.5 (4.5) | 1.00 | 1.03 | pfa_q4k_r32_sg4_n64_k32_p1 |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 17 | 4.25 (4.25) | 1.00 | 1.59 | sga_iq4xsT_m32_c32_sg4_k32_b1_p1 |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 128 | 4.25 (4.25) | 1.00 | 1.18 | pfa_iq4xsT_r16_sg8_n64_k64_p1 |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 512 | 4.25 (4.25) | 1.00 | 1.07 | pfa_iq4xsT_r32_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | iq4xs | attn_gate x12 | 2048 | 4.25 (4.25) | 1.00 | 1.04 | pfa_iq4xsB_r32_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 17 | 5.5 (5.5) | 1.00 | 1.23 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 128 | 5.5 (5.5) | 1.00 | 1.14 | pfa_q5k_r16_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 512 | 5.5 (5.5) | 1.00 | 1.03 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_gate 6144x5120 | q5k | attn_gate x12 | 2048 | 5.5 (5.5) | 1.00 | 1.03 | pfa_q5k_r32_sg4_n64_k64_p1 |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 17 | 8.5 (8.5) | 1.00 | 1.56 | sga_q80_m32_c16_sg8_k64_b1_p1 |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 128 | 8.5 (8.5) | 1.00 | 1.40 | pfa_q80_r16_sg4_n64_k64_p1 |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 512 | 8.5 (8.5) | 1.00 | 1.53 | pfa_q80_r16_sg8_n64_k64_p1 |
| ab 256x5120 | q80 | ssm_alpha x48, ssm_beta x48 | 2048 | 8.5 (8.5) | 1.00 | 1.49 | pfa_q80_r16_sg4_n128_k32_p1 |
| out 5120x6144 | q4k | ssm_out x5 | 17 | 4.5 (4.5) | 1.00 | 1.35 | sga_q4k_m32_c32_sg4_k32_b1_p1 |
| out 5120x6144 | q4k | ssm_out x5 | 128 | 4.5 (4.5) | 1.00 | 1.35 | pfa_q4k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q4k | ssm_out x5 | 512 | 4.5 (4.5) | 1.00 | 1.07 | pfa_q4k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q4k | ssm_out x5 | 2048 | 4.5 (4.5) | 1.00 | 0.99 | pfa_q4k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 17 | 4.25 (4.25) | 1.00 | 1.54 | sga_iq4xsB_m32_c32_sg4_k32_b1_p1 |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 128 | 4.25 (4.25) | 1.00 | 1.35 | pfa_iq4xsT_r32_sg4_n64_k64_p1 |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 512 | 4.25 (4.25) | 1.00 | 1.07 | pfa_iq4xsB_r32_sg4_n64_k64_p1 |
| out 5120x6144 | iq4xs | attn_output x1, ssm_out x5 | 2048 | 4.25 (4.25) | 1.00 | 1.02 | pfa_iq4xsB_r32_sg4_n64_k32_p1 |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 17 | 5.5 (5.5) | 1.00 | 1.20 | sga_q5k_m32_c32_sg4_k32_b1_p1 |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 128 | 5.5 (5.5) | 1.00 | 1.34 | pfa_q5k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 512 | 5.5 (5.5) | 1.00 | 1.04 | pfa_q5k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q5k | attn_output x11, ssm_out x33 | 2048 | 5.5 (5.5) | 1.00 | 1.01 | pfa_q5k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 17 | 6.562 (6.625) | 1.00 | 1.22 | sga_q6k_m32_c32_sg4_k32_b1_p1 |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 128 | 6.562 (6.625) | 1.00 | 1.34 | pfa_q6k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 512 | 6.562 (6.625) | 1.00 | 1.04 | pfa_q6k_r32_sg4_n64_k64_p1 |
| out 5120x6144 | q6k | attn_output x4, ssm_out x5 | 2048 | 6.562 (6.625) | 1.00 | 1.00 | pfa_q6k_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | q4k | attn_q x6 | 17 | 4.5 (4.5) | 1.00 | 1.35 | sga_q4k_m32_c32_sg4_k32_b2_p1 |
| attn_q 12288x5120 | q4k | attn_q x6 | 128 | 4.5 (4.5) | 1.00 | 1.11 | pfa_q4k_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | q4k | attn_q x6 | 512 | 4.5 (4.5) | 1.00 | 1.06 | pfa_q4k_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | q4k | attn_q x6 | 2048 | 4.5 (4.5) | 1.00 | 1.03 | pfa_q4k_r32_sg4_n64_k32_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 17 | 4.25 (4.25) | 1.00 | 1.36 | sga_iq4xsT_m32_c64_sg1_k32_b2_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 128 | 4.25 (4.25) | 1.00 | 1.11 | pfa_iq4xsB_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 512 | 4.25 (4.25) | 1.00 | 1.06 | pfa_iq4xs_r32_sg4_n64_k32_p1 |
| attn_q 12288x5120 | iq4xs | attn_q x8 | 2048 | 4.25 (4.25) | 1.00 | 1.03 | pfa_iq4xs_r32_sg4_n64_k32_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 17 | 5.5 (5.5) | 1.00 | 1.22 | sga_q5k_m32_c32_sg1_k32_b2_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 128 | 5.5 (5.5) | 1.00 | 1.09 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 512 | 5.5 (5.5) | 1.00 | 1.03 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_q 12288x5120 | q5k | attn_q x2 | 2048 | 5.5 (5.5) | 1.00 | 0.97 | pfa_q5k_r32_sg4_n64_k32_p1 |
| attn_kv 1024x5120 | q4k | attn_k x1 | 17 | 4.5 (4.5) | 1.00 | 1.44 | sga_q4k_m32_c16_sg8_k64_b1_p1 |
| attn_kv 1024x5120 | q4k | attn_k x1 | 128 | 4.5 (4.5) | 1.00 | 1.27 | pfa_q4k_r16_sg8_n64_k64_p1 |
| attn_kv 1024x5120 | q4k | attn_k x1 | 512 | 4.5 (4.5) | 1.00 | 1.35 | pfa_q4k_r16_sg4_n128_k32_p1 |
| attn_kv 1024x5120 | q4k | attn_k x1 | 2048 | 4.5 (4.5) | 1.00 | 1.09 | pfa_q4k_r16_sg4_n128_k32_p1 |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 17 | 5.5 (5.5) | 1.00 | 1.23 | sga_q5k_m32_c16_sg8_k64_b1_p1 |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 128 | 5.5 (5.5) | 1.00 | 1.16 | pfa_q5k_r16_sg8_n64_k64_p1 |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 512 | 5.5 (5.5) | 1.00 | 1.30 | pfa_q5k_r16_sg4_n128_k32_p1 |
| attn_kv 1024x5120 | q5k | attn_k x5, attn_v x6 | 2048 | 5.5 (5.5) | 1.00 | 1.07 | pfa_q5k_r32_sg4_n64_k64_p1 |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 17 | 6.562 (6.625) | 1.00 | 1.30 | sga_q6k_m32_c16_sg8_k64_b1_p1 |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 128 | 6.562 (6.625) | 1.00 | 1.22 | pfa_q6k_r16_sg8_n64_k64_p1 |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 512 | 6.562 (6.625) | 1.00 | 1.36 | pfa_q6k_r16_sg4_n128_k32_p1 |
| attn_kv 1024x5120 | q6k | attn_k x9, attn_v x3 | 2048 | 6.562 (6.625) | 1.00 | 1.09 | pfa_q6k_r32_sg4_n64_k64_p1 |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 17 | 8.5 (8.5) | 1.00 | 1.30 | sga_q80_m32_c16_sg8_k64_b1_p1 |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 128 | 8.5 (8.5) | 1.00 | 1.21 | pfa_q80_r16_sg8_n64_k64_p1 |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 512 | 8.5 (8.5) | 1.00 | 1.34 | pfa_q80_r16_sg4_n128_k32_p1 |
| attn_kv 1024x5120 | q80 | attn_k x1, attn_v x7 | 2048 | 8.5 (8.5) | 1.00 | 1.09 | pfa_q80_r16_sg8_n64_k64_p1 |

missing cells: [('prefill', 'lm_head 248320x5120', 'q6k')]
