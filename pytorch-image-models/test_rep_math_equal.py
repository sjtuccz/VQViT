'''
The script prove that the output results before and after token-wise rep are mathematically equivalent 
    
Token-wise operation refers to an operation that is applied independently to each token in a sequence, 
w/o any interaction or information exchange between tokens.
    Key Characteristics:
        Independence: Each token is processed in isolation.
        No Information Mixing: The operation does not allow tokens to "see" or "communicate" with each other.
    Example of token-wise operation:
        Layer Normalization (LayerNorm): Computes the mean and standard deviation for each token individually and normalizes that token's features.
        Activation Function: Applies a non-linear activation function (e.g., GELU) to each token independently.
        Linear Projection (nn.Linear): Transforms the feature vector of each token using the same set of weights, but does not combine features from different tokens.

In architectures like Transformers, token-wise operations are used to process features (e.g., in the MLP block), 
while non-token-wise operations (like attention) are used to mix information between tokens.
'''

import torch
import torch.nn as nn

from timm.models.tq_block import choose_tq

def  mathematically_equivalent():
    N,L,C=4,3,5 # input token shape
    data = torch.randn(N,L,C)
    print(f"input shape: {data.shape}")
    print(f"input range: [{data.min():.3f}, {data.max():.3f}]")

    test_layer = nn.Sequential(
        nn.LayerNorm(C),
        nn.Linear(C,4*C),
        nn.GELU(),
        nn.Linear(4*C,C),
    ) #   define token-wise operation layer
    tq = choose_tq(tq_type="TQ", dic_n=0, dim=C, dic_dim=5, tq_level=[3,3,3,3,3], tq_Tinit=1.0)

    #   before token-wise rep
    x = tq(data)
    result = test_layer(x)

    #  token-wise rep
    fixed_codebook = tq.reparameterize()
    fixed_codebook = test_layer(fixed_codebook)

    #   after token-wise rep
    embedding_index =  tq(data)
    result_rep = fixed_codebook[embedding_index]

    #       compare
    is_same = torch.allclose(result, result_rep, atol=1e-4)
    print("output is consistent before and after token-wise reparameterization ? " ,is_same)



mathematically_equivalent()