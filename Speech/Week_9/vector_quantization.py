import torch
import torch.nn as nn
import torch.nn.functional as F

## for further efficiency EMA can be implemented for dealign with dead codebook problem but for now Random Awakening is also good solution!

class Perplexity(nn.Module):
    EPS = 1e-8
    def __init__(self, n_codecs):
        super().__init__()
        self.n_codecs = n_codecs
    
    def forward(self, indices):
        device = indices.device

        arange = torch.arange(self.n_codecs, device=device)
        indices = indices.flatten()
        encodings = torch.eq(arange.unsqueeze(dim=1), indices.unsqueeze(dim=0))

        probs = torch.mean(encodings.float(), dim=1)
        perplexity = torch.exp(- torch.sum(probs * torch.log(probs + self.EPS)))
        return perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super().__init__()

        self.dead_threshold = 5 # +- right one
        self.register_buffer('dead_counter', torch.zeros(codebook_size, dtype=torch.long))

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim)

        self._init_weight()

    def _init_weight(self):
        init_size = 1 / self.codebook_size
        torch.nn.init.uniform_(self.codebook.weight, a=-init_size, b=init_size)

    def calculate_squared_distances(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        tensor_1: float tensor with shape [sequence_1, embedding] 
        tensor_2: float tensor with shape [sequence_2, embedding]
        output: float tensor with shape [sequence_1, sequence_2]
        """
        # Your code here MSE will not work here as dientions missmatch: ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
        t1_squared = torch.sum(tensor_1 ** 2, dim=1, keepdim=True)  
        t2_squared = torch.sum(tensor_2 ** 2, dim=1, keepdim=True).t() 
        matrix_mul = torch.matmul(tensor_1, tensor_2.t()) 

        distances = t1_squared + t2_squared - 2 * matrix_mul
        return distances


    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings, by the indices of closest embeddings from the codebook
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, height, width]
        """
        assert embeddings.dim() == 4
        B, E, H, W = embeddings.shape

        # [sequence_1, embedding_dim] -> (B*H*W, E)
        embeddings_flat = embeddings.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = self.calculate_squared_distances(embeddings_flat, self.codebook.weight)
        indices_flat = torch.argmin(distances, dim=1) # argmin standard

        # --- Persistent Random Awakening ---
        if self.training:
            used_indices = torch.unique(indices_flat)
        
            self.dead_counter += 1
            self.dead_counter[used_indices] = 0
            
            # 3. Identify vectors that have been dead for longer than the threshold
            dead_indices = torch.where(self.dead_counter >= self.dead_threshold)[0]
            
            if len(dead_indices) > 0:
                perm = torch.randperm(embeddings_flat.size(0), device=embeddings.device)
                num_to_replace = min(len(dead_indices), len(perm))
                
                if num_to_replace > 0:
                    # Replace dead vectors with active ones
                    dead_to_revive = dead_indices[:num_to_replace]
                    self.codebook.weight.data[dead_to_revive] = embeddings_flat[perm[:num_to_replace]]
                    
                    # Reset the counter for the newly awakened vectors
                    self.dead_counter[dead_to_revive] = 0

        indices = indices_flat.view(B, H, W)
        return indices

    def decode(self, indices: torch.Tensor):
        """
        Inserts embeddings from the codebook instead of indices
        Indices: Longtensor of indices from the codebook of size [batch, height, width]
        For each index: 0 <= index < codebook_size
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        # Your code here
        B, H, W = indices.shape

        indices_flat = indices.reshape(-1)
        quantized_flat = self.codebook(indices_flat)

        decoded = quantized_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

        return decoded

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)

        quantized = embeddings + (quantized - embeddings).detach()

        return quantized

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, n_codebooks):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        
        self.codebooks = [VectorQuantizer(codebook_size, embedding_dim) for _ in range(n_codebooks)]
        self.codebooks = nn.ModuleList(self.codebooks)

    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings by the indices of closest embeddings from the first codebook.
        Then iteratively encodes the residuals between the embedding and vectors from the codebook.
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, n_codebooks, height, width]
        """
        B, E, H, W = embeddings.shape
        codecs_list = []

        residual = embeddings.clone()
        
        for quantizer in self.codebooks:
            # 1. Get indices for this layer
            indices = quantizer.encode(residual)
            codecs_list.append(indices)

            quantized = quantizer.decode(indices)
 
            residual = residual - quantized # we need to find how much error is left after this layer so we substract

        codecs = torch.stack(codecs_list, dim=1)
        return codecs

    def decode(self, codecs: torch.Tensor):
        """
        Sums the embeddings from the codebooks with dedicated indices
        Indices: Longtensor of indices from the codebook of size [batch, n_codebooks, height, width]
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        B, n_cb, H, W = codecs.shape
        quantized_sum = 0.0
        
        for i, quantizer in enumerate(self.codebooks):
            indices_layer = codecs[:, i, :, :]
            quantized_layer = quantizer.decode(indices_layer)
            quantized_sum = quantized_sum + quantized_layer # now as as we reconstruct
            
        return quantized_sum

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)

        quantized = embeddings + (quantized - embeddings).detach()

        return quantized


class VectorQuantizationLoss(nn.Module):
    def __init__(self, commitment_cost=1.):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, inputs, quantized):
        """
        Calculates the vector quantisation loss
        inputs: vector of embeddings of size [batch, embedding, height, width]
        quantized: the vector of embeddings, processed by VectorQuantisation or ResidualVectorQuantization
        output: differentiable loss of size [1]
        """
        # L2 Codebook Loss
        loss_l2 = F.mse_loss(quantized, inputs.detach())
        
        # L3 not drift too far from codebook encoder loss
        loss_l3 = F.mse_loss(quantized.detach(), inputs)
        
        # Total optimization penalty
        loss = loss_l2 + self.commitment_cost * loss_l3 # main image reconstrcution loss will be added in the pipleine later
        return loss
