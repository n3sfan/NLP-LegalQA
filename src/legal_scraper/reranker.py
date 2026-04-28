"""Reranker module using AITeamVN/Vietnamese_Reranker."""

from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class VietnameseReranker:
    """A cross-encoder reranker for Vietnamese text using AITeamVN/Vietnamese_Reranker.
    Loads the model in FP16 to stay well within 4GB VRAM limits.
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Reranker",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2304,
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load in half precision (FP16) to fit inside 4GB VRAM
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5, batch_size: int = 4
    ) -> List[Tuple[int, float]]:
        """Rerank a list of candidate documents against the query in mini-batches.

        Args:
            query: The user query string.
            documents: List of contextual document texts to score.
            top_k: Number of top results to return.
            batch_size: Number of documents to process in a single GPU pass. 
                       Small values (1-4) are recommended for 4GB VRAM.

        Returns:
            A list of tuples (original_index, score), sorted descending by score.
        """
        if not documents:
            return []

        # Form the correct pairs for the cross-encoder: (query, document)
        pairs = [[query, doc] for doc in documents]
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                ).to(self.device)
                
                # Model returns logits; we take them as scores directly
                batch_logits = self.model(**inputs, return_dict=True).logits.view(-1,).float()
                all_scores.append(batch_logits)
            
            # Combine scores from all batches
            scores = torch.cat(all_scores)

        # Convert scores to pure python floats
        score_list = scores.cpu().tolist()

        # Pair each score with its original index
        indexed_scores = list(enumerate(score_list))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]
