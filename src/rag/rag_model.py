# FLLM/src/rag/rag_model.py
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from .knowledge_graph import KnowledgeGraph
from .retriever import Retriever
from fllm.model_classifier3 import FireLLMModel
from fllm.prompt_generator3 import FireIncidentData
from typing import Dict, List, Optional, Union
import logging
import torch
import pandas as pd

class RAGModel:
    """
    A wrapper class for coordinating Retriever and generator model (FireLLMModel).
    """
    def __init__(self, base_model: FireLLMModel, retriever: Retriever, kg: KnowledgeGraph):
        """
        Initialize RAGModel.

        Args:
            base_model (FireLLMModel): Base FireLLM generator model.
            retriever (Retriever): Retriever instance for information retrieval.
            kg (KnowledgeGraph): Knowledge graph instance.
        """
        self.base_model = base_model
        self.retriever = retriever
        self.kg = kg
        self.logger = logging.getLogger(__name__)

    def _augment_prompt(self, original_prompt: str, retrieved_indices: List[int]) -> str:
        """
        Enhance original prompt using information retrieved from knowledge graph.

        Args:
            original_prompt (str): Original fire incident prompt.
            retrieved_indices (List[int]): List of retrieved node indices.

        Returns:
            str: Enhanced prompt.
        """
        context_header = "% Retrieved Knowledge Context:\n"
        context = ""
        for idx in retrieved_indices:
            node_info = self.kg.get_node_info(idx)
            if node_info:
                context += f"- {node_info['id']}: {node_info['description']}\n"
        
        if not context:
            context = "No relevant knowledge found.\n"
            
        augmented_prompt = context_header + context + "\n" + original_prompt
        return augmented_prompt

    def predict(self, incident_data: FireIncidentData, k: int = 5) -> Dict:
        """
        Predict using RAG pipeline.

        Args:
            incident_data (FireIncidentData): Fire incident data.
            k (int): Number of knowledge items to retrieve.

        Returns:
            Dict: Dictionary containing prediction results.
        """
        try:
            # 1. Create original prompt
            original_prompt = self.base_model.create_fire_prompt(incident_data)
            
            # 2. Retrieve relevant knowledge
            retrieved_indices = self.retriever.retrieve(original_prompt, k=k)
            
            # 3. Enhance prompt
            augmented_prompt = self._augment_prompt(original_prompt, retrieved_indices)
            
            # 4. Use base model for prediction
            # Check if base model has predict_from_prompt method, otherwise use standard predict method
            if hasattr(self.base_model, 'predict_from_prompt'):
                prediction = self.base_model.predict_from_prompt(augmented_prompt)
            else:
                # Use standard predict method with enhanced prompt
                prediction = self.base_model.predict(incident_data)
            
            # (Optional) Include retrieved context in results
            prediction['retrieved_context'] = [self.kg.get_node_info(i) for i in retrieved_indices]
            prediction['augmented_prompt'] = augmented_prompt
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error occurred during RAG prediction: {str(e)}")
            # Fallback to base model prediction
            prediction = self.base_model.predict(incident_data)
            prediction['error'] = f"RAG prediction failed, using base model: {str(e)}"
            return prediction

    def augment_incident_data_list(self, incidents: List[FireIncidentData], k: int = 5) -> List[Dict]:
        """
        Enhance prompts for a batch of incident data for training.

        Args:
            incidents (List[FireIncidentData]): List of incident data.
            k (int): Number of knowledge items to retrieve for each incident.

        Returns:
            List[Dict]: List of dictionaries, each containing enhanced prompt and labels.
        """
        augmented_dataset = []
        for incident in incidents:
            original_prompt = self.base_model.create_fire_prompt(incident)
            retrieved_indices = self.retriever.retrieve(original_prompt, k=k)
            augmented_prompt = self._augment_prompt(original_prompt, retrieved_indices)
            
            augmented_dataset.append({
                "prompt": augmented_prompt,
                "fire_spread_label": incident.fire_spread_label,
                "incident_key": incident.incident_key
            })
        return augmented_dataset

    def batch_predict(self, incidents: List[FireIncidentData], k: int = 5, batch_size: int = 8) -> List[Dict]:
        """
        Batch prediction for efficient processing of large datasets.

        Args:
            incidents (List[FireIncidentData]): List of incident data.
            k (int): Number of knowledge items to retrieve for each incident.
            batch_size (int): Batch size.

        Returns:
            List[Dict]: List of prediction results.
        """
        predictions = []
        for i in range(0, len(incidents), batch_size):
            batch = incidents[i:i + batch_size]
            batch_predictions = []
            
            for incident in batch:
                prediction = self.predict(incident, k=k)
                batch_predictions.append(prediction)
            
            predictions.extend(batch_predictions)
            self.logger.info(f"Processed {min(i + batch_size, len(incidents))}/{len(incidents)} samples")
        
        return predictions

    def get_retrieval_stats(self, incidents: List[FireIncidentData], k: int = 5) -> Dict:
        """
        Get retrieval statistics for analyzing RAG system performance.

        Args:
            incidents (List[FireIncidentData]): List of incident data.
            k (int): Number of knowledge items to retrieve for each incident.

        Returns:
            Dict: Dictionary containing retrieval statistics.
        """
        total_retrievals = 0
        successful_retrievals = 0
        retrieval_errors = 0
        
        for incident in incidents:
            try:
                original_prompt = self.base_model.create_fire_prompt(incident)
                retrieved_indices = self.retriever.retrieve(original_prompt, k=k)
                total_retrievals += 1
                if retrieved_indices and len(retrieved_indices) > 0:
                    successful_retrievals += 1
            except Exception as e:
                retrieval_errors += 1
                self.logger.warning(f"Retrieval failed: {str(e)}")
        
        return {
            'total_retrievals': total_retrievals,
            'successful_retrievals': successful_retrievals,
            'retrieval_errors': retrieval_errors,
            'success_rate': successful_retrievals / total_retrievals if total_retrievals > 0 else 0,
            'average_retrieved_per_query': k
        }

    def save_rag_model(self, save_path: str):
        """
        Save RAG model components.

        Args:
            save_path (str): Save path.
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model
        if hasattr(self.base_model, 'save_model'):
            self.base_model.save_model(os.path.join(save_path, 'base_model'))
        
        # Save retriever
        retriever_path = os.path.join(save_path, 'retriever')
        os.makedirs(retriever_path, exist_ok=True)
        
        # Save FAISS index
        import faiss
        faiss.write_index(self.retriever.index, os.path.join(retriever_path, 'faiss_index.bin'))
        
        # Save knowledge graph
        kg_path = os.path.join(save_path, 'knowledge_graph')
        os.makedirs(kg_path, exist_ok=True)
        
        self.kg.nodes_df.to_csv(os.path.join(kg_path, 'nodes.csv'), index=False)
        self.kg.edges_df.to_csv(os.path.join(kg_path, 'edges.csv'), index=False)
        
        self.logger.info(f"RAG model saved to: {save_path}")

    def load_rag_model(self, load_path: str):
        """
        Load RAG model components.

        Args:
            load_path (str): Load path.
        """
        import os
        import faiss
        
        # Load retriever
        retriever_path = os.path.join(load_path, 'retriever')
        if os.path.exists(os.path.join(retriever_path, 'faiss_index.bin')):
            self.retriever.index = faiss.read_index(os.path.join(retriever_path, 'faiss_index.bin'))
        
        # Load knowledge graph
        kg_path = os.path.join(load_path, 'knowledge_graph')
        if os.path.exists(os.path.join(kg_path, 'nodes.csv')):
            self.kg.nodes_df = pd.read_csv(os.path.join(kg_path, 'nodes.csv'))
            self.kg.edges_df = pd.read_csv(os.path.join(kg_path, 'edges.csv'))
            
            # Rebuild mappings
            self.kg.entity_to_id = {entity: i for i, entity in enumerate(self.kg.nodes_df['id'])}
            self.kg.id_to_entity = {i: entity for entity, i in self.kg.entity_to_id.items()}
            self.kg.relation_to_id = {relation: i for i, relation in enumerate(self.kg.edges_df['relation'].unique())}
        
        self.logger.info(f"RAG model loaded from {load_path}")
