# FLLM/src/rag/knowledge_graph.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
import os
import json

class KnowledgeGraph:
    """
    Class for loading, processing, and querying fire knowledge graphs.
    """

    def __init__(self, nodes_path: str, edges_path: str):
        """
        Initialize knowledge graph.

        Args:
            nodes_path (str): Path to nodes CSV file.
            edges_path (str): Path to edges CSV file.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.nodes_df = pd.read_csv(nodes_path)
            self.edges_df = pd.read_csv(edges_path)
            self.logger.info(f"Successfully loaded knowledge graph: {len(self.nodes_df)} nodes, {len(self.edges_df)} edges.")

            # Create entity to ID mapping
            self.entity_to_id = {entity: i for i, entity in enumerate(self.nodes_df['id'])}
            self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}

            # Create relation to ID mapping
            self.relation_to_id = {relation: i for i, relation in enumerate(self.edges_df['relation'].unique())}

            self.num_nodes = len(self.entity_to_id)
            self.num_relations = len(self.relation_to_id)
            self.logger.info(f"Graph contains {self.num_nodes} unique nodes and {self.num_relations} unique relations.")

        except FileNotFoundError as e:
            self.logger.error(f"Failed to load knowledge graph files: {e}")
            raise

    def get_triplets(self) -> Tuple[List[Tuple[int, int, int]], Dict, Dict]:
        """
        Convert graph data to list of (head entity ID, relation ID, tail entity ID) triplets.

        Returns:
            Tuple containing the list of triplets and mapping dictionaries.
        """
        triplets = []
        for _, row in self.edges_df.iterrows():
            source_id = self.entity_to_id.get(row['source'])
            target_id = self.entity_to_id.get(row['target'])
            relation_id = self.relation_to_id.get(row['relation'])

            if source_id is not None and target_id is not None and relation_id is not None:
                triplets.append((source_id, relation_id, target_id))

        return triplets, self.id_to_entity, self.relation_to_id

    def get_node_info(self, node_id: int) -> Dict:
        """
        Get node information by node ID.

        Args:
            node_id (int): Node ID.

        Returns:
            Dict: Dictionary containing node information.
        """
        entity_name = self.id_to_entity.get(node_id)
        if entity_name:
            node_data = self.nodes_df[self.nodes_df['id'] == entity_name].iloc[0]
            return {
                "id": node_data['id'],
                "description": node_data.get('description', 'No description available.'),
                "type": node_data.get('type', 'Generic')
            }
        return None

    def get_entity_by_name(self, entity_name: str) -> Optional[Dict]:
        """
        Get entity information by entity name.

        Args:
            entity_name (str): Entity name.

        Returns:
            Optional[Dict]: Dictionary containing entity information, None if not found.
        """
        if entity_name in self.entity_to_id:
            node_id = self.entity_to_id[entity_name]
            return self.get_node_info(node_id)
        return None

    def get_related_entities(self, entity_name: str, relation_type: Optional[str] = None) -> List[Dict]:
        """
        Get entities related to the specified entity.

        Args:
            entity_name (str): Entity name.
            relation_type (Optional[str]): Relation type, if None returns all relations.

        Returns:
            List[Dict]: List of related entity information.
        """
        if entity_name not in self.entity_to_id:
            return []
        
        entity_id = self.entity_to_id[entity_name]
        related_entities = []
        
        # Find relations where this entity is the source
        source_relations = self.edges_df[self.edges_df['source'] == entity_name]
        if relation_type:
            source_relations = source_relations[source_relations['relation'] == relation_type]
        
        for _, row in source_relations.iterrows():
            target_info = self.get_entity_by_name(row['target'])
            if target_info:
                target_info['relation'] = row['relation']
                target_info['relation_direction'] = 'outgoing'
                related_entities.append(target_info)
        
        # Find relations where this entity is the target
        target_relations = self.edges_df[self.edges_df['target'] == entity_name]
        if relation_type:
            target_relations = target_relations[target_relations['relation'] == relation_type]
        
        for _, row in target_relations.iterrows():
            source_info = self.get_entity_by_name(row['source'])
            if source_info:
                source_info['relation'] = row['relation']
                source_info['relation_direction'] = 'incoming'
                related_entities.append(source_info)
        
        return related_entities

    def get_entity_types(self) -> List[str]:
        """
        Get all entity types.

        Returns:
            List[str]: List of entity types.
        """
        if 'type' in self.nodes_df.columns:
            return self.nodes_df['type'].unique().tolist()
        return []

    def get_relation_types(self) -> List[str]:
        """
        Get all relation types.

        Returns:
            List[str]: List of relation types.
        """
        return list(self.relation_to_id.keys())

    def get_statistics(self) -> Dict:
        """
        Get knowledge graph statistics.

        Returns:
            Dict: Dictionary containing statistical information.
        """
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': len(self.edges_df),
            'num_relations': self.num_relations,
            'entity_types': self.get_entity_types(),
            'relation_types': self.get_relation_types(),
            'avg_degree': 0,
            'max_degree': 0,
            'min_degree': 0
        }
        
        # Calculate degree statistics
        if len(self.edges_df) > 0:
            # Calculate degree for each node
            node_degrees = {}
            for _, row in self.edges_df.iterrows():
                source = row['source']
                target = row['target']
                node_degrees[source] = node_degrees.get(source, 0) + 1
                node_degrees[target] = node_degrees.get(target, 0) + 1
            
            if node_degrees:
                degrees = list(node_degrees.values())
                stats['avg_degree'] = np.mean(degrees)
                stats['max_degree'] = np.max(degrees)
                stats['min_degree'] = np.min(degrees)
        
        return stats

    def search_entities(self, query: str, search_fields: List[str] = None) -> List[Dict]:
        """
        Search entities containing the specified query string.

        Args:
            query (str): Search query.
            search_fields (List[str]): List of fields to search, defaults to ['id', 'description'].

        Returns:
            List[Dict]: List of matching entity information.
        """
        if search_fields is None:
            search_fields = ['id', 'description']
        
        matches = []
        query_lower = query.lower()
        
        for _, row in self.nodes_df.iterrows():
            for field in search_fields:
                if field in row and pd.notna(row[field]):
                    if query_lower in str(row[field]).lower():
                        entity_info = {
                            'id': row['id'],
                            'description': row.get('description', ''),
                            'type': row.get('type', 'Generic'),
                            'matched_field': field
                        }
                        matches.append(entity_info)
                        break
        
        return matches

    def get_subgraph(self, center_entity: str, max_depth: int = 2) -> Dict:
        """
        Get subgraph centered on the specified entity.

        Args:
            center_entity (str): Center entity name.
            max_depth (int): Maximum depth.

        Returns:
            Dict: Dictionary containing subgraph information.
        """
        if center_entity not in self.entity_to_id:
            return {'nodes': [], 'edges': []}
        
        visited_entities = set()
        subgraph_nodes = []
        subgraph_edges = []
        
        def explore_entity(entity_name, current_depth):
            if current_depth > max_depth or entity_name in visited_entities:
                return
            
            visited_entities.add(entity_name)
            entity_info = self.get_entity_by_name(entity_name)
            if entity_info:
                subgraph_nodes.append(entity_info)
            
            # Get related entities
            related_entities = self.get_related_entities(entity_name)
            for related in related_entities:
                if related['id'] not in visited_entities:
                    subgraph_edges.append({
                        'source': entity_name,
                        'target': related['id'],
                        'relation': related['relation'],
                        'direction': related['relation_direction']
                    })
                    explore_entity(related['id'], current_depth + 1)
        
        explore_entity(center_entity, 0)
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'center_entity': center_entity,
            'max_depth': max_depth
        }

    def save_graph(self, save_path: str):
        """
        Save knowledge graph to files.

        Args:
            save_path (str): Save path.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save nodes and edges data
        self.nodes_df.to_csv(os.path.join(save_path, 'nodes.csv'), index=False)
        self.edges_df.to_csv(os.path.join(save_path, 'edges.csv'), index=False)
        
        # Save statistics
        stats = self.get_statistics()
        with open(os.path.join(save_path, 'statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Save mapping information
        mappings = {
            'entity_to_id': self.entity_to_id,
            'id_to_entity': self.id_to_entity,
            'relation_to_id': self.relation_to_id
        }
        with open(os.path.join(save_path, 'mappings.json'), 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Knowledge graph saved to: {save_path}")

    def validate_graph(self) -> Dict:
        """
        Validate knowledge graph integrity.

        Returns:
            Dict: Dictionary containing validation results.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check node data
        if self.nodes_df.empty:
            validation_results['errors'].append("Node data is empty")
            validation_results['is_valid'] = False
        
        if 'id' not in self.nodes_df.columns:
            validation_results['errors'].append("Node data missing 'id' column")
            validation_results['is_valid'] = False
        
        # Check edge data
        if self.edges_df.empty:
            validation_results['warnings'].append("Edge data is empty")
        
        if not self.edges_df.empty:
            required_edge_columns = ['source', 'target', 'relation']
            for col in required_edge_columns:
                if col not in self.edges_df.columns:
                    validation_results['errors'].append(f"Edge data missing '{col}' column")
                    validation_results['is_valid'] = False
            
            # Check if entities in edges exist in nodes
            if 'source' in self.edges_df.columns and 'target' in self.edges_df.columns:
                all_entities = set(self.nodes_df['id'].unique())
                edge_entities = set(self.edges_df['source'].unique()) | set(self.edges_df['target'].unique())
                missing_entities = edge_entities - all_entities
                if missing_entities:
                    validation_results['warnings'].append(f"Edges reference non-existent entities: {list(missing_entities)[:10]}")
        
        return validation_results
