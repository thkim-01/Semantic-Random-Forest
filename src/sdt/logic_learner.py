from typing import List, Optional, Tuple
import os
import numpy as np
from src.sdt.tree import TreeNode, SemanticDecisionTree
from src.sdt.logic_refinement import (
    OntologyRefinementGenerator,
    load_refinements_json,
)


class LogicTreeNode(TreeNode):
    """TreeNode subclass for Logic SDT that handles Ontology instances."""
    def __init__(self, instances: List, depth: int = 0, node_id: int = 0):
        self.instances = instances
        self.depth = depth
        self.node_id = node_id
        self.refinement = None
        self.is_leaf = False

        self.predicted_label = None
        self.left_child = None
        self.right_child = None
        
        self.num_instances = len(instances)
        self.label_counts = self._count_labels()
        # Note: We calculate unweighted entropy here for initialization,
        # but Learner might recalculate weighted entropy.
        self.entropy = self._calculate_entropy()

    def _get_label_from_inst(self, inst):
        if hasattr(inst, 'hasLabel'):  # Ontology instance
            return inst.hasLabel[0] if inst.hasLabel else None
        return getattr(inst, 'label', None)

    def _count_labels(self) -> dict:
        counts = {}
        for inst in self.instances:
            label = self._get_label_from_inst(inst)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _calculate_entropy(self) -> float:
        if self.num_instances == 0:
            return 0.0
        entropy = 0.0
        total = sum(self.label_counts.values())
        if total == 0:
            return 0.0
        
        for count in self.label_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy


class LogicSDTLearner:
    """
    Semantic Decision Tree Learner using Ontology/Logic-based Refinements.
    """
    
    def __init__(
        self,
        ontology_manager,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = None,
        verbose: bool = True,
        refinement_mode: str = 'dynamic',
        refinement_file: Optional[str] = None,
        split_criterion: str = 'information_gain',
    ):
        self.onto_manager = ontology_manager
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.verbose = verbose

        self.refinement_mode = refinement_mode
        self.refinement_file = refinement_file
        self.split_criterion = split_criterion

        static_refs = None
        if refinement_mode == 'static':
            if not refinement_file:
                raise ValueError(
                    "refinement_file is required when refinement_mode='static'"
                )
            if not os.path.exists(refinement_file):
                raise FileNotFoundError(refinement_file)
            static_refs = load_refinements_json(
                ontology_manager.onto,
                refinement_file,
            )

        self.refinement_generator = OntologyRefinementGenerator(
            ontology_manager,
            static_refinements=static_refs,
        )
        self.tree = None
        self.class_weights_dict = {}

    def fit(self, instances: List):
        """Train the SDT"""
        self.tree = SemanticDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        
        if self.class_weight == 'balanced':
            self.class_weights_dict = self._compute_class_weights(instances)
            if self.verbose:
                print(f"Class weights: {self.class_weights_dict}")
        
        # Root node
        root_node = LogicTreeNode(
            instances,
            depth=0,
            node_id=self.tree.get_next_node_id(),
        )
        self.tree.root = root_node
        self.tree.nodes.append(root_node)
        
        # Start recursive build with initial center class = Molecule
        self._build_tree(root_node, center_class=self.onto_manager.Molecule)
        
        if self.verbose:
            total_nodes = len(self.tree.nodes)
            print(f"Logic SDT training completed. Total nodes: {total_nodes}")
            
        return self.tree

    def _build_tree(self, node: TreeNode, center_class):
        """Recursive tree building"""
        # Stop conditions
        unique_labels = {
            self._get_label(inst)
            for inst in node.instances
        }
        if (
            node.depth >= self.max_depth
            or node.num_instances < self.min_samples_split
            or len(unique_labels) == 1
        ):
            self._set_leaf(node)
            return

        # Generate candidate refinements
        # In True SDT, center_class might evolve, but here we keep it as
        # Molecule for V1
        # or update it if we handle qualification refinements.
        candidate_refinements = self.refinement_generator.generate_refinements(
            center_class,
            node.instances,
        )
        
        if not candidate_refinements:
            self._set_leaf(node)
            return

        # Find best split
        best_refinement = None
        best_score = -1.0
        best_sets = None
        
        for ref in candidate_refinements:
            satisfying, non_satisfying = self._split_instances(
                node.instances,
                ref,
            )
            
            if (
                len(satisfying) < self.min_samples_leaf
                or len(non_satisfying) < self.min_samples_leaf
            ):
                continue
                
            score = self._calculate_split_score(
                parent_node=node,
                left_inst=satisfying,
                right_inst=non_satisfying,
            )
            
            if score > best_score:
                best_score = score
                best_refinement = ref
                best_sets = (satisfying, non_satisfying)
        
        if best_refinement and best_sets:
            node.set_refinement(best_refinement)
            
            left_instances, right_instances = best_sets
            
            # Create children
            node.left_child = LogicTreeNode(
                left_instances,
                depth=node.depth + 1,
                node_id=self.tree.get_next_node_id(),
            )
            node.right_child = LogicTreeNode(
                right_instances,
                depth=node.depth + 1,
                node_id=self.tree.get_next_node_id(),
            )
            
            self.tree.nodes.append(node.left_child)
            self.tree.nodes.append(node.right_child)
            
            # Recurse
            # Logic for updating Center Class
            left_center = center_class
            # If split was IsA(Concept), the left child (satisfying) is now
            # of type Concept.
            if best_refinement.ref_type == 'concept':
                left_center = best_refinement.concept

            # Search space expansion (paper behavior): if we split on an object
            # property refinement, the satisfying branch moves the center class
            # to the property's range class.
            if best_refinement.ref_type in ('cardinality', 'qualification'):
                left_center = self._get_object_property_range_center(
                    best_refinement.property,
                    satisfying,
                    fallback=center_class,
                )
                
            self._build_tree(node.left_child, left_center)
            # Right child (non-satisfying) remains the parent center class
            self._build_tree(node.right_child, center_class)
        else:
            self._set_leaf(node)

    def _get_object_property_range_center(
        self,
        property_name: Optional[str],
        satisfying_instances: List,
        fallback,
    ):
        """Resolve the range class for an object property.

        Priority:
        1) ontology ObjectProperty.range (first non-Thing)
        2) infer from observed related object types in satisfying instances
        3) fallback
        """
        if not property_name:
            return fallback

        prop = getattr(self.onto_manager.onto, property_name, None)
        if prop is not None:
            rng = getattr(prop, 'range', None) or []
            for r in rng:
                if getattr(r, 'name', None) and r.name != 'Thing':
                    return r

        # Fallback: infer from observed related object types
        type_counts = {}
        for inst in satisfying_instances:
            related = getattr(inst, property_name, [])
            for obj in related:
                for t in getattr(obj, 'is_a', []):
                    if getattr(t, 'name', None) and t.name != 'Thing':
                        type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            return max(type_counts, key=type_counts.get)

        return fallback

    def _calculate_split_score(
        self,
        parent_node,
        left_inst,
        right_inst,
    ) -> float:
        """Choose split scoring: information gain or C4.5 gain ratio."""
        if self.split_criterion in ('gain_ratio', 'c45_gain_ratio'):
            return self._calculate_gain_ratio(
                parent_node,
                left_inst,
                right_inst,
            )
        return self._calculate_information_gain(
            parent_node,
            left_inst,
            right_inst,
        )

    def _calculate_gain_ratio(
        self,
        parent_node,
        left_inst,
        right_inst,
    ) -> float:
        """C4.5 gain ratio = information_gain / split_info."""
        gain = self._calculate_information_gain(
            parent_node,
            left_inst,
            right_inst,
        )
        if self.class_weights_dict:
            n = self._get_total_weight(parent_node.instances)
            n_left = self._get_total_weight(left_inst)
            n_right = self._get_total_weight(right_inst)
        else:
            n = len(parent_node.instances)
            n_left = len(left_inst)
            n_right = len(right_inst)

        if n <= 0:
            return 0.0

        p_left = n_left / n
        p_right = n_right / n
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)

        if split_info <= 0:
            return 0.0

        return gain / split_info

    def _split_instances(self, instances, refinement) -> Tuple[List, List]:
        satisfying = []
        non_satisfying = []
        for inst in instances:
            if self.refinement_generator.instance_satisfies_refinement(
                inst,
                refinement,
            ):
                satisfying.append(inst)
            else:
                non_satisfying.append(inst)
        return satisfying, non_satisfying

    def _get_label(self, inst) -> int:
        """Helper to get label from instance (Ontology or Object)"""
        if hasattr(inst, 'hasLabel'):  # Ontology instance
            return inst.hasLabel[0] if inst.hasLabel else None
        return getattr(inst, 'label', None)

    def _set_leaf(self, node: TreeNode):
        """Set node as leaf"""
        labels = [self._get_label(inst) for inst in node.instances]
        labels = [
            label
            for label in labels
            if label is not None
        ]
        
        if not labels:
            predicted = 0
        else:
            # Weighted majority vote if weights exist
            if self.class_weights_dict:
                vote = {0: 0.0, 1: 0.0}
                for label in labels:
                    vote[label] += self.class_weights_dict.get(label, 1.0)
                predicted = max(vote, key=vote.get)
            else:
                predicted = max(set(labels), key=labels.count)
        
        node.set_as_leaf(predicted)

    def _compute_class_weights(self, instances: List) -> dict:
        labels = [self._get_label(inst) for inst in instances]
        labels = [label for label in labels if label is not None]
        
        if not labels:
            return {}
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        n_classes = len(unique)
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (n_classes * count)
        return weights

    def _get_total_weight(self, instances: List) -> float:
        if not self.class_weights_dict:
            return float(len(instances))
        return sum(
            self.class_weights_dict.get(self._get_label(inst), 1.0)
            for inst in instances
        )

    def _calculate_entropy(self, instances: List) -> float:
        if not instances:
            return 0.0
        
        label_counts = {}
        for inst in instances:
            label = self._get_label(inst)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
            
        if self.class_weights_dict:
            weighted_counts = {
                label: count * self.class_weights_dict.get(label, 1.0)
                for label, count in label_counts.items()
            }
            total = sum(weighted_counts.values())
            counts_eval = weighted_counts
        else:
            total = len(instances)
            counts_eval = label_counts
            
        entropy = 0.0
        for count in counts_eval.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_information_gain(
        self,
        parent_node,
        left_inst,
        right_inst,
    ) -> float:
        if self.class_weights_dict:
            n = self._get_total_weight(parent_node.instances)
            n_left = self._get_total_weight(left_inst)
            n_right = self._get_total_weight(right_inst)
        else:
            n = len(parent_node.instances)
            n_left = len(left_inst)
            n_right = len(right_inst)
            
        if n == 0:
            return 0.0
        
        parent_entropy = self._calculate_entropy(parent_node.instances)
        left_entropy = self._calculate_entropy(left_inst)
        right_entropy = self._calculate_entropy(right_inst)
        
        gain = (
            parent_entropy
            - (n_left / n) * left_entropy
            - (n_right / n) * right_entropy
        )
        return gain
