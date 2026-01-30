# flake8: noqa

from typing import List, Tuple, Any, Set, Optional, Dict
from owlready2 import *
import json
from pathlib import Path

class OntologyRefinement:
    """
    Represents a specific refinement condition in the ontology.
    """
    def __init__(self, ref_type: str, property_name: str = None, 
                 operator: str = None, value: Any = None, 
                 concept = None):
        self.ref_type = ref_type # 'concept', 'cardinality', 'domain', 'qualification', 'conjunction'
        self.property = property_name # property name (str)
        self.operator = operator # '>', '==', 'contains', 'is_a'
        self.value = value # threshold or class
        self.concept = concept # owlready2 Class object (for qualification/concept)

    def __repr__(self):
        if self.ref_type == 'concept':
            return f"IsA({self.concept.name})"
        elif self.ref_type == 'cardinality':
            return f"Cardinality({self.property} {self.operator} {self.value})"
        elif self.ref_type == 'domain':
            return f"Domain({self.property} {self.operator} {self.value})"
        elif self.ref_type == 'qualification':
            return f"Exists({self.property}.{self.concept.name})"
        elif self.ref_type == 'conjunction':
            return f"Conjunction({self.value})" # Value is tuple of classes
        return "UnknownRefinement"
    
    def __eq__(self, other):
        return (self.ref_type == other.ref_type and 
                self.property == other.property and 
                self.operator == other.operator and 
                self.value == other.value and
                self.concept == other.concept)
                
    def __hash__(self):
        return hash((self.ref_type, self.property, self.operator, str(self.value), self.concept))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize refinement to a JSON-friendly dict."""
        data: Dict[str, Any] = {
            'ref_type': self.ref_type,
            'property': self.property,
            'operator': self.operator,
            'value': self.value,
        }

        if self.concept is not None:
            data['concept_name'] = getattr(self.concept, 'name', None)
            data['concept_iri'] = getattr(self.concept, 'iri', None)

        if self.ref_type == 'conjunction':
            # value is a tuple of sub-refinements
            data['value'] = [r.to_dict() for r in (self.value or [])]

        return data

    @staticmethod
    def _resolve_concept(onto, concept_iri: Optional[str], concept_name: Optional[str]):
        """Resolve an OWL class in the loaded ontology."""
        if concept_iri:
            found = onto.search_one(iri=concept_iri)
            if found is not None:
                return found
        if concept_name:
            return getattr(onto, concept_name, None)
        return None

    @classmethod
    def from_dict(cls, onto, data: Dict[str, Any]) -> 'OntologyRefinement':
        """Reconstruct refinement from JSON dict using `onto` for class resolution."""
        ref_type = data.get('ref_type')
        prop = data.get('property')
        operator = data.get('operator')
        value = data.get('value')

        if ref_type == 'conjunction' and isinstance(value, list):
            sub_refs = [cls.from_dict(onto, d) for d in value]
            return cls(ref_type='conjunction', value=tuple(sub_refs))

        concept = None
        if ref_type in ('concept', 'qualification') or data.get('concept_name'):
            concept = cls._resolve_concept(
                onto,
                data.get('concept_iri'),
                data.get('concept_name'),
            )

        return cls(
            ref_type=ref_type,
            property_name=prop,
            operator=operator,
            value=value,
            concept=concept,
        )


def save_refinements_json(
    refinements: List[OntologyRefinement],
    out_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save refinements as JSON for later reuse."""
    payload: Dict[str, Any] = {
        'metadata': metadata or {},
        'refinements': [r.to_dict() for r in refinements],
    }

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_refinements_json(onto, path: str) -> List[OntologyRefinement]:
    """Load refinements JSON and resolve referenced concepts in `onto`."""
    p = Path(path)
    payload = json.loads(p.read_text(encoding='utf-8'))
    items = payload.get('refinements', [])
    refinements: List[OntologyRefinement] = []

    for item in items:
        ref = OntologyRefinement.from_dict(onto, item)
        # If a concept couldn't be resolved, we still keep the refinement for
        # numeric/domain/cardinality rules; for concept/qualification this would
        # be unusable, so we skip those.
        if ref.ref_type in ('concept', 'qualification') and ref.concept is None:
            continue
        refinements.append(ref)

    return refinements


class OntologyRefinementGenerator:
    """
    Generates candidate refinements based on the current center class and ontology structure.
    """
    def __init__(self, ontology_manager, static_refinements: Optional[List[OntologyRefinement]] = None):
        self.ontology_manager = ontology_manager
        self.onto = ontology_manager.onto

        # If provided, these candidates are reused across runs (static mode)
        self.static_refinements = static_refinements

        # Safety knobs to avoid refinement explosion on large ontologies (e.g., DTO)
        self.max_qualification_concepts_per_property = 50

        # Cache disjoint class pairs for conjunction filtering
        self._disjoint_pairs = self._build_disjoint_pairs()

    def _build_disjoint_pairs(self) -> Set[frozenset]:
        pairs: Set[frozenset] = set()
        try:
            for dj in self.onto.disjoints():
                entities = list(getattr(dj, 'entities', []) or [])
                if len(entities) < 2:
                    continue
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        c1 = entities[i]
                        c2 = entities[j]
                        if (
                            getattr(c1, 'name', None)
                            and getattr(c2, 'name', None)
                        ):
                            pairs.add(frozenset((c1, c2)))
        except Exception:
            # Some ontologies may not expose disjoint axioms cleanly.
            return set()
        return pairs

    def _is_applicable_domain(self, prop, center_class) -> bool:
        """Return True if prop.domain matches center_class.

        If domain is unspecified, we treat it as applicable.
        """
        domain = getattr(prop, 'domain', None) or []
        if not domain:
            return True
        try:
            center_anc = set(center_class.ancestors())
        except Exception:
            center_anc = {center_class}
        for d in domain:
            if d in center_anc:
                return True
        return False

    def _is_used_by_instances(self, prop, instances: List) -> bool:
        for inst in instances:
            vals = getattr(inst, prop.name, [])
            if vals:
                return True
        return False

    def _get_relevant_object_properties(self, center_class, instances: List):
        props = []
        for prop in self.onto.object_properties():
            if not self._is_applicable_domain(prop, center_class):
                continue
            # If domain is missing/loose, require it to be actually used.
            if (
                not (getattr(prop, 'domain', None) or [])
                and not self._is_used_by_instances(prop, instances)
            ):
                continue
            props.append(prop)
        return props

    def _get_relevant_data_properties(self, center_class, instances: List):
        props = []
        for prop in self.onto.data_properties():
            if prop.name == 'hasLabel':
                continue
            if not self._is_applicable_domain(prop, center_class):
                continue
            if (
                not (getattr(prop, 'domain', None) or [])
                and not self._is_used_by_instances(prop, instances)
            ):
                continue
            props.append(prop)
        return props

    def _concept_is_in_range(self, concept, range_classes: List) -> bool:
        if not range_classes:
            return True
        try:
            anc = set(concept.ancestors())
        except Exception:
            anc = {concept}
        for r in range_classes:
            if r in anc:
                return True
        return False

    def generate_refinements(
        self,
        center_class,
        instances: List,
    ) -> List[OntologyRefinement]:
        """
        Generate all valid refinements for the given center class.
        """
        # Static mode: reuse previously extracted refinements
        # (still filter by current instances)
        if self.static_refinements is not None:
            return self._filter_valid_refinements(
                self.static_refinements,
                instances,
            )

        refinements = []
        
        # 1. Concept Constructor Refinement
        # Direct subclasses of center_class
        refinements.extend(self._generate_concept_refinements(center_class))
        
        # 2. Cardinality Restriction Refinement
        # Properties where domain is center_class (or superclass)
        refinements.extend(
            self._generate_cardinality_refinements(center_class, instances)
        )

        # 3. Domain Restriction Refinement (Data Properties)
        refinements.extend(
            self._generate_domain_refinements(center_class, instances)
        )
        
        # 4. Qualification Refinement
        # Object properties range subclasses
        refinements.extend(
            self._generate_qualification_refinements(center_class, instances)
        )
        
        # 5. Conjunction Refinement
        # Intersection of non-disjoint subclasses
        refinements.extend(
            self._generate_conjunction_refinements(center_class, instances)
        )
        
        # Filter valid refinements (must split instances)
        valid_refinements = self._filter_valid_refinements(
            refinements,
            instances,
        )
        
        return valid_refinements

    def _generate_concept_refinements(
        self,
        center_class,
    ) -> List[OntologyRefinement]:
        refs = []
        try:
            for subclass in center_class.subclasses():
                refs.append(OntologyRefinement('concept', operator='is_a', concept=subclass))
        except:
            pass
        return refs

    def _generate_cardinality_refinements(self, center_class, instances) -> List[OntologyRefinement]:
        refs = []
        # Find object properties relevant to center_class
        # Ideally, query ontology for properties with domain = center_class
        # For typical OWL, domain is loose, so we check all properties or manually defined ones.
        
        # Object properties relevant to center_class (prefer domain-based filtering)
        target_props = self._get_relevant_object_properties(center_class, instances)
        
        for prop in target_props:
            # Check if property makes sense (e.g. hasSubstructure)
            # Generate thresholds based on instance data
            counts = set()
            for inst in instances:
                # Count related objects
                curr_vals = getattr(inst, prop.name, [])
                counts.add(len(curr_vals))
            
            # Suggest thresholds
            sorted_counts = sorted(list(counts))
            for c in sorted_counts[:-1]: # Don't need max
                refs.append(OntologyRefinement('cardinality', property_name=prop.name, operator='>', value=c))
        
        return refs

    def _generate_domain_refinements(self, center_class, instances) -> List[OntologyRefinement]:
        # Data properties
        refs = []
        target_props = self._get_relevant_data_properties(center_class, instances)
        
        for prop in target_props:
            
            # Collect values from instances
            values = []
            for inst in instances:
                val = getattr(inst, prop.name, [])
                if val: values.append(val[0])
            
            if not values: continue
            
            # Create thresholds (simple: unique values or quantiles)
            # For efficiency, just use some quantiles or unique values if few
            unique_vals = sorted(list(set(values)))
            if len(unique_vals) > 10:
                 # Optimize: Use percentiles
                 import numpy as np
                 thresholds = np.percentile(unique_vals, [25, 50, 75])
            else:
                thresholds = unique_vals[:-1]
                
            for t in thresholds:
                refs.append(OntologyRefinement('domain', property_name=prop.name, operator='>', value=t))
                
        return refs

    def _generate_qualification_refinements(self, center_class, instances: List) -> List[OntologyRefinement]:
        """
        Qualification refinement: Exists(R.C) where
        - R is an ObjectProperty
        - C is a class (concept)

        To keep this practical on large ontologies (e.g., DTO.owl), we derive candidate
        concepts from *what we actually observe* in the current instance set.
        """
        refs: List[OntologyRefinement] = []

        target_props = self._get_relevant_object_properties(center_class, instances)
        for prop in target_props:
            observed_concepts: Set[Any] = set()

            # If range is defined, constrain concepts to be within range.
            range_classes = list(getattr(prop, 'range', None) or [])

            for inst in instances:
                related_objects = getattr(inst, prop.name, [])
                if not related_objects:
                    continue

                for obj in related_objects:
                    # owlready2 individuals keep asserted types in .is_a (classes + restrictions)
                    for t in getattr(obj, 'is_a', []):
                        # Keep only OWL classes (ThingClass), not restrictions
                        if hasattr(t, 'name') and hasattr(t, 'ancestors'):
                            observed_concepts.add(t)
                            # Also allow generalization (ancestors) for more usable splits
                            for anc in getattr(t, 'ancestors', lambda: [])():
                                if hasattr(anc, 'name') and hasattr(anc, 'ancestors'):
                                    if anc.name != 'Thing':
                                        observed_concepts.add(anc)

            # Bound per property to avoid combinatorial blow-up
            observed_concepts = {
                c
                for c in observed_concepts
                if getattr(c, 'name', None) not in (None, 'Thing')
                and self._concept_is_in_range(c, range_classes)
            }

            if not observed_concepts:
                continue

            concepts_sorted = sorted(
                observed_concepts,
                key=lambda c: (str(getattr(c, 'name', '')))
            )
            for concept in concepts_sorted[: self.max_qualification_concepts_per_property]:
                refs.append(
                    OntologyRefinement(
                        'qualification',
                        property_name=prop.name,
                        concept=concept,
                    )
                )

        return refs

    def _generate_conjunction_refinements(
        self,
        center_class,
        instances,
    ) -> List[OntologyRefinement]:
        """
        Generate intersection of refinements.
        Strategy (paper-aligned): combine non-disjoint subclasses of center_class.
        """
        refs = []

        concepts = self._generate_concept_refinements(center_class)
        if len(concepts) < 2:
            return []

        # Cap to avoid explosion
        concepts = concepts[:50]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                c1 = concepts[i].concept
                c2 = concepts[j].concept
                if frozenset((c1, c2)) in self._disjoint_pairs:
                    continue
                refs.append(
                    OntologyRefinement(
                        'conjunction',
                        value=(concepts[i], concepts[j]),
                    )
                )

        return refs

    def _filter_valid_refinements(self, refinements, instances) -> List[OntologyRefinement]:
        valid = []
        for ref in refinements:
            satisfying_count = 0
            for inst in instances:
                if self.instance_satisfies_refinement(inst, ref):
                    satisfying_count += 1
            
            if 0 < satisfying_count < len(instances):
                valid.append(ref)
        return valid

    def instance_satisfies_refinement(self, instance, refinement: OntologyRefinement) -> bool:
        """
        Check if instance satisfies the refinement.
        Logic-based check using instance properties.
        """
        if refinement.ref_type == 'concept':
            return isinstance(instance, refinement.concept)
            
        elif refinement.ref_type == 'cardinality':
            vals = getattr(instance, refinement.property, [])
            count = len(vals)
            if refinement.operator == '>': return count > refinement.value
            if refinement.operator == '==': return count == refinement.value
            
        elif refinement.ref_type == 'domain':
            vals = getattr(instance, refinement.property, [])
            if not vals: return False
            val = vals[0]
            if refinement.operator == '>': return val > refinement.value
            
        elif refinement.ref_type == 'qualification':
            # Exists prop.concept
            related_objects = getattr(instance, refinement.property, [])
            for obj in related_objects:
                if isinstance(obj, refinement.concept):
                    return True
            return False
            
        elif refinement.ref_type == 'conjunction':
            # Check all refinements in the value tuple
            # refinement.value is (ref1, ref2)
            sub_refinements = refinement.value
            for sub_ref in sub_refinements:
                if not self.instance_satisfies_refinement(instance, sub_ref):
                    return False
            return True
            
        return False
