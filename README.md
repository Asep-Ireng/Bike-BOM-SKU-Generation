# Bike BOM & SKU Generation: Hybrid GraphDB + GNN Architecture

## 1. Executive Summary & Problem Formulation
Configuring a valid bicycle Bill of Materials (BOM) from dynamic warehouse stock is a combinatorial optimization and constraint satisfaction problem. 

* **Physical Feasibility (Hard Constraints):** Components must mechanically mate across non-negotiable interface standards (e.g., Bottom Bracket shell type, headtube/steerer taper, rear axle spacing, freehub driver body, brake caliper mounting).
* **Inventory Utilization (Soft Constraints / Business Objectives):** Assembly specs must prioritize overstocked parts, account for procurement lead times, maintain target profit margins, and keep component quality tiers cohesive.

Pure probabilistic models (like standard LLMs) fail because they hallucinate mechanical compatibilities and lack deterministic inventory state tracking. Pure GraphDB queries struggle with combinatorial explosion and multi-objective heuristic ranking. 

**Solution:** A **Neuro-Symbolic Hybrid Architecture** pairing **GraphDB (Hard Constraint Filtering & Live Inventory Truth)** with a **Graph Neural Network (GNN / Link Prediction & Subgraph Scoring Head)**.

---

## 2. System Architecture

```
                                +---------------------------+
                                | Target Objective / Spec   |
                                | (e.g., Gravel Mid-Tier,   |
                                |  Clear Old Stock)         |
                                +-------------+-------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
| STEP 1: GRAPH DATABASE (Deterministic Constraint Engine - Neo4j / Memgraph)             |
|                                                                                         |
|  * Filters parts with stock_qty > 0                                                     |
|  * Traverses hard interface standards (BSA_73, BOOST_148, TAPERED_ZS44_56, XD_DRIVER)    |
|  * Prunes 95%+ of physically incompatible combinations                                  |
|  * Extracts Candidate Subgraph of valid mating components                               |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
| STEP 2: GRAPH NEURAL NETWORK (Scoring & Ranking Layer - PyTorch Geometric / DGL)        |
|                                                                                         |
|  * Embeds candidate components using Heterogeneous Message Passing (R-GCN / HGT)       |
|  * Node Features: [normalized_cost, stock_level, procurement_type, tier, weight]        |
|  * Evaluates link compatibility & multi-objective reward score:                         |
|      Score(u, v) = CompatScore(u, v) + w1*StockBurn(v) - w2*TierDeviation(u, v)        |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
| STEP 3: SKU COMPOSITION & INVENTORY STATE WRITE-BACK                                    |
|                                                                                         |
|  * Assembles top-ranked complete build tree                                             |
|  * Generates final SKU BOM spec sheet                                                   |
|  * Atomic stock decrement in DB / ERP                                                   |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Data Ingestion & Form-to-Graph Mapping

Existing inventory forms provide structured key-value attributes. These are split into two categories:

### A. Interface Connectors (Graph Relationships / Structural Edges)
Direct dimensional interfaces defining mechanical compatibility:

| Component Pair | Interface Standard Fields | Canonical ID Examples |
|---|---|---|
| **Frame $\leftrightarrow$ Fork** | Headtube standard, steerer taper | `TAPERED_1.5_1.125`, `ZS44_ZS56` |
| **Frame $\leftrightarrow$ Rear Wheel** | Rear axle standard, spacing | `BOOST_148_12`, `NON_BOOST_142_12` |
| **Frame $\leftrightarrow$ Bottom Bracket** | BB shell standard, shell width | `BSA_THREADED_73`, `PF92_PRESSFIT` |
| **BB $\leftrightarrow$ Crankset** | Spindle standard, spindle diameter | `SRAM_DUB_28.99`, `SHIMANO_HT2_24` |
| **Rear Wheel $\leftrightarrow$ Cassette** | Freehub body driver standard | `SHIMANO_MICROSPLINE`, `SRAM_XD`, `HG_SPLINE` |
| **Frame/Fork $\leftrightarrow$ Brakes** | Caliper mount type, rotor size | `POST_MOUNT_180`, `FLAT_MOUNT_160` |

### B. Node Features (Continuous & Categorical Embeddings)
Attributes used by the GNN for feature representation and scoring:
* `stock_qty` (Integer, dynamically normalized)
* `unit_cost` (Float, normalized)
* `weight_grams` (Float)
* `material` (Categorical One-Hot: Alloy, Carbon, Chromoly, Titanium)
* `procurement_type` (Categorical One-Hot: CKD, CBU, Local, Import, In-House)
* `component_tier` (Ordinal / Embedding: Entry, Mid, High, Pro)

---

## 4. Graph Schema Design (Neo4j / Cypher)

Using **Shared Standard Nodes** simplifies graph expansion. Adding a new component requires connecting it only to its standard node rather than every existing mating part.

```cypher
// 1. Create Constraints
CREATE CONSTRAINT FOR (p:Part) REQUIRE p.sku IS UNIQUE;
CREATE CONSTRAINT FOR (s:Standard) REQUIRE s.code IS UNIQUE;

// 2. Canonical Interface Nodes
MERGE (:Standard {code: 'BSA_73', type: 'BB_SHELL'});
MERGE (:Standard {code: 'DUB_SPINDLE', type: 'CRANK_SPINDLE'});
MERGE (:Standard {code: 'BOOST_148_12', type: 'REAR_AXLE'});

// 3. Ingesting Components & Relationships
MERGE (f:Part:Frame {sku: 'FRM-AL-29-01', name: 'Trail Hardtail 29', stock: 45, cost: 250.0, tier: 2})
MERGE (bb:Part:BottomBracket {sku: 'BB-SRAM-DUB-BSA', name: 'SRAM DUB BSA 73mm', stock: 120, cost: 38.0, tier: 2})
MERGE (c:Part:Crankset {sku: 'CRK-GX-DUB-170', name: 'SRAM GX Eagle DUB', stock: 15, cost: 140.0, tier: 2})

// Connect via Standards
MERGE (f)-[:HAS_BB_SHELL]->(:Standard {code: 'BSA_73'})
MERGE (bb)-[:FITS_SHELL]->(:Standard {code: 'BSA_73'})
MERGE (bb)-[:ACCEPTS_SPINDLE]->(:Standard {code: 'DUB_SPINDLE'})
MERGE (c)-[:HAS_SPINDLE]->(:Standard {code: 'DUB_SPINDLE'})
```

---

## 5. Machine Learning Formulation (PyTorch Geometric)

The assembly is represented as a `HeteroData` object in PyG.

### Graph Neural Network Architecture
* **Backbone:** Heterogeneous Graph Transformer (HGT) or Relational Graph Convolutional Network (R-GCN) to pass messages across diverse node types (`Frame`, `Fork`, `Wheelset`, `Drivetrain`).
* **Link Predictor / Edge Scorer:** Multi-Layer Perceptron (MLP) calculating compatibility and reward scores over concatenated node embeddings.

```python
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

class BikeBOMScorer(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=32, num_heads=4, num_layers=2):
        super().__init__()
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict):
        # 1. Project heterogeneous input features to common dimension
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Relational Message Passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

    def score_edge(self, emb_src, emb_dst):
        cat_emb = torch.cat([emb_src, emb_dst], dim=-1)
        return self.link_predictor(cat_emb)
```

### Multi-Objective Optimization Score

$$	ext{FinalScore}(u, v) =  lpha \cdot \hat{y}_{	ext{compat}}(u, v) +  eta \cdot \log(1 + 	ext{Stock}_v) - \gamma \cdot |	ext{Tier}_u - 	ext{Tier}_v| - \delta \cdot 	ext{LeadTimePenalty}_v$$

Where:
* $\hat{y}_{	ext{compat}}$: GNN-predicted compatibility probability.
* $	ext{Stock}_v$: Available warehouse quantity of candidate component $v$.
* $|	ext{Tier}_u - 	ext{Tier}_v|$: Penalty for mixing disparate component tiers (e.g., pairing a high-end carbon frame with entry-level mechanical disc brakes).
* $	ext{LeadTimePenalty}_v$: Extra penalty if part procurement type has long replenishment cycles.

---

## 6. Implementation Pipeline & Execution Flow

1. **Extract Candidate Subgraph (Cypher):**
   Query Neo4j for all available parts with matching physical interfaces and `stock > 0`.
2. **Construct PyG Subgraph:**
   Convert retrieved nodes and active edges into a local `HeteroData` structure.
3. **Inference & Assembly Search:**
   * Compute node embeddings via GNN.
   * Greedily or via Beam Search, assemble complete candidate trees rooted at the targeted frame.
   * Rank complete BOMs using the Multi-Objective scoring function.
4. **Validation & ERP Sync:**
   * Return top BOM configurations to the product manager / engineering team.
   * Lock stock allocation upon approval.

---

## 7. Next Steps for Deep Dive
1. **Canonical Schema Normalizer:** Write an ETL script to map raw form text fields to standardized interface enums.
2. **Historical Data Extraction:** Export past successful production BOMs to serve as positive training pairs for link prediction.
3. **Prototype Environment:** Set up a local Neo4j instance with test warehouse stock and a PyG inference script.
