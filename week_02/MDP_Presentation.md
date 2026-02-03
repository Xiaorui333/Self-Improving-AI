# Supply Chain Inventory Management MDP
## Two-Slide Presentation

---

## SLIDE 1: MDP Formulation

### **Problem Statement**
Optimize inventory ordering decisions for supply chain management using real-world data to minimize costs while meeting demand.

---

### **Formal MDP Components**

#### **1. State Space (S)**
\[ s \in S = \{0, 1, 2, \ldots, N\} \]

- **Definition**: Current inventory level at the beginning of a period
- **Source**: Extracted from `Inventory_Level` column in dataset
- **Capacity (N)**: Set to 1.2 × max observed inventory (provides buffer)
- **Example**: For a typical SKU, N ≈ 1188 units

#### **2. Action Space (A)**
\[ a \in A(s) = \{0, 1, 2, \ldots, N - s\} \]

- **Definition**: Order quantity (number of units to purchase)
- **Constraint**: Cannot exceed remaining capacity: \( s + a \leq N \)
- **Interpretation**: How many units to order given current inventory s

#### **3. Demand Distribution (φ)**
\[ \phi(d) = P(\text{Demand} = d) \]

- **Type**: **Empirical Distribution** (data-driven, not theoretical)
- **Source**: Constructed from `Units_Sold` column
- **Calculation**: 
  \[
  \phi(d) = \frac{\text{Count}(Units\_Sold = d)}{\text{Total Observations}}
  \]
- **Advantage**: Captures real demand patterns (seasonality, trends, anomalies)
- **Example**: For SKU_1, mean demand μ = 20.05, σ = 9.07, max = 59

#### **4. Transition Dynamics (P)**
\[ P(s' | s, a) = P(\text{next inventory} = s' | \text{current} = s, \text{order} = a) \]

**Transition Logic**:
```
After ordering a units:
  Total available = s + a
  After demand d is realized:
    Remaining inventory = max(0, s + a - d)
    Next state s' = min(N, remaining inventory)
```

**Probability of transitioning to s'**:
\[
P(s' | s, a) = \sum_{d: \min(N, \max(0, s+a-d)) = s'} \phi(d)
\]

- States are capped at capacity N
- Inventory cannot go negative (lost sales, not backorders)

#### **5. Reward Function (R)**
\[ R(s, a, d) = -c \cdot a - h \cdot s' - p \cdot \max(0, d - (s + a)) \]

**Cost Components**:

1. **Purchase Cost**: \( c \cdot a \)
   - \( c \) = Unit_Cost from dataset (avg: $12.20)
   - Incurred when placing order

2. **Holding Cost**: \( h \cdot s' \)
   - \( h = 0.01 \times c \) (1% of unit cost per period)
   - Cost of storing inventory: \( h \approx \$0.12 \) per unit
   - \( s' = \min(N, \max(0, s + a - d)) \) = end-of-period inventory

3. **Shortage/Penalty Cost**: \( p \cdot \max(0, d - (s + a)) \)
   - \( p = 3 \times \text{profit margin} \)
   - Profit margin = Unit_Price - Unit_Cost ≈ $6.06
   - \( p \approx \$18.18 \) per unit short
   - Represents lost sales + customer goodwill

**Expected Immediate Reward**:
\[
R(s, a) = \mathbb{E}_d[R(s, a, d)] = -c \cdot a - \sum_{d=0}^{\infty} \phi(d) \left[ h \cdot s' + p \cdot \max(0, d - (s + a)) \right]
\]

#### **6. Objective Function**
\[
V^*(s) = \max_{\pi} \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s, \pi \right]
\]

**Where**:
- \( V^*(s) \) = Optimal value function (expected discounted total reward from state s)
- \( \gamma = 0.9 \) = Discount factor (future costs weighted 90% of immediate costs)
- \( \pi: S \to A \) = Policy mapping states to actions
- **Goal**: Find optimal policy \( \pi^* \) that maximizes \( V^* \)

#### **7. Bellman Optimality Equation**
\[
V^*(s) = \max_{a \in A(s)} \left\{ R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^*(s') \right\}
\]

**Optimal Policy**:
\[
\pi^*(s) = \arg\max_{a \in A(s)} \left\{ R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^*(s') \right\}
\]

---

### **Solution Method: Policy Iteration**

**Algorithm**:
1. **Initialize**: \( \pi_0(s) = 0 \) for all s (order nothing)
2. **Repeat until convergence**:
   
   a. **Policy Evaluation**: Solve for \( V^{\pi_k} \)
   \[
   V^{\pi_k}(s) = R(s, \pi_k(s)) + \gamma \sum_{s'} P(s'|s, \pi_k(s)) V^{\pi_k}(s')
   \]
   
   b. **Policy Improvement**: Update policy
   \[
   \pi_{k+1}(s) = \arg\max_{a} \left\{ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi_k}(s') \right\}
   \]
   
   c. **Check**: If \( \pi_{k+1} = \pi_k \), stop (optimal policy found)

**Convergence**: Guaranteed to find optimal policy in finite iterations

---

## SLIDE 2: Experimental Results

### **Dataset Overview**
- **Source**: High-Dimensional Supply Chain Inventory Dataset
- **Size**: 91,250 observations across 15 features
- **SKUs**: 25 unique products analyzed
- **Time Period**: Daily data from 2024-01-01
- **Approach**: Separate MDP solved for each SKU_ID

---

### **Parameter Extraction (Example: SKU_1)**

| Parameter | Value | Source/Calculation |
|-----------|-------|-------------------|
| **State Space** | 0 to 1,188 | 1.2 × max(Inventory_Level) |
| **Demand μ** | 20.05 units | mean(Units_Sold) |
| **Demand σ** | 9.07 units | std(Units_Sold) |
| **Max Demand** | 59 units | max(Units_Sold) |
| **Purchase Cost (c)** | $12.20 | mean(Unit_Cost) |
| **Holding Cost (h)** | $0.122 | 1% × c |
| **Shortage Cost (p)** | $18.18 | 3 × (Unit_Price - Unit_Cost) |
| **Discount Factor (γ)** | 0.9 | Standard |

---

### **Optimal Policy Results**

#### **Overall Summary (25 SKUs)**

| Metric | Value |
|--------|-------|
| **SKUs Analyzed** | 25 |
| **Average Base Stock Level** | ~29 units |
| **Average Reorder Point** | 0 units |
| **Dominant Policy Type** | Base-Stock Policy (100%) |
| **Convergence** | All policies converged |

#### **Policy Type: Base-Stock Policy**
- **Definition**: Always order up to target level S
- **Rule**: If inventory < S, order (S - current inventory)
- **Characteristic**: Reorder point = 0 (always order when below target)

---

### **Detailed Results: Sample SKU Analysis**

#### **SKU_1: Base-Stock Policy**
```
Reorder Point (s): 0 units
Base Stock Level (S): 29 units
Policy: Order up to 29
```

**Sample Policy Rules**:
| Current Inventory | Order Quantity | Target Stock |
|-------------------|----------------|--------------|
| 0 | 29 | 29 |
| 5 | 24 | 29 |
| 10 | 19 | 29 |
| 15 | 14 | 29 |
| 20 | 9 | 29 |
| 25 | 4 | 29 |
| 29 | 0 | 29 |
| 30+ | 0 | (no order) |

**Economic Interpretation**:
- Base stock of 29 ≈ 1.45 × mean demand (20.05)
- Provides ~1.5 periods of safety stock
- Balances holding costs vs shortage penalties

---

### **Value Function Analysis**

Expected discounted cost from different starting inventory levels:

| Inventory Level | Expected Value (V) | Interpretation |
|----------------|-------------------|----------------|
| 0 | -$2,624.64 | Worst case: must order full amount |
| 29 (base) | -$2,270.74 | Optimal target level |
| 100 | -$1,961.43 | Better: already have stock |
| 297 | -$733.55 | Excess inventory (high holding costs) |
| 594 | -$611.68 | Very high holding costs |
| 1188 (max) | -$1,211.04 | Extreme holding costs |

**Key Insight**: Value function shows trade-off between shortage risk (low inventory) and holding costs (high inventory)

---

### **Per-SKU Policy Summary (Sample)**

| SKU | Observations | Demand (μ±σ) | Capacity | Purchase Cost | Reorder Point | Base Stock | Policy Type |
|-----|-------------|-------------|----------|---------------|---------------|------------|-------------|
| SKU_1 | 3,650 | 20.1±9.1 | 1,188 | $12.20 | 0 | 29 | Base-Stock (29) |
| SKU_2 | 3,650 | 19.8±8.9 | 1,176 | $12.15 | 0 | 28 | Base-Stock (28) |
| SKU_3 | 3,650 | 20.3±9.2 | 1,200 | $12.25 | 0 | 30 | Base-Stock (30) |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

### **Key Findings**

#### **1. Policy Structure**
- **All SKUs**: Exhibited Base-Stock policies
- **No (s,S) policies**: Suggests uniform demand patterns across SKUs
- **Simple implementation**: Easy to operationalize in practice

#### **2. Economic Insights**
- **Base stock ≈ 1.4-1.5 × mean demand**: Consistent safety stock multiple
- **Low holding costs** (1% of unit cost): Encourages higher inventory
- **High shortage costs** (3× profit margin): Penalizes stockouts heavily
- **Result**: Conservative policies that maintain buffer stock

#### **3. Demand Patterns**
- **Empirical distribution**: Captured real-world variability
- **Mean demand**: ~20 units across SKUs (relatively consistent)
- **Variability**: σ ≈ 9 units (CV ≈ 45%)

#### **4. Computational Performance**
- **Convergence**: Policy iteration converged for all SKUs
- **State space**: 1,000+ states per SKU
- **Scalability**: Successfully handled 25 independent MDPs

---

### **Comparison: Data-Driven vs. Theoretical**

| Aspect | Our Approach (Empirical) | Traditional (Poisson) |
|--------|-------------------------|----------------------|
| **Demand Model** | Empirical φ(d) from data | Poisson(λ) assumption |
| **Advantages** | Captures real patterns | Simple, well-studied |
| **Data Requirement** | Requires historical data | Only needs mean |
| **Flexibility** | Adapts to any distribution | Limited to Poisson shape |
| **Our Result** | Base stock = 29 | (would differ) |

---

### **Practical Implementation**

#### **Recommended Policy for SKU_1**:
```
IF current_inventory < 29 THEN
    order_quantity = 29 - current_inventory
ELSE
    order_quantity = 0
END IF
```

#### **Benefits**:
1. **Simple**: Single parameter (base stock level)
2. **Optimal**: Derived from MDP solution
3. **Data-driven**: Based on actual demand patterns
4. **Adaptive**: Can be updated as new data arrives

---

### **Validation & Sensitivity**

**Robustness Checks** (Future Work):
- Cross-validation: Train on historical data, test on hold-out period
- Sensitivity: Analyze impact of cost parameter changes (c, h, p)
- Lead time: Extend model to include supplier lead time
- Multi-echelon: Consider warehouse-to-store distribution

---

### **Conclusions**

1. **MDP Framework**: Successfully formulated inventory problem as finite-horizon MDP
2. **Data Integration**: Extracted MDP parameters from real supply chain dataset
3. **Solution Method**: Policy iteration efficiently computed optimal policies
4. **Policy Type**: Base-stock policies optimal for all 25 SKUs
5. **Practical Value**: Provides actionable ordering rules for operations

**Key Takeaway**: Data-driven MDP approach yields simple, optimal, implementable inventory policies that balance costs while meeting demand.

---

### **Technical Specifications**

**Implementation**:
- Language: Python 3.8+
- Libraries: NumPy, Pandas
- Algorithm: Policy Iteration (exact dynamic programming)
- Convergence Criterion: ||V_{k+1} - V_k||_∞ < 10^{-4}
- Max Iterations: 100 per evaluation step

**Code Structure**:
```python
class InventoryMDP:
    - State space: [0, N]
    - Action space: [0, N-s]
    - Transition: Empirical demand
    - Reward: Purchase + Holding + Shortage costs
    
def policy_iteration(mdp):
    - Policy Evaluation (iterative)
    - Policy Improvement (greedy)
    - Returns: optimal policy π*, value function V*
```

---

## **References & Data**

**Dataset**: High-Dimensional Supply Chain Inventory Dataset
- 91,250 observations, 15 features, 25 SKUs
- Features: Date, SKU_ID, Units_Sold, Inventory_Level, Unit_Cost, Unit_Price, etc.

**MDP Formulation**:
- States: Inventory levels
- Actions: Order quantities  
- Dynamics: Empirical demand distribution
- Costs: Purchase, holding, shortage

**Solution**: Policy Iteration (Bellman, 1957)
