import numpy as np
import pandas as pd

class InventoryMDP:
    def __init__(self, capacity=20, max_demand=10, c=2, h=1, p=10, gamma=0.9, demand_probs=None):
        self.N = capacity
        self.max_demand = max_demand
        self.states = np.arange(self.N + 1)
        self.actions = np.arange(self.N + 1)
        self.c = c  # Purchase cost
        self.h = h  # Holding cost
        self.p = p  # Penalty/Shortage cost
        self.gamma = gamma
        
        # Demand Distribution (phi) - can be empirical or Poisson
        if demand_probs is not None:
            self.demand_probs = demand_probs
        else:
            # Default: Poisson Demand Distribution
            self.mu = 3.0
            self.demand_probs = self._poisson_pmf(self.mu, max_demand * 2)

    def _poisson_pmf(self, mu, k_max):
        from math import exp, factorial
        return [exp(-mu) * (mu**k) / factorial(k) for k in range(k_max + 1)]

    def get_reward_and_transitions(self, s, a):
        expected_reward = -self.c * a
        transitions = np.zeros(self.N + 1)
        
        for d, prob in enumerate(self.demand_probs):
            next_inventory = max(0, s + a - d)
            next_state = min(self.N, next_inventory)
            
            # Costs
            holding = self.h * next_state
            shortage = self.p * max(0, d - (s + a))
            
            expected_reward -= prob * (holding + shortage)
            transitions[next_state] += prob
            
        return expected_reward, transitions

def load_supply_chain_dataset(csv_path):
    """
    Load the supply chain dataset from CSV file
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())
    
    return df

def extract_empirical_demand_distribution(df):
    """
    Extract empirical demand distribution from Units_Sold column
    φ(d) = P(Demand = d) = empirical probability from data
    """
    print("\n=== Extracting Empirical Demand Distribution ===")
    
    # Get demand data from Units_Sold
    demand_data = df['Units_Sold'].dropna()
    
    # Count frequency of each demand value
    demand_counts = demand_data.value_counts().sort_index()
    max_demand = int(demand_data.max())
    
    # Convert to probabilities
    total_observations = len(demand_data)
    demand_probs = np.zeros(max_demand + 1)
    
    for demand_value, count in demand_counts.items():
        demand_probs[int(demand_value)] = count / total_observations
    
    print(f"Observed demand range: 0 to {max_demand}")
    print(f"Total observations: {total_observations}")
    print(f"Mean demand: {demand_data.mean():.2f}")
    print(f"Std demand: {demand_data.std():.2f}")
    print(f"\nTop 10 most common demand values:")
    for demand_val in demand_counts.head(10).index:
        prob = demand_probs[int(demand_val)]
        print(f"  Demand = {int(demand_val):3d}: probability = {prob:.4f} ({int(prob*total_observations)} occurrences)")
    
    return demand_probs, max_demand

def extract_mdp_parameters_for_sku(df_sku, sku_id):
    """
    Extract MDP parameters from supply chain dataset for a specific SKU
    
    Mappings:
    - State (s): Inventory_Level
    - Demand Distribution (φ): Empirical distribution from Units_Sold
    - c (Purchase cost): Unit_Cost
    - h (Holding cost): ~1% of Unit_Cost per period
    - p (Shortage cost): 2x-5x the profit margin
    """
    print(f"\n{'='*70}")
    print(f"  SKU: {sku_id}")
    print(f"{'='*70}")
    
    params = {}
    
    # 1. State Space: Based on Inventory_Level
    inventory_data = df_sku['Inventory_Level'].dropna()
    params['capacity'] = int(np.ceil(inventory_data.max() * 1.2))
    
    # 2. Demand Distribution: Empirical from Units_Sold
    demand_data = df_sku['Units_Sold'].dropna()
    demand_counts = demand_data.value_counts().sort_index()
    max_demand = int(demand_data.max())
    
    total_observations = len(demand_data)
    demand_probs = np.zeros(max_demand + 1)
    
    for demand_value, count in demand_counts.items():
        demand_probs[int(demand_value)] = count / total_observations
    
    params['demand_probs'] = demand_probs
    params['max_demand'] = max_demand
    params['mean_demand'] = float(demand_data.mean())
    params['std_demand'] = float(demand_data.std())
    
    # 3. Purchase Cost (c): From Unit_Cost
    unit_cost_data = df_sku['Unit_Cost'].dropna()
    params['c'] = float(unit_cost_data.mean())
    
    # 4. Holding Cost (h): ~1% of Unit_Cost
    params['h'] = params['c'] * 0.01
    
    # 5. Shortage/Penalty Cost (p): 3x profit margin
    unit_price_data = df_sku['Unit_Price'].dropna()
    avg_price = float(unit_price_data.mean())
    profit_margin = avg_price - params['c']
    params['p'] = profit_margin * 3.0
    
    # 6. Discount Factor
    params['gamma'] = 0.9
    
    # Store summary info
    params['num_observations'] = len(df_sku)
    params['avg_price'] = avg_price
    params['profit_margin'] = profit_margin
    
    # Print summary
    print(f"  Observations: {params['num_observations']}")
    print(f"  Demand: μ={params['mean_demand']:.2f}, σ={params['std_demand']:.2f}, max={params['max_demand']}")
    print(f"  Capacity: {params['capacity']} (max inventory: {int(inventory_data.max())})")
    print(f"  Costs: c=${params['c']:.2f}, h=${params['h']:.4f}, p=${params['p']:.2f}")
    
    return params

def extract_mdp_parameters(df):
    """
    Extract MDP parameters from supply chain dataset (legacy function for all data)
    
    Mappings:
    - State (s): Inventory_Level
    - Demand Distribution (φ): Empirical distribution from Units_Sold
    - c (Purchase cost): Unit_Cost
    - h (Holding cost): ~1% of Unit_Cost per period
    - p (Shortage cost): 2x-5x the profit margin
    """
    print("\n" + "="*70)
    print("EXTRACTING MDP PARAMETERS FROM DATASET")
    print("="*70)
    
    params = {}
    
    # 1. State Space: Based on Inventory_Level
    print("\n1. STATE SPACE (from Inventory_Level):")
    inventory_data = df['Inventory_Level'].dropna()
    params['capacity'] = int(np.ceil(inventory_data.max() * 1.2))  # Add 20% buffer
    print(f"   Max observed inventory: {inventory_data.max():.0f}")
    print(f"   Capacity (with buffer): {params['capacity']}")
    
    # 2. Demand Distribution: Empirical from Units_Sold
    print("\n2. DEMAND DISTRIBUTION (φ from Units_Sold):")
    demand_probs, max_demand = extract_empirical_demand_distribution(df)
    params['demand_probs'] = demand_probs
    params['max_demand'] = max_demand
    
    # 3. Purchase Cost (c): From Unit_Cost
    print("\n3. PURCHASE COST (c from Unit_Cost):")
    unit_cost_data = df['Unit_Cost'].dropna()
    params['c'] = float(unit_cost_data.mean())
    print(f"   Average Unit Cost: ${params['c']:.2f}")
    print(f"   Min: ${unit_cost_data.min():.2f}, Max: ${unit_cost_data.max():.2f}")
    
    # 4. Holding Cost (h): ~1% of Unit_Cost
    print("\n4. HOLDING COST (h = 1% of Unit_Cost):")
    params['h'] = params['c'] * 0.01
    print(f"   Holding cost per unit per period: ${params['h']:.4f}")
    
    # 5. Shortage/Penalty Cost (p): 2x-5x profit margin
    print("\n5. SHORTAGE/PENALTY COST (p = 3x profit margin):")
    unit_price_data = df['Unit_Price'].dropna()
    avg_price = float(unit_price_data.mean())
    profit_margin = avg_price - params['c']
    params['p'] = profit_margin * 3.0  # Using 3x as middle ground
    print(f"   Average Unit Price: ${avg_price:.2f}")
    print(f"   Profit Margin: ${profit_margin:.2f}")
    print(f"   Shortage cost (3x margin): ${params['p']:.2f}")
    
    # 6. Discount Factor
    params['gamma'] = 0.9
    print(f"\n6. DISCOUNT FACTOR (γ): {params['gamma']}")
    
    print("\n" + "="*70)
    print("PARAMETER EXTRACTION COMPLETE")
    print("="*70)
    
    return params

def analyze_sku(df_sku, sku_id):
    """
    Run complete MDP analysis for a single SKU
    Returns the optimal policy and key metrics
    """
    # Extract parameters
    params = extract_mdp_parameters_for_sku(df_sku, sku_id)
    
    # Create MDP
    env = InventoryMDP(
        capacity=params['capacity'],
        max_demand=params['max_demand'],
        c=params['c'],
        h=params['h'],
        p=params['p'],
        gamma=params['gamma'],
        demand_probs=params['demand_probs']
    )
    
    # Run policy iteration
    print(f"  Running policy iteration...")
    optimal_policy, values = policy_iteration(env)
    
    # Extract key insights
    reorder_point = None
    for s in range(len(optimal_policy)):
        if optimal_policy[s] > 0:
            reorder_point = s
            break
    
    targets = [s + optimal_policy[s] for s in range(len(optimal_policy)) if optimal_policy[s] > 0]
    if targets:
        base_stock = max(set(targets), key=targets.count)
    else:
        base_stock = 0
    
    max_order = max(optimal_policy)
    avg_order = np.mean([a for a in optimal_policy if a > 0]) if any(optimal_policy > 0) else 0
    
    # Determine policy type
    if len(set(targets)) == 1 if targets else False:
        policy_type = f"Base-Stock ({base_stock})"
    elif reorder_point is not None:
        policy_type = f"(s,S) ({reorder_point}, {base_stock})"
    else:
        policy_type = "Custom"
    
    # Create result summary
    result = {
        'sku_id': sku_id,
        'num_observations': params['num_observations'],
        'mean_demand': params['mean_demand'],
        'std_demand': params['std_demand'],
        'max_demand': params['max_demand'],
        'capacity': params['capacity'],
        'purchase_cost': params['c'],
        'holding_cost': params['h'],
        'shortage_cost': params['p'],
        'reorder_point': reorder_point,
        'base_stock': base_stock,
        'max_order': max_order,
        'avg_order': avg_order,
        'policy_type': policy_type,
        'optimal_policy': optimal_policy,
        'values': values,
        'expected_value_at_0': values[0],
        'expected_value_at_base': values[base_stock] if base_stock <= len(values)-1 else values[-1]
    }
    
    print(f"  ✓ Reorder Point: {reorder_point}, Base Stock: {base_stock}")
    print(f"  ✓ Policy Type: {policy_type}")
    
    return result

def policy_iteration(mdp):
    # Initialize random policy and zero value function
    policy = np.zeros(len(mdp.states), dtype=int)
    V = np.zeros(len(mdp.states))
    
    while True:
        # 1. Policy Evaluation
        for _ in range(100):
            V_prev = V.copy()
            for s in mdp.states:
                a = policy[s]
                reward, transitions = mdp.get_reward_and_transitions(s, a)
                V[s] = reward + mdp.gamma * np.dot(transitions, V_prev)
            if np.max(np.abs(V - V_prev)) < 1e-4: break

        # 2. Policy Improvement
        policy_stable = True
        for s in mdp.states:
            old_action = policy[s]
            best_action = None
            best_value = -float('inf')
            
            # Check all possible order quantities
            for a in range(mdp.N - s + 1):
                reward, transitions = mdp.get_reward_and_transitions(s, a)
                val = reward + mdp.gamma * np.dot(transitions, V)
                if val > best_value:
                    best_value = val
                    best_action = a
            
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable: break
            
    return policy, V


if __name__ == "__main__":
    # ==================== MAIN EXECUTION ====================
    
    print("\n" + "="*70)
    print("SUPPLY CHAIN INVENTORY MDP - PER SKU ANALYSIS")
    print("="*70)
    
    # Step 1: Load the local CSV dataset
    csv_path = "supply_chain_dataset1.csv"
    
    try:
        df = load_supply_chain_dataset(csv_path)
        
        # Step 2: Group by SKU_ID
        print("\n" + "="*70)
        print("GROUPING DATA BY SKU")
        print("="*70)
        
        sku_groups = df.groupby('SKU_ID')
        num_skus = len(sku_groups)
        
        print(f"\nFound {num_skus} unique SKUs")
        print(f"SKU IDs: {sorted(df['SKU_ID'].unique())[:10]}{'...' if num_skus > 10 else ''}")
        
        # Step 3: Analyze each SKU
        print("\n" + "="*70)
        print("ANALYZING EACH SKU")
        print("="*70)
        
        all_results = []
        
        for sku_id, df_sku in sku_groups:
            result = analyze_sku(df_sku, sku_id)
            all_results.append(result)
        
        # Step 4: Display Summary Table
        print("\n" + "="*70)
        print("SUMMARY: OPTIMAL POLICIES FOR ALL SKUs")
        print("="*70)
        
        print("\n" + "-"*130)
        print(f"{'SKU':<10} {'Obs':<6} {'Demand':<15} {'Capacity':<10} {'Costs (c,h,p)':<20} {'Reorder':<8} {'Base':<6} {'Policy Type':<20}")
        print("-"*130)
        
        for r in all_results:
            demand_str = f"{r['mean_demand']:.1f}±{r['std_demand']:.1f}"
            cost_str = f"{r['purchase_cost']:.1f},{r['holding_cost']:.3f},{r['shortage_cost']:.1f}"
            reorder_str = str(r['reorder_point']) if r['reorder_point'] is not None else 'N/A'
            
            print(f"{r['sku_id']:<10} {r['num_observations']:<6} {demand_str:<15} {r['capacity']:<10} "
                  f"{cost_str:<20} {reorder_str:<8} {r['base_stock']:<6} {r['policy_type']:<20}")
        
        print("-"*130)
        
        # Additional analysis
        print("\n" + "="*70)
        print("AGGREGATE INSIGHTS")
        print("="*70)
        
        avg_base_stock = np.mean([r['base_stock'] for r in all_results])
        avg_reorder = np.mean([r['reorder_point'] for r in all_results if r['reorder_point'] is not None])
        
        policy_types = {}
        for r in all_results:
            policy_types[r['policy_type']] = policy_types.get(r['policy_type'], 0) + 1
        
        print(f"\nNumber of SKUs analyzed: {len(all_results)}")
        print(f"Average Base Stock Level: {avg_base_stock:.2f} units")
        print(f"Average Reorder Point: {avg_reorder:.2f} units")
        
        print(f"\nPolicy Type Distribution:")
        for ptype, count in sorted(policy_types.items(), key=lambda x: -x[1]):
            print(f"  {ptype}: {count} SKUs ({100*count/len(all_results):.1f}%)")
        
        # Save detailed results for specific SKUs (optional)
        print("\n" + "="*70)
        print("DETAILED POLICY FOR SAMPLE SKUs")
        print("="*70)
        
        # Show detailed policy for first 3 SKUs
        for i, result in enumerate(all_results[:3]):
            print(f"\n--- {result['sku_id']} ---")
            print(f"Policy Type: {result['policy_type']}")
            print(f"Reorder Point: {result['reorder_point']}, Base Stock: {result['base_stock']}")
            print(f"\nSample Policy Rules (first 30 states):")
            print(f"{'Inventory':<12} {'Order':<12} {'Target':<12}")
            print("-"*36)
            for s in range(min(30, len(result['optimal_policy']))):
                a = result['optimal_policy'][s]
                print(f"{s:<12} {a:<12} {s+a:<12}")
        
    except FileNotFoundError:
        print(f"\nError: Could not find '{csv_path}'")
        print("Please make sure the file exists in the current directory.")
        print("\nRunning with default parameters instead...")
        
        env = InventoryMDP()
        optimal_policy, values = policy_iteration(env)
        
        print("\nOptimal Inventory Policy (State: Order Quantity):")
        for s, a in enumerate(optimal_policy):
            print(f"Inventory {s:2d} -> Order {a:2d} (Target Stock: {s+a})")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()