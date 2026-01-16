#!/usr/bin/env python3
"""
Script to analyze the partisan skew (pro-ruling vs pro-opposition ratio) 
of the 35 selected aspects to justify their selection criteria.

Calculates various ratio metrics to find a threshold that explains
why these aspects were chosen over others in the same percentile range.
"""

import pandas as pd
import os

# All 35 aspects (15 seed + 20 extended)
SEED_ASPECTS = [
    'caa', 'congress', 'farm_laws', 'farmers_protests',
    'hindu', 'hindutva', 'kashmir', 'kashmiri_pandits',
    'modi', 'muslim', 'new_parliament', 'rahulgandhi',
    'ram_mandir', 'shaheen_bagh', 'china'
]

EXTENDED_ASPECTS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'unemployment',
    'sangh', 'sharia', 'spyware', 'suicides'
]

TARGET_KEYWORDS = SEED_ASPECTS + EXTENDED_ASPECTS

def main():
    # Load the generated table data
    table_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_data_for_latex.csv")
    df = pd.read_csv(table_csv)
    
    print("=" * 80)
    print("PARTISAN SKEW ANALYSIS FOR ASPECT SELECTION JUSTIFICATION")
    print("=" * 80)
    
    # Calculate different ratio metrics
    results = []
    
    for _, row in df.iterrows():
        pro_ruling = row['Pro_Ruling_Count']
        pro_opp = row['Pro_Opp_Count']
        total = row['Total_Tweets']
        
        # Calculate various ratio metrics
        # 1. Simple ratio (larger/smaller)
        if min(pro_ruling, pro_opp) > 0:
            simple_ratio = max(pro_ruling, pro_opp) / min(pro_ruling, pro_opp)
        else:
            simple_ratio = float('inf')
        
        # 2. Percentage skew (how far from 50-50)
        pct_ruling = (pro_ruling / total * 100) if total > 0 else 0
        pct_opp = (pro_opp / total * 100) if total > 0 else 0
        skew_from_50 = abs(pct_ruling - pct_opp)  # 0 = perfectly balanced, 100 = completely one-sided
        
        # 3. Dominance ratio (percentage of dominant side)
        dominance = max(pct_ruling, pct_opp)
        
        # 4. Which side dominates
        dominant_side = "Pro-Ruling" if pro_ruling > pro_opp else "Pro-Opp" if pro_opp > pro_ruling else "Balanced"
        
        results.append({
            'Aspect': row['Aspect'],
            'Total': total,
            'Pro_Ruling': pro_ruling,
            'Pro_Opp': pro_opp,
            'Simple_Ratio': round(simple_ratio, 2),
            'Skew_From_50': round(skew_from_50, 2),
            'Dominance_Pct': round(dominance, 2),
            'Dominant_Side': dominant_side,
            'Pct_Ruling': round(pct_ruling, 2),
            'Pct_Opp': round(pct_opp, 2)
        })
    
    results_df = pd.DataFrame(results)
    
    # Print results sorted by skew
    print("\n--- ASPECTS SORTED BY PARTISAN SKEW (most skewed first) ---\n")
    print(f"{'Aspect':<20} {'Total':>8} {'Pro-Rul':>8} {'Pro-Opp':>8} {'Ratio':>8} {'Skew%':>8} {'Dominant':>12}")
    print("-" * 80)
    
    for _, row in results_df.sort_values('Skew_From_50', ascending=False).iterrows():
        ratio_str = f"{row['Simple_Ratio']:.1f}x" if row['Simple_Ratio'] != float('inf') else "∞"
        print(f"{row['Aspect']:<20} {row['Total']:>8,} {row['Pro_Ruling']:>8,} {row['Pro_Opp']:>8,} {ratio_str:>8} {row['Skew_From_50']:>7.1f}% {row['Dominant_Side']:>12}")
    
    # Calculate thresholds
    print("\n" + "=" * 80)
    print("MINIMUM THRESHOLD ANALYSIS")
    print("=" * 80)
    
    min_skew = results_df['Skew_From_50'].min()
    min_ratio = results_df[results_df['Simple_Ratio'] != float('inf')]['Simple_Ratio'].min()
    
    # Find the aspect with minimum skew
    min_skew_aspect = results_df[results_df['Skew_From_50'] == min_skew].iloc[0]['Aspect']
    min_ratio_aspect = results_df[results_df['Simple_Ratio'] == min_ratio].iloc[0]['Aspect']
    
    print(f"\nMinimum Partisan Skew: {min_skew:.2f}% (Aspect: {min_skew_aspect})")
    print(f"Minimum Ratio (max/min): {min_ratio:.2f}x (Aspect: {min_ratio_aspect})")
    
    # Suggest thresholds
    print("\n--- SUGGESTED SELECTION CRITERIA ---")
    print(f"\n1. SKEW THRESHOLD: All selected aspects have a partisan skew >= {min_skew:.1f}%")
    print(f"   (i.e., the difference between Pro-Ruling% and Pro-Opp% is at least {min_skew:.1f} percentage points)")
    
    print(f"\n2. RATIO THRESHOLD: All selected aspects have a ratio >= {min_ratio:.2f}x")
    print(f"   (i.e., the dominant side has at least {min_ratio:.2f}x more tweets than the other)")
    
    # Distribution of skew values
    print("\n--- SKEW DISTRIBUTION ---")
    skew_ranges = [
        (0, 10, "Low skew (0-10%)"),
        (10, 25, "Moderate skew (10-25%)"),
        (25, 50, "High skew (25-50%)"),
        (50, 100, "Very high skew (50%+)")
    ]
    
    for low, high, label in skew_ranges:
        count = len(results_df[(results_df['Skew_From_50'] >= low) & (results_df['Skew_From_50'] < high)])
        aspects = results_df[(results_df['Skew_From_50'] >= low) & (results_df['Skew_From_50'] < high)]['Aspect'].tolist()
        print(f"  {label}: {count} aspects")
        if aspects:
            print(f"    -> {', '.join(aspects)}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "partisan_skew_analysis.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
