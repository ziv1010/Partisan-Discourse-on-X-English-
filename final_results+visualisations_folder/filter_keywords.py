"""
Script to filter out rows where keyword is 'ucc' or 'ratetvdebate'
from combined_stance_results.csv
Uses only built-in csv module (no pandas required)
"""

import csv

# Define paths
input_path = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/old_copy/combined_stance_results.csv'
output_path = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv'

# Keywords to filter out (case-insensitive)
keywords_to_remove = ['ucc', 'ratetvdebate']

print(f"Reading from: {input_path}")

original_count = 0
filtered_count = 0

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        original_count += 1
        keyword = row.get('keyword', '').lower()
        
        if keyword not in keywords_to_remove:
            writer.writerow(row)
            filtered_count += 1

print(f"Original row count: {original_count}")
print(f"Filtered row count: {filtered_count}")
print(f"Rows removed: {original_count - filtered_count}")
print(f"Saved filtered CSV to: {output_path}")
