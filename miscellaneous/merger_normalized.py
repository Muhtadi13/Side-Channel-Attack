"""
Dataset Merger Script

This script merges all dataset.json files from subfolders in the data directory,
performs validation checks, normalizes each dataset individually, reassigns website indices,
and adds metadata and source information.
"""

import json
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Any, Tuple

DATA_ROOT = "./data"
# DATA_ROOT = "./offline-2-datasets-security"

# Set to None for no limit
MAX_PER_WEBSITE_PER_SOURCE = None

# Set to True to balance samples across websites within each source (take min(a,b,c) from each website)
BALANCE_WITHIN_SOURCE = True

def load_json_file(filepath: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading {filepath}: {e}")

def normalize_dataset(dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Normalize trace_data within a dataset using Min-Max normalization.
    Returns the normalized dataset and normalization parameters.
    """
    if not dataset:
        return dataset, {"min_val": 0, "max_val": 1}
    
    all_traces = []
    for item in dataset:
        all_traces.extend(item['trace_data'])
    
    min_val = min(all_traces) if all_traces else 0
    max_val = max(all_traces) if all_traces else 1
    
    normalized_dataset = []
    for item in dataset:
        normalized_trace = [(x - min_val) / (max_val - min_val) if max_val != min_val else x for x in item['trace_data']]
        normalized_dataset.append({
            "website": item["website"],
            "trace_data": normalized_trace
        })
    
    return normalized_dataset, {"min_val": min_val, "max_val": max_val}

def validate_dataset_structure(data: List[Dict], source_folder: str) -> Tuple[bool, str]:
    """
    Validate the structure of a dataset.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, list):
        return False, f"Dataset in {source_folder} is not an array"
    
    if not data:
        return False, f"Dataset in {source_folder} is empty"
    
    trace_data_length = None
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {i} in {source_folder} is not a JSON object"
        
        # Check required fields
        if 'website' not in item:
            return False, f"Item {i} in {source_folder} missing 'website' field"
        
        if 'trace_data' not in item:
            return False, f"Item {i} in {source_folder} missing 'trace_data' field"
        
        # Check field types
        if not isinstance(item['website'], str):
            return False, f"Item {i} in {source_folder} has non-string 'website' field"
        
        if not isinstance(item['trace_data'], list):
            return False, f"Item {i} in {source_folder} has non-array 'trace_data' field"
        
        # Check if all trace_data elements are numbers
        for j, trace_val in enumerate(item['trace_data']):
            if not isinstance(trace_val, (int, float)):
                return False, f"Item {i} in {source_folder} has non-numeric value at trace_data[{j}]"
        
        # Check trace_data length consistency
        current_length = len(item['trace_data'])
        if trace_data_length is None:
            trace_data_length = current_length
        elif trace_data_length != current_length:
            return False, f"Item {i} in {source_folder} has trace_data length {current_length}, expected {trace_data_length}"
    
    return True, ""

def validate_cross_dataset_consistency(all_datasets: Dict[str, List[Dict]]) -> Tuple[bool, str]:
    """
    Validate that all datasets have the same trace_data length across all files.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_length = None
    
    for source_folder, data in all_datasets.items():
        if data:  # Skip empty datasets
            current_length = len(data[0]['trace_data'])
            if expected_length is None:
                expected_length = current_length
            elif expected_length != current_length:
                return False, f"Dataset {source_folder} has trace_data length {current_length}, but other datasets have length {expected_length}"
    
    return True, ""

def collect_unique_websites(all_datasets: Dict[str, List[Dict]]) -> Dict[str, int]:
    """
    Collect all unique websites and assign them new indices.
    
    Returns:
        Dictionary mapping website URL to new index
    """
    unique_websites = set()
    
    for data in all_datasets.values():
        for item in data:
            unique_websites.add(item['website'])
    
    # Sort websites for consistent ordering
    sorted_websites = sorted(unique_websites)
    
    # Create mapping from website to index
    website_to_index = {website: idx for idx, website in enumerate(sorted_websites)}
    
    return website_to_index

def limit_items_per_website_per_source(all_datasets: Dict[str, List[Dict]], max_per_website_per_source: int) -> Dict[str, List[Dict]]:
    """
    Limit the number of items per website per source by randomly sampling.
    
    Args:
        all_datasets: Dictionary mapping source folder to list of items
        max_per_website_per_source: Maximum number of items per website per source
        
    Returns:
        Dictionary with limited datasets
    """
    if max_per_website_per_source <= 0:
        return all_datasets
    
    limited_datasets = {}
    
    for source_folder, dataset in all_datasets.items():
        # Group items by website
        website_groups = defaultdict(list)
        for item in dataset:
            website_groups[item['website']].append(item)
        
        # Sample from each website group
        limited_items = []
        original_count = 0
        sampled_count = 0
        
        for website, items in website_groups.items():
            original_count += len(items)
            
            if len(items) <= max_per_website_per_source:
                # If we have fewer items than the limit, keep all
                limited_items.extend(items)
                sampled_count += len(items)
            else:
                # Randomly sample the required number
                sampled_items = random.sample(items, max_per_website_per_source)
                limited_items.extend(sampled_items)
                sampled_count += max_per_website_per_source
        
        limited_datasets[source_folder] = limited_items
        
        if original_count != sampled_count:
            print(f"  🎲 {source_folder}: Sampled {sampled_count} items from {original_count} (limit: {max_per_website_per_source} per website)")
    
    return limited_datasets

def balance_websites_within_source(all_datasets: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Balance samples across websites within each source by taking min(a,b,c) samples 
    from each website where a,b,c are the counts of samples for each website.
    
    Args:
        all_datasets: Dictionary mapping source folder to list of items
        
    Returns:
        Dictionary with balanced datasets
    """
    balanced_datasets = {}
    
    for source_folder, dataset in all_datasets.items():
        # Group items by website
        website_groups = defaultdict(list)
        for item in dataset:
            website_groups[item['website']].append(item)
        
        if not website_groups:
            balanced_datasets[source_folder] = []
            continue
        
        # Find the minimum count across all websites in this source
        website_counts = {website: len(items) for website, items in website_groups.items()}
        min_count = min(website_counts.values())
        
        # Sample min_count items from each website
        balanced_items = []
        original_total = sum(website_counts.values())
        
        for website, items in website_groups.items():
            if len(items) <= min_count:
                # If we have fewer or equal items than min_count, keep all
                balanced_items.extend(items)
            else:
                # Randomly sample min_count items
                sampled_items = random.sample(items, min_count)
                balanced_items.extend(sampled_items)
        
        balanced_datasets[source_folder] = balanced_items
        
        new_total = len(balanced_items)
        if original_total != new_total:
            print(f"  ⚖️  {source_folder}: Balanced to {min_count} samples per website ({len(website_groups)} websites)")
            print(f"      Original: {original_total} items → Balanced: {new_total} items")
            for website, count in website_counts.items():
                print(f"        {website}: {count} → {min_count}")
    
    return balanced_datasets

def merge_datasets(data_folder: str = "data", max_per_website_per_source: int = None, balance_within_source: bool = False) -> None:
    """
    Main function to merge all datasets.
    
    Args:
        data_folder: Path to the data folder containing subfolders with datasets
        max_per_website_per_source: Maximum number of items per website per source (None for no limit)
        balance_within_source: Whether to balance samples across websites within each source
    """
    print("🚀 Dataset Merger Starting...")
    
    if max_per_website_per_source is not None:
        print(f"🎯 Max items per website per source: {max_per_website_per_source}")
    
    if balance_within_source:
        print("⚖️  Balancing enabled: Will take min(a,b,c) samples from each website within each source")
    
    if max_per_website_per_source is not None or balance_within_source:
        # Set random seed for reproducible sampling
        random.seed(42)
    
    print("🔍 Scanning data folder...")
    
    if not os.path.exists(data_folder):
        print(f"\n\n❌❌❌ Error: Data folder '{data_folder}' not found\n\n")
        sys.exit(1)
    
    # Find all subfolders
    subfolders = [item for item in os.listdir(data_folder) 
                  if os.path.isdir(os.path.join(data_folder, item))]
    
    if not subfolders:
        print(f"\n\n❌❌❌ Error: No subfolders found in '{data_folder}'\n\n")
        sys.exit(1)
    
    print(f"📁 Found {len(subfolders)} subfolders: {', '.join(subfolders)}")
    
    all_datasets = {}
    all_metadata = {}
    
    # Load, validate, and normalize each dataset
    print("\n📊 Loading, validating, and normalizing datasets...")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder, subfolder)
        dataset_path = os.path.join(subfolder_path, "dataset.json")
        metadata_path = os.path.join(subfolder_path, "metadata.json")
        
        print(f"  Processing {subfolder}...")
        
        # Check if dataset.json exists (required)
        if not os.path.exists(dataset_path):
            print(f"\n\n❌❌❌ Error: dataset.json not found in {subfolder}\n\n")
            sys.exit(1)
        
        # Check if metadata.json exists (optional)
        metadata_exists = os.path.exists(metadata_path)
        if not metadata_exists:
            print(f"    ⚠️  metadata.json not found in {subfolder}, using empty metadata")
        
        try:
            # Load dataset
            dataset = load_json_file(dataset_path)
            
            # Load metadata (optional)
            if metadata_exists:
                metadata = load_json_file(metadata_path)
            else:
                metadata = {}  # Use empty object if metadata.json doesn't exist
            
            # Validate dataset structure
            is_valid, error_msg = validate_dataset_structure(dataset, subfolder)
            if not is_valid:
                print(f"\n\n❌❌❌ Validation Error: {error_msg}\n\n")
                sys.exit(1)
            
            # Normalize the dataset
            normalized_dataset, norm_params = normalize_dataset(dataset)
            all_datasets[subfolder] = normalized_dataset
            
            # Update metadata with normalization parameters
            if metadata_exists:
                metadata.update({"normalization": norm_params})
            else:
                metadata = {"normalization": norm_params}
            all_metadata[subfolder] = metadata
            
            print(f"    ✅ {len(normalized_dataset)} items loaded and normalized successfully")
            
        except Exception as e:
            print(f"\n\n❌❌❌ Error processing {subfolder}: {e}\n\n")
            sys.exit(1)
    
    # Apply balancing within each source if enabled
    if balance_within_source:
        print(f"\n⚖️  Balancing websites within each source...")
        all_datasets = balance_websites_within_source(all_datasets)
    
    # Apply sampling limit if specified
    if max_per_website_per_source is not None:
        print(f"\n🎲 Applying sampling limit ({max_per_website_per_source} per website per source)...")
        all_datasets = limit_items_per_website_per_source(all_datasets, max_per_website_per_source)
    
    # Validate cross-dataset consistency
    print("\n🔍 Validating cross-dataset consistency...")
    is_valid, error_msg = validate_cross_dataset_consistency(all_datasets)
    if not is_valid:
        print(f"\n\n❌❌❌ Consistency Error: {error_msg}\n\n")
        sys.exit(1)
    
    print("✅ All datasets are consistent")
    
    # Collect unique websites and create new mapping
    print("\n🌐 Collecting unique websites...")
    website_to_index = collect_unique_websites(all_datasets)
    
    print(f"Found {len(website_to_index)} unique websites:")
    for website, idx in sorted(website_to_index.items(), key=lambda x: x[1]):
        print(f"  {idx}: {website}")
    
    # Merge all datasets
    print("\n🔄 Merging datasets...")
    merged_dataset = []
    
    for source_folder, dataset in all_datasets.items():
        metadata = all_metadata[source_folder]
        
        for item in dataset:
            # Create merged item
            merged_item = {
                "website": item["website"],
                "website_index": website_to_index[item["website"]],
                "trace_data": item["trace_data"],
                "metadata": metadata,
                "source": source_folder
            }
            
            merged_dataset.append(merged_item)
    
    print(f"✅ Merged {len(merged_dataset)} total items from {len(all_datasets)} sources")
    
    # Write merged dataset
    output_path = "dataset.json"
    print(f"\n💾 Writing merged dataset to {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully created {output_path}")
        print(f"📈 Final dataset contains {len(merged_dataset)} items")
        print(f"🌐 Covering {len(website_to_index)} unique websites")
        
        # Print summary statistics
        source_counts = defaultdict(int)
        website_counts = defaultdict(int)
        
        for item in merged_dataset:
            source_counts[item['source']] += 1
            website_counts[item['website']] += 1
        
        print("\n📊 Summary by source:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} items")
        
        print("\n🌐 Summary by website:")
        for website, count in sorted(website_counts.items(), key=lambda x: website_to_index[x[0]]):
            website_idx = website_to_index[website]
            print(f"  [{website_idx}] {website}: {count} items")
            
    except Exception as e:
        print(f"\n\n❌❌❌ Error writing output file: {e}\n\n")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Dataset Merger Starting...")
    merge_datasets(data_folder=DATA_ROOT, max_per_website_per_source=MAX_PER_WEBSITE_PER_SOURCE, balance_within_source=BALANCE_WITHIN_SOURCE)
    print("\n🎉 Dataset merger completed successfully!")
