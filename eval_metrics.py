import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime
import os

def setup_paths():
    """
    Set up all file paths for the analysis using the specified directory structure.
    
    Returns:
    dict: Dictionary containing all relevant paths
    """
    # Base directories
    base_dir = "/Users/Divya/LLM Project"
    
    # Input paths
    datasets_dir = os.path.join(base_dir, "datasets")
    batch_results_dir = os.path.join(base_dir, "batch_results")
    
    # Create metrics directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join(base_dir, "metrics", timestamp)
    os.makedirs(metrics_dir, exist_ok=True)
    
    paths = {
        'input': {
            'original': os.path.join(datasets_dir, "Impact_of_Remote_Work_on_Mental_Health.csv"),
            'error': os.path.join(datasets_dir, "errors_5.csv"),
            'cleaned_batches': [
                os.path.join(batch_results_dir, f"cleaned_data_batch_{i}.csv")
                for i in range(1, 6)
            ]
        },
        'output': {
            'pickle': os.path.join(metrics_dir, "cleaning_evaluation.pkl"),
            'batch_metrics': os.path.join(metrics_dir, "batch_metrics.csv"),
            'column_metrics': os.path.join(metrics_dir, "column_metrics.csv"),
            'aggregated_metrics': os.path.join(metrics_dir, "aggregated_metrics.csv")
        },
        'metrics_dir': metrics_dir
    }
    
    return paths

def load_and_validate_columns(file_path, is_cleaned_file=False):
    """
    Load CSV and validate that it contains the expected columns.
    For cleaned files, drop the additional columns before processing.
    
    Parameters:
    file_path (str): Path to the CSV file
    is_cleaned_file (bool): Whether this is a cleaned data file that needs column dropping
    
    Returns:
    pd.DataFrame: Loaded DataFrame
    """
    expected_columns = [
        'Employee_ID', 'Age', 'Gender', 'Job_Role', 'Industry',
        'Years_of_Experience', 'Work_Location', 'Hours_Worked_Per_Week',
        'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
        'Stress_Level', 'Mental_Health_Condition',
        'Access_to_Mental_Health_Resources', 'Productivity_Change',
        'Social_Isolation_Rating', 'Satisfaction_with_Remote_Work',
        'Company_Support_for_Remote_Work', 'Physical_Activity',
        'Sleep_Quality', 'Region'
    ]
    
    df = pd.read_csv(file_path)
    
    if is_cleaned_file:
        # Drop the additional columns if they exist
        columns_to_drop = ['cleaning_status', 'changes_made']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
    
    # Verify columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in {file_path}: {missing_cols}")
    
    # Ensure columns are in the same order
    df = df[expected_columns]
    
    return df

def process_all_batches(paths, batch_size=1000):
    """
    Process and evaluate all batches of data.
    
    Parameters:
    paths (dict): Dictionary containing all file paths
    batch_size (int): Size of each batch
    
    Returns:
    list: List of results dictionaries for each batch
    dict: Aggregated results across all batches
    """
    # Load full original and error datasets
    original_df = load_and_validate_columns(paths['input']['original'])
    error_df = load_and_validate_columns(paths['input']['error'])
    
    all_results = []
    
    # Process each batch file
    for batch_num, batch_path in enumerate(paths['input']['cleaned_batches'], 1):
        print(f"Processing batch {batch_num}...")
        
        # Load cleaned batch with column dropping
        cleaned_batch = load_and_validate_columns(batch_path, is_cleaned_file=True)
        
        # Calculate start and end indices for this batch
        start_idx = (batch_num - 1) * batch_size
        end_idx = start_idx + len(cleaned_batch)
        
        # Get corresponding rows from original and error datasets
        original_batch = original_df.iloc[start_idx:end_idx].reset_index(drop=True)
        error_batch = error_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Evaluate this batch
        batch_results = evaluate_cleaning_batch(
            original_batch,
            error_batch,
            cleaned_batch,
            batch_num
        )
        all_results.append(batch_results)
    
    # Aggregate results across all batches
    aggregated_results = aggregate_batch_results(all_results)
    
    return all_results, aggregated_results
   

def evaluate_cleaning_batch(original_df, error_df, cleaned_df, batch_number):
    """
    Evaluate the performance of data cleaning for a specific batch.
    
    Parameters:
    original_df (pd.DataFrame): The original clean dataset batch
    error_df (pd.DataFrame): Dataset batch with introduced errors
    cleaned_df (pd.DataFrame): Dataset batch after cleaning attempts
    batch_number (int): Current batch number
    
    Returns:
    dict: Dictionary containing all evaluation metrics for this batch
    """
    results = {
        'batch_number': batch_number,
        'error_metrics': {},
        'global_metrics': {},
        'classification_metrics': {},
        'column_metrics': {}
    }
    
    # Calculate metrics for each column
    for column in original_df.columns:
        original_col = original_df[column]
        error_col = error_df[column]
        cleaned_col = cleaned_df[column]
        
        # Column-specific metrics
        col_errors = (original_col != error_col).sum()
        col_corrections = sum((original_col != error_col) & (original_col == cleaned_col))
        col_unsuccessful = sum((original_col != error_col) & (original_col != cleaned_col))
        col_unchanged = sum((original_col != error_col) & (error_col == cleaned_col))
        
        results['column_metrics'][column] = {
            'errors': col_errors,
            'corrections': col_corrections,
            'unsuccessful': col_unsuccessful,
            'unchanged': col_unchanged
        }
    
    # Overall batch metrics
    original_vs_error = original_df != error_df
    original_vs_cleaned = original_df != cleaned_df
    error_vs_cleaned = error_df != cleaned_df
    
    # Error metrics
    total_errors = original_vs_error.sum().sum()
    errors_corrected = sum((original_vs_error & ~original_vs_cleaned).sum())
    unsuccessful_corrections = sum((original_vs_error & original_vs_cleaned).sum())
    unchanged_errors = sum((original_vs_error & ~error_vs_cleaned).sum())
    
    results['error_metrics'] = {
        'total_errors': total_errors,
        'errors_corrected': errors_corrected,
        'unsuccessful_corrections': unsuccessful_corrections,
        'unchanged_errors': unchanged_errors
    }
    
    # Global metrics
    total_cells = original_df.size
    non_erroneous_cells = total_cells - total_errors
    changed_non_erroneous = sum((~original_vs_error & error_vs_cleaned).sum())
    unchanged_non_erroneous = non_erroneous_cells - changed_non_erroneous
    
    results['global_metrics'] = {
        'total_non_erroneous_cells': non_erroneous_cells,
        'changed_non_erroneous_cells': changed_non_erroneous,
        'unchanged_non_erroneous_cells': unchanged_non_erroneous
    }
    
    # Classification metrics
    original_flat = original_df.values.flatten()
    error_flat = error_df.values.flatten()
    cleaned_flat = cleaned_df.values.flatten()
    
    y_true = (original_flat == error_flat).astype(int)
    y_pred = (original_flat == cleaned_flat).astype(int)
    
    results['classification_metrics'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return results

def process_all_batches(paths, batch_size=1000):
    """
    Process and evaluate all batches of data.
    
    Parameters:
    paths (dict): Dictionary containing all file paths
    batch_size (int): Size of each batch
    
    Returns:
    list: List of results dictionaries for each batch
    dict: Aggregated results across all batches
    """
    # Load full original and error datasets
    original_df = load_and_validate_columns(paths['input']['original'])
    error_df = load_and_validate_columns(paths['input']['error'])
    
    all_results = []
    
    # Process each batch file
    for batch_num, batch_path in enumerate(paths['input']['cleaned_batches'], 1):
        print(f"Processing batch {batch_num}...")
        
        # Load cleaned batch
        cleaned_batch = load_and_validate_columns(batch_path)
        
        # Calculate start and end indices for this batch
        start_idx = (batch_num - 1) * batch_size
        end_idx = start_idx + len(cleaned_batch)
        
        # Get corresponding rows from original and error datasets
        original_batch = original_df.iloc[start_idx:end_idx].reset_index(drop=True)
        error_batch = error_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Evaluate this batch
        batch_results = evaluate_cleaning_batch(
            original_batch,
            error_batch,
            cleaned_batch,
            batch_num
        )
        all_results.append(batch_results)
    
    # Aggregate results across all batches
    aggregated_results = aggregate_batch_results(all_results)
    
    return all_results, aggregated_results

def aggregate_batch_results(batch_results):
    """
    Aggregate results from all batches into overall metrics.
    
    Parameters:
    batch_results (list): List of results dictionaries from each batch
    
    Returns:
    dict: Aggregated metrics
    """
    aggregated = {
        'error_metrics': {
            'total_errors': 0,
            'errors_corrected': 0,
            'unsuccessful_corrections': 0,
            'unchanged_errors': 0
        },
        'global_metrics': {
            'total_non_erroneous_cells': 0,
            'changed_non_erroneous_cells': 0,
            'unchanged_non_erroneous_cells': 0
        },
        'classification_metrics': {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'column_metrics': {}
    }
    
    n_batches = len(batch_results)
    
    # Sum up error and global metrics
    for batch in batch_results:
        for metric_type in ['error_metrics', 'global_metrics']:
            for key in aggregated[metric_type]:
                aggregated[metric_type][key] += batch[metric_type][key]
        
        # Average classification metrics
        for key in aggregated['classification_metrics']:
            aggregated['classification_metrics'][key] += batch['classification_metrics'][key] / n_batches
        
        # Aggregate column metrics
        for column, metrics in batch['column_metrics'].items():
            if column not in aggregated['column_metrics']:
                aggregated['column_metrics'][column] = {
                    'errors': 0,
                    'corrections': 0,
                    'unsuccessful': 0,
                    'unchanged': 0
                }
            for key in metrics:
                aggregated['column_metrics'][column][key] += metrics[key]
    
    return aggregated

def print_evaluation_report(batch_results, aggregated_results):
    """
    Print a formatted report of the evaluation results.
    
    Parameters:
    batch_results (list): Results for each batch
    aggregated_results (dict): Aggregated results across all batches
    """
    print("=== Overall Data Cleaning Evaluation Report ===\n")
    
    print("Aggregate Error Metrics:")
    for key, value in aggregated_results['error_metrics'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print()
    
    print("Aggregate Global Metrics:")
    for key, value in aggregated_results['global_metrics'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print()
    
    print("Average Classification Metrics:")
    for key, value in aggregated_results['classification_metrics'].items():
        print(f"{key.title()}: {value:.4f}")
    print()
    
    print("Column-wise Error Analysis:")
    for column, metrics in aggregated_results['column_metrics'].items():
        print(f"\n{column}:")
        for key, value in metrics.items():
            print(f"  {key.title()}: {value}")
    
    print("\n=== Batch-wise Summary ===")
    for batch in batch_results:
        print(f"\nBatch {batch['batch_number']}:")
        print(f"Total Errors: {batch['error_metrics']['total_errors']}")
        print(f"Errors Corrected: {batch['error_metrics']['errors_corrected']}")
        print(f"F1 Score: {batch['classification_metrics']['f1']:.4f}")

def save_results(batch_results, aggregated_results, paths):
    """
    Save evaluation results to CSV, PKL files, and a text report in the specified directory.
    
    Parameters:
    batch_results (list): Results for each batch
    aggregated_results (dict): Aggregated results across all batches
    paths (dict): Dictionary containing all file paths
    """
    # Save pickle file
    with open(paths['output']['pickle'], 'wb') as f:
        pickle.dump({
            'batch_results': batch_results,
            'aggregated_results': aggregated_results
        }, f)
    
    # Save evaluation report to a text file
    report_path = os.path.join(paths['metrics_dir'], 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        # Redirect print statements to the file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print("=== Overall Data Cleaning Evaluation Report ===\n")
            
            print("Aggregate Error Metrics:")
            for key, value in aggregated_results['error_metrics'].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            print()
            
            print("Aggregate Global Metrics:")
            for key, value in aggregated_results['global_metrics'].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            print()
            
            print("Average Classification Metrics:")
            for key, value in aggregated_results['classification_metrics'].items():
                print(f"{key.title()}: {value:.4f}")
            print()
            
            print("Column-wise Error Analysis:")
            for column, metrics in aggregated_results['column_metrics'].items():
                print(f"\n{column}:")
                for key, value in metrics.items():
                    print(f"  {key.title()}: {value}")
            
            print("\n=== Batch-wise Summary ===")
            for batch in batch_results:
                print(f"\nBatch {batch['batch_number']}:")
                print(f"Total Errors: {batch['error_metrics']['total_errors']}")
                print(f"Errors Corrected: {batch['error_metrics']['errors_corrected']}")
                print(f"F1 Score: {batch['classification_metrics']['f1']:.4f}")
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    
    # Batch metrics
    batch_metrics = []
    for batch in batch_results:
        batch_row = {
            'batch_number': batch['batch_number'],
            **{f'error_{k}': v for k, v in batch['error_metrics'].items()},
            **{f'global_{k}': v for k, v in batch['global_metrics'].items()},
            **{f'classification_{k}': v for k, v in batch['classification_metrics'].items()}
        }
        # Add column metrics
        for col, metrics in batch['column_metrics'].items():
            for metric_name, value in metrics.items():
                batch_row[f'{col}_{metric_name}'] = value
        batch_metrics.append(batch_row)
    
    batch_df = pd.DataFrame(batch_metrics)
    batch_df.to_csv(paths['output']['batch_metrics'], index=False)
    
    # Column metrics
    column_metrics = []
    for column, metrics in aggregated_results['column_metrics'].items():
        metrics['column'] = column
        column_metrics.append(metrics)
    
    column_df = pd.DataFrame(column_metrics)
    column_df.to_csv(paths['output']['column_metrics'], index=False)
    
    # Aggregated metrics
    agg_metrics = {}
    for metric_type in ['error_metrics', 'global_metrics', 'classification_metrics']:
        for k, v in aggregated_results[metric_type].items():
            agg_metrics[f'{metric_type}_{k}'] = [v]
    
    agg_df = pd.DataFrame(agg_metrics)
    agg_df.to_csv(paths['output']['aggregated_metrics'], index=False)
    
    print(f"\nResults saved in directory: {paths['metrics_dir']}")
    print("Files created:")
    for output_type, path in paths['output'].items():
        print(f"- {output_type}: {os.path.basename(path)}")
    print(f"- report: evaluation_report.txt")

if __name__ == "__main__":
    # Set up paths
    paths = setup_paths()
    
    # Verify all files exist
    for path_type, path in paths['input'].items():
        if path_type == 'cleaned_batches':
            for batch_path in path:
                if not os.path.exists(batch_path):
                    raise FileNotFoundError(f"Batch file not found: {batch_path}")
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")
    
    # Process the data
    print("Starting evaluation...")
    batch_results, aggregated_results = process_all_batches(paths)
    
    # Print report
    print("\nGenerating evaluation report...")
    print_evaluation_report(batch_results, aggregated_results)
    
    # Save results
    print("\nSaving results...")
    save_results(batch_results, aggregated_results, paths)