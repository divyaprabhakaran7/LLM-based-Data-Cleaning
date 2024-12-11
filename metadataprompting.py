import csv
import os
import time
import random
import logging
import anthropic
import pandas as pd
import numpy as np
from io import StringIO

def setup_logging(output_dir):
    """
    Set up logging configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, 'data_cleaning.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def split_dataframe_into_batches(df, batch_size=1000):
    """
    Split a DataFrame into batches of specified size
    """
    num_batches = (len(df) // batch_size) + (1 if len(df) % batch_size > 0 else 0)
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx].copy()
        batches.append(batch)
    
    return batches, num_batches

def truncate_dataframe(df, max_chars=20000):
    """
    Truncate the dataframe to fit within token limits
    """
    csv_content = df.to_csv(sep='\t', index=False)
    
    if len(csv_content) <= max_chars:
        return df
    
    chars_per_row = len(csv_content) / len(df)
    max_rows = int(max_chars / chars_per_row)
    
    return df.head(max_rows)

def clean_and_standardize_df(df, columns):
    """
    Clean and standardize DataFrame
    """
    for col in columns:
        if col not in df.columns:
            df[col] = ''
    
    df = df[columns]
    
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    
    return df

def process_sub_chunk(client, sub_chunk, original_columns, logger, max_retries=3):
    """
    Process a single sub-chunk of data
    """
    COLUMN_METADATA_PROMPT = """The description of the input data columns and their format is given below: 
Employee ID: A 7-character string indicating the employee's ID, starting with EMP following by four digits. Eg., EMP0001.
Age: Non-negative number value greater than 17 of the employee's age. Eg., 72.  
Gender: String data type of the employee's gender. Eligible options are Female, Male, or Non-binary. 
Job Role: String data type of the employee's job title. Eligible pptions are HR, Data Scientist, Designer, Marketing, Project Manager, Sales, or Software Engineer.
Industry: String data type of the industry employee's job is in. Eligible options are Consulting, Education, Finance, Healthcare, IT, Manufacturing, or Retail.
Years of Experience: Number of years of work experience an employee has, non-negative. For example, 4. 
Work Location: String data type of employee's job location. Eligible options are Remote, Hybrid, or Onsite.
Hours Worked Per Week: Non-negative number value of the number of hours the employee works per week. Cannot exceed the number of hours in a week. For example, 48.
Number of Virtual Meetings: Non-negative number of virtual meetings the employee takes in a week.
Work Life Balance Rating: Employee's self-reported work life balance rating, a number value from 1 through 5.
Stress Level: Employee's self-reported stress, a string data type. Eligible options are Low, Medium, or High.
Mental Health Condition: Employee's self-reported mental health conditions, a string data type. Eligible options are Depression, Anxiety, Burnout, or None.
Access to Mental Health Resources: Employee's rating of access to mental health resources. Eligible options are No or Yes. 
Productivity Change: Employee's self-reported change in productivity post-remote work, a string data type. Eligible options are Decrease, Increase, or No Change.
Social Isolation Rating: Employee's self-reported social isolation post-remote work, a number value from 1 through 5.
Satisfaction with Remote Work: Employee's self-reported satisfaction with remote work policy, a string data type. Eligible options are Satisfied, Unsatisfied, or Neutral.
Company Support for Remote Work: Employee's self-reported rating for their employer's support for remote work, a number value from 1 through 5. 
Physical Activity: Employee's rating for physical activity levels, a string data type. Eligible options are Daily, Weekly, or None.
Sleep Quality: Employee's self-reported sleep quality, a string data type. Eligible options are Average, Poor, or Good.
Region: Employee's region of employment, a string data type. Eligible options are Africa, Asia, Europe, North America, Oceania, or South America.
The columns in input data appear in the same order as given above."""

    expected_row_count = len(sub_chunk)
    
    for attempt in range(max_retries):
        try:
            truncated_chunk = truncate_dataframe(sub_chunk)
            
            if len(truncated_chunk) != len(sub_chunk):
                logger.warning(f"Sub-chunk was truncated from {len(sub_chunk)} to {len(truncated_chunk)} rows")
                return sub_chunk.assign(
                    cleaning_status='Original (Chunk Too Large)',
                    changes_made='No changes'
                ), 0
            
            csv_content = truncated_chunk.to_csv(sep='\t', index=False, header=False)
            
            prompt = f"""{COLUMN_METADATA_PROMPT}

You are a data cleaning expert. Your task is to fix errors in the following tabular data.

CRITICAL REQUIREMENTS:
1. The input data contains {expected_row_count} rows WITHOUT column headers
2. Output EXACTLY {expected_row_count} rows of cleaned data
3. DO NOT include column headers in the output
4. Each row must be separated by a single newline
5. Each row must contain values separated by tabs
6. Do not add any extra text or formatting

Input data ({expected_row_count} rows):
{csv_content}

Clean the data and output exactly {expected_row_count} rows. Do not include headers or any other text."""

            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip() if message.content else ""
            
            cleaned_lines = [
                line.strip() 
                for line in response_text.splitlines() 
                if line.strip() and '\t' in line
            ]
            
            if not cleaned_lines:
                logger.warning("Received empty response from Claude")
                continue

            if len(cleaned_lines) != expected_row_count:
                logger.warning(
                    f"Response has incorrect number of rows. Expected: {expected_row_count}, "
                    f"Got: {len(cleaned_lines)}"
                )
                continue

            cleaned_df = pd.read_csv(
                StringIO('\n'.join(cleaned_lines)),
                sep='\t',
                names=original_columns,
                header=None,
                dtype=str
            )

            cleaned_df = clean_and_standardize_df(cleaned_df, original_columns)
            standardized_chunk = clean_and_standardize_df(sub_chunk.copy(), original_columns)

            changes_by_row = []
            total_changes = 0
            rows_changed = 0

            for idx in range(expected_row_count):
                row_changes = []
                row_has_changes = False
                
                for col in original_columns:
                    original_val = standardized_chunk.iloc[idx][col]
                    cleaned_val = cleaned_df.iloc[idx][col]
                    
                    if original_val != cleaned_val and cleaned_val:
                        row_changes.append(col)
                        total_changes += 1
                        row_has_changes = True

                if row_has_changes:
                    rows_changed += 1
                
                changes_by_row.append(
                    'Changes in: ' + ', '.join(row_changes) if row_changes else 'No changes'
                )

            cleaned_df['cleaning_status'] = 'Cleaned'
            cleaned_df['changes_made'] = changes_by_row

            return cleaned_df, total_changes, rows_changed

        except Exception as e:
            logger.error(f"Error processing sub-chunk (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
            continue
    
    logger.error("Failed to process sub-chunk after multiple attempts")
    return sub_chunk.assign(
        cleaning_status='Original (Max Retries Exceeded)',
        changes_made='No changes'
    ), 0, 0

def process_batch(client, batch, batch_num, original_columns, logger, output_dir):
    """
    Process a batch with sub-chunking
    """
    SUB_CHUNK_SIZE = 30
    logger.info(f"Processing batch {batch_num} ({len(batch)} rows)")
    
    if len(batch) > SUB_CHUNK_SIZE:
        num_sub_chunks = (len(batch) // SUB_CHUNK_SIZE) + (1 if len(batch) % SUB_CHUNK_SIZE > 0 else 0)
        sub_chunks = np.array_split(batch, num_sub_chunks)
        
        logger.info(f"Batch {batch_num}: Split into {len(sub_chunks)} sub-chunks")
        
        processed_sub_chunks = []
        total_changes = 0
        total_rows_changed = 0
        
        for i, sub_chunk in enumerate(sub_chunks, 1):
            logger.info(f"Batch {batch_num}: Processing sub-chunk {i}/{len(sub_chunks)} ({len(sub_chunk)} rows)")
            
            processed_sub_chunk, changes, rows_changed = process_sub_chunk(
                client,
                sub_chunk.reset_index(drop=True),
                original_columns,
                logger
            )
            
            processed_sub_chunks.append(processed_sub_chunk)
            total_changes += changes
            total_rows_changed += rows_changed
            
            time.sleep(random.uniform(1, 2))
        
        processed_batch = pd.concat(processed_sub_chunks, ignore_index=True)
        
    else:
        processed_batch, total_changes, total_rows_changed = process_sub_chunk(
            client, batch, original_columns, logger
        )
    
    if len(processed_batch) != len(batch):
        raise ValueError(
            f"Batch {batch_num}: Processed size ({len(processed_batch)}) "
            f"!= Input size ({len(batch)})"
        )
    
    batch_output_path = os.path.join(output_dir, f'cleaned_data_batch_{batch_num}_30perc_errors.csv')
    processed_batch.to_csv(batch_output_path, index=False, encoding='utf-8')
    
    logger.info(
        f"Batch {batch_num}: Processed {len(batch)} rows with {total_changes} changes "
        f"affecting {total_rows_changed} rows"
    )
    
    return processed_batch, total_changes, total_rows_changed

def main():
    client = anthropic.Anthropic(
        api_key="<<your API Key>>"
    )

    base_dir = ""
    input_csv_path = os.path.join(base_dir, "datasets", "errors_30percent.csv")
    output_dir = os.path.join(base_dir, "batch_results")
    
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    try:
        # Read input file
        logger.info(f"Reading input file: {input_csv_path}")
        full_df = pd.read_csv(input_csv_path)
        total_rows = len(full_df)
        original_columns = list(full_df.columns)
        logger.info(f"Total rows in input file: {total_rows}")
        
        # Split into 1000-row batches
        batches, num_batches = split_dataframe_into_batches(full_df, batch_size=1000)
        logger.info(f"Split data into {num_batches} batches")
        
        # Process batches
        all_processed_batches = []
        total_statistics = {
            'total_rows_processed': 0,
            'total_changes_made': 0,
            'total_rows_changed': 0
        }
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Starting batch {batch_num}/{num_batches}")
            
            try:
                processed_batch, changes, rows_changed = process_batch(
                    client, batch, batch_num, original_columns, logger, output_dir
                )
                
                all_processed_batches.append(processed_batch)
                total_statistics['total_rows_processed'] += len(batch)
                total_statistics['total_changes_made'] += changes
                total_statistics['total_rows_changed'] += rows_changed
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add original batch with error status
                error_batch = batch.assign(
                    cleaning_status=f'Original (Error: {str(e)[:100]}...)',
                    changes_made='No changes'
                )
                all_processed_batches.append(error_batch)
                total_statistics['total_rows_processed'] += len(batch)
            
            time.sleep(random.uniform(1, 3))
        
        # Combine and verify results
        final_df = pd.concat(all_processed_batches, ignore_index=True)
        
        if len(final_df) != total_rows:
            raise ValueError(
                f"Row count mismatch: Final ({len(final_df)}) != Input ({total_rows})"
            )
        
        # Save combined results
        final_output_path = os.path.join(output_dir, 'cleaned_data_combined_30perc_errors.csv')
        final_df.to_csv(final_output_path, index=False, encoding='utf-8')
        
        # Calculate additional statistics
        total_statistics['original_rows'] = len(final_df[final_df['cleaning_status'].str.startswith('Original')])
        total_statistics['cleaned_rows'] = len(final_df[final_df['cleaning_status'] == 'Cleaned'])
        total_statistics['processing_success_rate'] = f"{(total_statistics['total_rows_changed']/total_statistics['total_rows_processed'])*100:.2f}%"
        
        # Save summary
        summary_path = os.path.join(output_dir, 'cleaning_summary.txt')
        with open(summary_path, 'w') as f:
            for key, value in total_statistics.items():
                f.write(f"{key}: {value}\n")
                logger.info(f"{key}: {value}")

        logger.info("Data cleaning completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during data cleaning: {e}")
        raise

if __name__ == "__main__":
    main()
