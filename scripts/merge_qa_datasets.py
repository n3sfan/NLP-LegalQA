import pandas as pd
import os

def main():
    base_dir = "/home/nhquan/NLP-LegalQA/qa_dataset"
    
    file2 = os.path.join(base_dir, "QA_Part2.csv")
    file3 = os.path.join(base_dir, "QA_Part3.csv")
    file4 = os.path.join(base_dir, "QA_Part4.csv")
    
    output_file = os.path.join(base_dir, "QA_Part234.csv")
    
    print(f"Reading {file2}...")
    df2 = pd.read_csv(file2)
    
    print(f"Reading {file3}...")
    df3 = pd.read_csv(file3)
    
    print(f"Reading {file4}...")
    df4 = pd.read_csv(file4)
    
    print("Merging datasets...")
    merged_df = pd.concat([df2, df3, df4], ignore_index=True)
    
    print("Reassigning IDs...")
    # Reassign ID starting from 1
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    print(f"Saving merged dataset to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged {len(merged_df)} rows.")

if __name__ == "__main__":
    main()
