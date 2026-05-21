import pandas as pd
import os

def main():
    base_dir = "qa_dataset"
    parts = [2, 3, 4, 5]
    part_str = "".join(map(str, parts))
    
    # 1. Merge QA datasets
    dfs = []
    for p in parts:
        file_path = os.path.join(base_dir, f"QA_Part{p}.csv")
        print(f"Reading {file_path}...")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            raise FileNotFoundError(f"Missing expected dataset file: {file_path}")
            
    output_file = os.path.join(base_dir, f"QA_Part{part_str}.csv")
    
    print("Merging datasets...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print("Reassigning IDs...")
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    print(f"Saving merged dataset to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged {len(merged_df)} rows.")

    # 2. Merge row results
    eval_subpath = "eval_results_pipeline_all_ablations_geminiflash_rerank_top_30"
    part_eval_dirs = {p: os.path.join("eval_results", f"QA_Part{p}", eval_subpath) for p in parts}
    output_eval_dir = os.path.join("eval_results", f"QA_Part{part_str}", eval_subpath)
    
    reference_part = parts[0]
    ref_dir = part_eval_dirs[reference_part]
    
    if os.path.exists(ref_dir):
        print(f"\nScanning for row_results in {ref_dir}...")
        os.makedirs(output_eval_dir, exist_ok=True)
        files = [f for f in os.listdir(ref_dir) if f.startswith("row_results_") and f.endswith(".csv")]
        
        for filename in files:
            # Check if this file exists in all part directories
            all_exist = True
            rdfs = []
            for p in parts:
                file_p = os.path.join(part_eval_dirs[p], filename)
                if os.path.exists(file_p):
                    rdf = pd.read_csv(file_p)
                    rdfs.append(rdf)
                else:
                    all_exist = False
                    print(f"Warning: {filename} is missing in QA_Part{p} evaluation directory.")
                    break
            
            if all_exist:
                print(f"Merging row results for {filename}...")
                merged_rdf = pd.concat(rdfs, ignore_index=True)
                merged_rdf['id'] = range(1, len(merged_rdf) + 1)
                
                out_file = os.path.join(output_eval_dir, filename)
                merged_rdf.to_csv(out_file, index=False)
                print(f"Successfully saved merged row results to {out_file} ({len(merged_rdf)} rows).")
            else:
                print(f"Warning: {filename} was not found in all part directories. Skipping.")
    else:
        print(f"Directory {ref_dir} does not exist. Skipping row results merge.")

if __name__ == "__main__":
    main()
