import pandas as pd
import os
import glob
import numpy as np

def combine_and_average():
    base_dir = "eval_results"
    parts = ["QA_Part2", "QA_Part3", "QA_Part4", "QA_Part5"]
    out_dir = os.path.join(base_dir, "combined_results")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. basic-rag and basic-reranker
    for name, out_name in [("eval_results_basic_rag/metrics_summary.csv", "metrics_summary.csv"),
                           ("eval_results_basic_reranker/metrics_summary_reranker.csv", "metrics_summary_reranker.csv")]:
        dfs = []
        for part in parts:
            fpath = os.path.join(base_dir, part, name)
            if os.path.exists(fpath):
                dfs.append(pd.read_csv(fpath))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            mean_df = combined.mean().to_frame().T
            mean_df.to_csv(os.path.join(out_dir, out_name), index=False)
            print(f"Saved {out_name}")

    # 2. LLM results
    llm_files = [
        "eval_finetuned_results.csv",
        "geminiflash_zeroshot.csv", 
        "geminiflash_fewshot.csv", 
        "gemma4_zeroshot.csv", 
        "gemma4_fewshot.csv"
    ]
    
    llm_summary_dfs = []
    
    for llm_file in llm_files:
        dfs = []
        for part in parts:
            fpath = os.path.join(base_dir, part, "eval_results_llm", llm_file)
            if os.path.exists(fpath):
                dfs.append(pd.read_csv(fpath))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Find numeric columns
            numeric_cols = combined.select_dtypes(include=[np.number]).columns
            # Exclude 'id' if it's there
            if 'id' in numeric_cols:
                numeric_cols = numeric_cols.drop('id')
            mean_df = combined[numeric_cols].mean().to_frame().T
            
            # Add a 'model_config' column so we can combine them into a single summary
            mean_df.insert(0, 'model_config', llm_file.replace('.csv', ''))
            
            mean_df.to_csv(os.path.join(out_dir, f"averaged_{llm_file}"), index=False)
            llm_summary_dfs.append(mean_df)
            print(f"Saved averaged_{llm_file}")
            
    if llm_summary_dfs:
        combined_llm_summary = pd.concat(llm_summary_dfs, ignore_index=True)
        combined_llm_summary.to_csv(os.path.join(out_dir, "llm_metrics_summary.csv"), index=False)
        print(f"Saved llm_metrics_summary.csv")

    # 3. Ablation results
    for pipeline in ["geminiflash", "gemma4"]:
        dfs = []
        for part in parts:
            folder_name = f"eval_results_pipeline_all_ablations_{pipeline}_rerank_top_30"
            fpath = os.path.join(base_dir, part, folder_name, "ablation_comparison.csv")
            if os.path.exists(fpath):
                dfs.append(pd.read_csv(fpath))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
            
            agg_dict = {}
            for col in numeric_cols:
                if col == 'num_evaluated':
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'
                    
            mean_df = combined.groupby("config").agg(agg_dict).reset_index()
            out_file = f"ablation_comparison_{pipeline}.csv"
            mean_df.to_csv(os.path.join(out_dir, out_file), index=False)
            print(f"Saved {out_file}")

if __name__ == "__main__":
    combine_and_average()
