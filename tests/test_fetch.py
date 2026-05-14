import os
import sys
from pathlib import Path
from neo4j import GraphDatabase

# Repo root is the parent of src/ (NLP-LegalQA/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm.eval_voter import fetch_law_texts

def test_fetch_article():
    uri = os.environ.get("NEO4J_URI", "neo4j+ssc://nguyenhoangquan.com:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "Neoneo4j")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password), database=database)
    
    uid = "100/2019/NĐ-CP::article::5::clause::3::point::a"
    print(f"--- Testing fetch_law_texts for UID: {uid} ---")
    
    try:
        uid_to_text, nodes, merged_text = fetch_law_texts(driver, [uid])
        
        print("\n[ASSEMBLED LAW TEXT (MERGED)]")
        print("--------------------")
        print(merged_text)
        print("--------------------")

        print("\n[PER-UID LAW TEXT]")
        text = uid_to_text.get(uid, "NOT FOUND")
        print("--------------------")
        print(text)
        print("--------------------")
        
        print(f"\nNodes fetched: {len(nodes)}")
        for u, n in nodes.items():
            print(f"  - {u} (label: {n.get('label')})")
            
    except Exception as e:
        print(f"Error during fetch: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_fetch_article()
