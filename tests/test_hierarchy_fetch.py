import os
from dotenv import load_dotenv

from legal_scraper.embedder import Neo4jEmbedder

def main():
    load_dotenv()
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    print(f"Connecting to {uri} with user {user} on db {database}")
    
    embedder = Neo4jEmbedder(uri, user, password, database)
    
    # We don't have a known uid immediately, so let's fetch a random Point or Clause to test
    with embedder._get_driver().session(database=database) as session:
        # Find 2 random uids that are deeply nested (e.g., Point)
        result = session.run("MATCH (n:Point) RETURN n.uid AS uid LIMIT 2")
        uids = [record["uid"] for record in result]
        
    print(f"Found random Point UIDs: {uids}")
    
    if uids:
        hierarchy_map = embedder.fetch_node_hierarchy(uids)
        for uid, text in hierarchy_map.items():
            print("="*60)
            print(f"UID: {uid}")
            print(f"TEXT:\n{text}")
            print("="*60)
    else:
        print("No points found in the database. Try clauses?")
        
    embedder.close()

if __name__ == "__main__":
    main()
