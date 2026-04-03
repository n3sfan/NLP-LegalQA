
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.graphiti.setup_graphiti import get_graphiti_client

async def vector_search(query_text):
    client = await get_graphiti_client()
    
    print(f"Generating embedding for: '{query_text}'...")
    # Get embedding from the PhoBERTEmbedder configured in setup_graphiti
    vector = await client.embedder.create(query_text)
    
    # Ensure index exists (optional but safe)
    print("Ensuring vector index exists...")
    create_index_query = """
    CREATE VECTOR INDEX entity_index IF NOT EXISTS
    FOR (n:Entity) ON (n.name_embedding)
    OPTIONS {indexConfig: {
     `vector.dimensions`: 768,
     `vector.similarity_function`: 'cosine'
    }}
    """
    await client.driver.execute_query(create_index_query)
    
    # Perform search
    print("Performing vector search...")
    search_query = """
    CALL db.index.vector.queryNodes('entity_index', 5, $vector)
    YIELD node, score
    RETURN node.name AS name, node.content AS content, score
    ORDER BY score DESC
    """
    
    results, _, _ = await client.driver.execute_query(search_query, vector=vector)
    
    print("\nSearch Results:")
    for record in results:
        print(f"[{record.get('score'):.4f}] {record.get('name')}")
        print(f"Content: {record.get('content')[:200]}...")
        print("-" * 40)

    await client.close()

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Cơ quan quản lý đường bộ là ai?"
    asyncio.run(vector_search(query))
