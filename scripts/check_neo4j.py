
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.graphiti.setup_graphiti import get_graphiti_client

async def check_props_and_indexes():
    client = await get_graphiti_client()
    
    print("--- LABEL COUNTS ---")
    results, _, _ = await client.driver.execute_query("MATCH (n) RETURN labels(n) as labels, count(n) as count")
    for record in results:
        print(f"Labels: {record.get('labels')}, Count: {record.get('count')}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(check_props_and_indexes())
