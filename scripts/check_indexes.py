
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.graphiti.setup_graphiti import get_graphiti_client

async def check_indexes():
    client = await get_graphiti_client()
    # Check default database
    results, _, _ = await client.driver.execute_query("SHOW INDEXES")
    for record in results:
        print(f"Index: {record.get('name')}, Type: {record.get('type')}, Labels: {record.get('labelsOrTypes')}, Properties: {record.get('properties')}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(check_indexes())
