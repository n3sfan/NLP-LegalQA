from dotenv import load_dotenv
load_dotenv()
import os

from legal_scraper.embedder import Neo4jEmbedder

e = Neo4jEmbedder(
    uri="neo4j+ssc://nguyenhoangquan.com:7687",
    user="neo4j",
    password="Neoneo4j",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
)

result = e.decompose_query_debug("Theo quy định mới nhất, trẻ em phải đạt tối thiểu bao nhiêu tuổi VÀ chiều cao bao nhiêu thì mới được phép ngồi ở hàng ghế cạnh người lái xe (ghế phụ phía trước) của ô tô?")

print("=== REASONING (CoT) ===")
print(result.reasoning)
print("\n=== SUB-QUERIES ===")
for sq in result.sub_queries:
    print(f"  [{sq['label']}] {sq['text']}")
print(f"\nSuccess: {result.success}")

e.close()
