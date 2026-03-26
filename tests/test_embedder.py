import pytest
from legal_scraper.embedder import Neo4jEmbedder


class TestNeo4jEmbedderInit:
    def test_embedder_stores_uri_and_credentials(self):
        embedder = Neo4jEmbedder(
            uri="neo4j+ssc://host:7687",
            user="neo4j",
            password="secret",
            database="neo4j",
        )
        assert embedder.uri == "neo4j+ssc://host:7687"
        assert embedder.user == "neo4j"
        assert embedder.password == "secret"
        assert embedder.database == "neo4j"
        assert embedder._driver is None  # lazy connection

    def test_close_without_driver_is_safe(self):
        embedder = Neo4jEmbedder(uri="bolt://x", user="x", password="x")
        embedder.close()  # should not raise

    def test_close_after_driver_open(self):
        embedder = Neo4jEmbedder(
            uri="neo4j+ssc://nguyenhoangquan.com:7687",
            user="neo4j",
            password="Neoneo4j",
        )
        driver = embedder._get_driver()
        assert driver is not None
        embedder.close()
        assert embedder._driver is None  # driver was closed

    def test_get_driver_is_lazy(self):
        embedder = Neo4jEmbedder(
            uri="neo4j+ssc://nguyenhoangquan.com:7687",
            user="neo4j",
            password="Neoneo4j",
        )
        assert embedder._driver is None
        driver = embedder._get_driver()
        assert driver is not None
        assert embedder._driver is driver  # same instance
        embedder.close()
