import os
import asyncio
import numpy as np
import itertools
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import igraph
import leidenalg
from neo4j import GraphDatabase
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Load environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
index_name = "text_cluster_index"

# Initialize Tavily client for web scraping (optional, if required)
tavily_client = TavilyClient()

# Instantiate LLM and embedding models
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "temperature": 0,
    },
)

embedding_model = SentenceTransformerEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
transformer_model = embedding_model.model

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to process a list of texts and detect communities
def detect_communities(texts: list, question: str = None):
    # ========== Utility Functions for LLM-based Naming ==========
    def generate_name_for_text(llm, text, prompt="Give a short descriptive name for this text:"):
        """
        LLM-based name for a single chunk of text.
        """
        if not text.strip():
            return "NoContent"
        response = llm.invoke(
            f"""{prompt}
            Text:
            {text}

            Short name:"""
        )
        return response.content.strip()

    def generate_name_for_group(llm, texts, group_prompt="Name this community based on the following texts:"):
        """
        LLM-based name for a group of texts (e.g., a community).
        """
        if not texts:
            return "EmptyGroup"
        snippet = "\n\n".join(texts[:10])  # limit to first 10 to avoid huge prompt
        response = llm.invoke(
            f"""{group_prompt}
            Here are some representative texts of this group:
            {snippet}

            Short descriptive name for this group:"""
        )
        return response.content.strip()

    # Initialize text processing pipeline
    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=10)
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=text_splitter
    )

    # ========== Ingest Documents ==========
    for doc in texts:
        asyncio.run(kg_builder.run_async(text=doc))

    # ========== Create Vector Index ==========
    dims = transformer_model.get_sentence_embedding_dimension()
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dims,
        similarity_fn="cosine",
    )

    # ========== Remove Invalid Nodes ==========
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL OR c.text IS NULL OR c.text = ''
            DETACH DELETE c
        """)

    # ========== Ensure All Chunks Have UUID ==========
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = elementId(c)
        """)

    # ========== Retrieve Chunks ==========
    with driver.session() as session:
        results = session.run("""
            MATCH (c:Chunk)
            RETURN 
                elementId(c) AS nodeId, 
                c.embedding AS embedding, 
                c.text AS text
        """)
        records = list(results)

    node_ids = []
    embeddings = []
    node_text_map = {}

    for r in records:
        emb = r["embedding"]
        txt = r["text"]
        if emb and txt:
            node_ids.append(r["nodeId"])
            embeddings.append(emb)
            node_text_map[r["nodeId"]] = txt

    X = np.array(embeddings)
    num_nodes = len(X)

    if num_nodes == 0:
        print("No valid chunks found. Exiting.")
        driver.close()
        return []

    # ========== First-Level Community Detection ==========
    k = 4
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    edges = []
    weights = []

    for i in range(num_nodes):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = max(0, 1 - dist)
            edges.append((i, j_idx))
            weights.append(sim)

    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=1.0
    )
    community_labels = partition.membership  # each node's first-level community

    # ========== Write SIMILAR edges and Community Labels to Neo4j ==========
    with driver.session() as session:
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")

        for i in range(num_nodes):
            this_id = node_ids[i]
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                sim = 1 - dist
                that_id = node_ids[j_idx]
                session.run(
                    """
                    MERGE (a:Chunk { uuid: $idA })
                    SET a.uuid = $idA
                    MERGE (b:Chunk { uuid: $idB })
                    SET b.uuid = $idB
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $sim
                    """,
                    {"idA": this_id, "idB": that_id, "sim": sim},
                )

        # Write first-level community label
        for i in range(num_nodes):
            this_id = node_ids[i]
            comm = int(community_labels[i])
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.community = $community
            """, {"idVal": this_id, "community": comm})

    # ========== LLM-based Naming of Chunks and Communities ==========
    for i in range(num_nodes):
        chunk_text = node_text_map[node_ids[i]]
        chunk_name = generate_name_for_text(llm, chunk_text, 
            prompt="Give a short descriptive name for this chunk's content:")
        with driver.session() as session:
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.name = $chunkName
            """, {"idVal": node_ids[i], "chunkName": chunk_name})

    # ========== Final Output of Communities and Names ==========
    final_results = []
    with driver.session() as session:
        final_results = session.run("""
            MATCH (c:Chunk)
            RETURN
                c.uuid AS uuid,
                c.name AS chunkName,
                c.community_name AS communityName,
                c.super_community_name AS superCommunityName
            ORDER BY uuid
        """)

    communities = []
    for record in final_results:
        communities.append({
            'uuid': record['uuid'],
            'chunk_name': record['chunkName'],
            'community_name': record['communityName'],
            'super_community_name': record['superCommunityName']
        })

    driver.close()
    return communities

# To use the module:
if __name__ == "__main__":
    sample_texts = [
        "Who is Leo Messi?",
        "Who is Till Lindemann?",
        "What is System of a Down?",
        "The history of football",
        "Music of the 2000s"
    ]
    question = "What is the similarity between Leo Messi and System of a Down?"
    communities = detect_communities(sample_texts, question)
    print(communities)
