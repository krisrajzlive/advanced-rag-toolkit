# The vector-search and RAG ecosystem

Qdrant is an open-source vector database written in Rust. Qdrant implements the HNSW algorithm for approximate nearest-neighbour search. Qdrant supports scalar quantization, binary quantization and product quantization to reduce memory use. Qdrant supports multi-tenancy through payload indexes marked as tenant fields.

HNSW stands for Hierarchical Navigable Small World. HNSW is a graph-based index. HNSW was introduced by Yury Malkov and Dmitry Yashunin. HNSW exposes the parameters m, ef_construct and ef.

Scalar quantization compresses float32 vectors into int8 values. Binary quantization compresses each vector dimension into a single bit. Product quantization splits a vector into sub-vectors and encodes each with a codebook. Quantization trades a small amount of recall for large memory savings, and rescoring recovers most of the lost recall.

LlamaIndex is a data framework for building LLM applications. LlamaIndex provides the PropertyGraphIndex for Graph RAG. PropertyGraphIndex stores entities and relations in a property graph store. LlamaIndex integrates with Neo4j through Neo4jPropertyGraphStore.

Neo4j is a graph database. Neo4j stores data as nodes and relationships and is queried with the Cypher language. Neo4j can also hold vector indexes, which lets one database serve both graph traversal and vector search.

LangChain is a framework for composing LLM applications. LangGraph is a library built by the LangChain team for stateful agent workflows. LangSmith is the LangChain platform for tracing and evaluating LLM applications.

Keycloak is an open-source identity and access management server. Keycloak issues JSON Web Tokens that follow the OpenID Connect standard. A Keycloak group-membership mapper can add a tenant claim to the token. A RAG service uses the tenant claim to filter retrieval per tenant.

Graph RAG is a retrieval technique that retrieves entities and their relationships from a knowledge graph. Graph RAG answers multi-hop questions better than plain vector RAG because it follows relationships across documents. Microsoft Research popularised Graph RAG with community summaries. A knowledge graph is built by extracting subject-relation-object triples from text with an LLM.

Reranking reorders retrieved passages by relevance. A cross-encoder is a model commonly used for reranking. L2 normalization rescales an embedding to unit length, which makes dot product equal to cosine similarity.
