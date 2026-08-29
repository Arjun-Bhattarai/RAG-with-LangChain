from integration.integration import RAGPipeline


class RAGService:

    def __init__(self):
        self.rag = RAGPipeline()

    def query(self, query: str):
        return self.rag.run(
            query=query,
            evaluate=False,
        )