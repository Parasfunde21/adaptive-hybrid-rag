from training.generate_queries import generator


class EvaluationDataset:

    """
    Loads evaluation queries.

    Each sample contains

        query

        source_document (ground truth)

    """

    def load(self):

        return generator.generate_queries()


dataset = EvaluationDataset()


if __name__ == "__main__":

    queries = dataset.load()

    print()

    print(f"Loaded {len(queries)} evaluation queries")

    print()

    print(queries[0])