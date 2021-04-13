
class DatasetEvaluator:
    """
    Base class for a dataset evaluator
    """
    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def update(self):
        """
        Update status given a mini-batch results
        """
        pass

    def mean_score(self):
        """
        Return mean score across all classes/samples
        """
        pass

    def class_score(self):
        """
        Return score for different classes
        """
        pass

    def num_samples(self):
        """
        return the evaluated samples
        """
        pass

    def main_metric(self):
        "return the name of the main metric"
        pass
