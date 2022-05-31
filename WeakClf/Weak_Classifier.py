class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    def classify(self, x):
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
