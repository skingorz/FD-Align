from architectures.feature_extractor.clip import ImageEncoder

def create_model():
    return ImageEncoder("RN50")