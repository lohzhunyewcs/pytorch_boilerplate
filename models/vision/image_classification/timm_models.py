import timm

def load_models(model_name: str, pretrained: str, num_class: int):
    model = timm.create_model(
        model_name=model_name, pretrained=pretrained,
    )
    