def CreateDataLoader(opt):
    from cloth_segmentation.data.custom_dataset_data_loader import CustomDatasetDataLoader

    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
