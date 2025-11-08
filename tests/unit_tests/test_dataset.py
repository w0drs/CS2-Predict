from ml_core.src.dataset.make_dataset import _load_dataset


def test_load_dataset():
    dataset_paths = [
        ("C:/Users/golik/OneDrive/Desktop/Stress_Dataset.csv", ","),
        ("C:/Users/golik/OneDrive/Desktop/Valorant_duo.csv", ","),
        ("C:/Desktop/Desktop/AllOldDesktops/D4/Valorant5.csv", ",")
    ]
    for path, sep in dataset_paths:
        dataset = _load_dataset(path=path, sep=sep)
        assert dataset is not None