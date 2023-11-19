import fiftyone as fo
import fiftyone.zoo as foz

class CocoDatasetManager:
    def __init__(self, dataset_name="coco-2017", classes=["person", "car"], max_samples=50):
        self.dataset_name = dataset_name
        self.classes = classes
        self.max_samples = max_samples

    def load_dataset(self, split):
        """
        Loads the specified split of the COCO dataset.

        Args:
        split (str): The dataset split to load ('train' or 'validation').

        Returns:
        fo.Dataset: The loaded FiftyOne dataset.
        """
        dataset = foz.load_zoo_dataset(
            self.dataset_name,
            split=split,
            label_types=["detections", "segmentations"],
            classes=self.classes,
            max_samples=self.max_samples
        )
        return dataset

    def save_dataset(self, dataset, export_dir):
        """
        Saves the specified dataset to disk.

        Args:
        dataset (fo.Dataset): The FiftyOne dataset to save.
        export_dir (str): The directory where the dataset will be exported.
        """
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset
        )

    def load_saved_dataset(self, dataset_dir):
        """
        Loads a dataset from the specified directory.

        Args:
        dataset_dir (str): The directory from which to load the dataset.

        Returns:
        fo.Dataset: The loaded FiftyOne dataset.
        """
        return fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneDataset
        )

