import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from collections import Counter

class CocoDatasetManager:
    def __init__(self, dataset_name="coco-2017", classes=None, max_samples=10000):
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
    
    def dominant_label_box_size(self,detections):
        max_area = 0
        dominant_class = None

        for detection in detections:
            x_min, y_min, width, height = detection.bounding_box
            area = width * height
            if area > max_area:
                max_area = area
                dominant_class = detection.label
        return dominant_class
    
    def dominant_majority_class(self,detections):
        labels = [detection.label for detection in detections]
        most_common_label = Counter(labels).most_common(1)[0][0]
        return most_common_label



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
    

    def load_dataset_seperate_folder(self, split, export_root_dir):
        """
        Loads the specified split of the COCO dataset and exports each class into its own folder.

        Args:
        split (str): The dataset split to load ('train' or 'validation').
        export_root_dir (str): The root directory where the dataset will be exported, with each class in its own folder.

        Returns:
        fo.Dataset: The loaded FiftyOne dataset.
        """
        # Load the dataset
        dataset = foz.load_zoo_dataset(
            self.dataset_name,
            split=split,
            label_types=["detections", "segmentations"],
            classes=self.classes,
            max_samples=self.max_samples
        )

        # Iterate over all samples in the dataset
        for sample in dataset:
            # Check if 'detections' field exists and is not None
            if 'detections' in sample and sample['detections'] is not None:
                # Get the dominant label
                label = self.dominant_label_box_size(sample['detections'].detections)

                # Create a directory for the class if it doesn't exist
                class_dir = os.path.join(export_root_dir, label)
                os.makedirs(class_dir, exist_ok=True)

                # Define the export path for the sample
                export_path = os.path.join(class_dir, os.path.basename(sample.filepath))

                # Copy the file to the class-specific directory
                shutil.copy(sample.filepath, export_path)

        return dataset
    

