import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_sem_seg


CUSTOM_DATASET_CATEGORIES = [{"id": 0, "name":"NOT tumor"},
                            {"id": 1, "name":"NECROTIC"},
                             {"id": 2, "name":"EDEMA"},
                             {"id": 3, "name":"ENHANCING"}]

def _get_custom_dataset_meta():
    # Define your custom dataset categories and ids
    stuff_ids = [k["id"] for k in CUSTOM_DATASET_CATEGORIES]
    
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in CUSTOM_DATASET_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_all_custom_dataset(root):
    meta = _get_custom_dataset_meta()
    print("meta: ",meta)
    for name, dirname in [("train", "training"), ("val", "validation")]:  # Adjust dir names as per your dataset
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"BraTS20_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tiff", image_ext="tiff")  # You can change the extensions
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label = 0
        )
