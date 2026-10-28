"""
YOLO Dataset Export node.

Writes the original images and the confirmed detections in YOLO format, so that
they can be used to fine-tune the detection model or opened in an annotation
tool (CVAT, Label Studio, ...) to add the boxes the model missed.

Output folder structure:
    <output folder>/
        images/<name>.png
        labels/<name>.txt      one line per box: class_id x_center y_center width height
        classes.txt            one class name per line, in class-ID order
        data.yaml              dataset description for Ultralytics
"""

import logging
import os
import re
from urllib.parse import unquote

import knime.extension as knext
import pandas as pd
from PIL import Image
from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)

knimeVis_category = kutil.get_knimeVis_category()

REQUIRED_DETECTION_COLUMNS = {
    "Image Row ID": knext.string(),
    "Class ID": knext.int64(),
    "Class": knext.string(),
    "X Center": knext.int64(),
    "Y Center": knext.int64(),
    "Width": knext.int64(),
    "Height": knext.int64(),
}

OUTPUT_COLUMNS = ["Image ID", "Image File", "Label File", "Box Count"]


def _is_string_column(column) -> bool:
    return column.ktype == knext.string()


def _file_name(value):
    """'/data/lcm/img_042.png' or 'file:/C:/data/img_042.png' -> 'img_042.png'."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    name = re.split(r"[\\/]", unquote(str(value)).rstrip("/\\"))[-1]
    return name or None


def _safe(name):
    """Removes characters that are not allowed in file names on common systems."""
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("._") or "image"


def _yolo_line(class_id, xc, yc, w, h, img_w, img_h):
    """Pixel box -> YOLO line with coordinates normalised to [0, 1]."""
    def clip(v):
        return min(max(v, 0.0), 1.0)

    return (
        f"{int(class_id)} {clip(xc / img_w):.6f} {clip(yc / img_h):.6f} "
        f"{clip(w / img_w):.6f} {clip(h / img_h):.6f}"
    )


@knext.node(
    name="YOLO Dataset Export",
    node_type=knext.NodeType.SINK,
    icon_path="icons/yolo_dataset_export.png",
    category=knimeVis_category,
    id="dl-yolo-dataset-export",
)
@knext.input_table(
    name="Images",
    description=(
        "The same table that enters the Malaria Detector node (for example the "
        "output of Malaria Preprocessing). It must contain the original images "
        "as loaded by the Image Loader, and its row IDs must not be changed in "
        "between, because the detections refer to the images through their row "
        "ID (column 'Image Row ID')."
    ),
)
@knext.input_table(
    name="Confirmed Detections",
    description=(
        "The detections to export, usually the output of the Detection Review "
        "component (only the detections confirmed by the user). It must contain "
        "the columns produced by the second output of the Malaria Detector: "
        "Image Row ID, Class ID, Class, X Center, Y Center, Width and Height. "
        "Other columns, such as Detection Crop, are ignored."
    ),
)
@knext.output_table(
    name="Exported Files",
    description=(
        "One row per exported image: image name, absolute path of the image "
        "file and of the label file, and number of boxes written."
    ),
)
class YoloDatasetExportNode:
    """
    YOLO Dataset Export Node

    Saves the original images together with the confirmed detections in YOLO
    format, so that they can be used to fine-tune the detection model or opened
    in an annotation tool (e.g. CVAT, Label Studio) to add the parasites that
    the model missed.

    **Typical workflow**
    Image Loader → Malaria Preprocessing → Malaria Detector → Detection Review
    → YOLO Dataset Export. Connect the table entering the Malaria Detector to
    the first input and the output of Detection Review to the second input.

    **Output folder structure**
    - `images/<name>.png`: the original images;
    - `labels/<name>.txt`: one line per box, `class_id x_center y_center width height`,
      with coordinates normalised between 0 and 1;
    - `classes.txt`: one class name per line, in class-ID order;
    - `data.yaml`: dataset description for Ultralytics training.

    **Original images.** Export the images *before* preprocessing: during
    training the filters are applied again by the data augmentation. Box
    coordinates remain valid because the filters do not move pixels.

    **Images without boxes.** If enabled, images with no confirmed detection
    are exported with an empty label file. For a detection model this means
    "this image contains no parasite", so make sure that this is true before
    using them for training: if the model missed some parasites, add them in an
    annotation tool first, or disable the option.
    """

    image_column = knext.ColumnParameter(
        label="Original image column (not preprocessed)",
        description=(
            "Column of the first input with the original images, i.e. the column "
            "created by the Image Loader node (by default 'Image'). Do not choose "
            "'Preprocessed Image' or other filtered or annotated images: the model "
            "must be fine-tuned on the original images, because the filters are "
            "applied by the data augmentation during training."
        ),
        port_index=0,
        column_filter=kutil.is_png,
        include_row_key=False,
    )

    name_column = knext.ColumnParameter(
        label="Image name column (optional)",
        description=(
            "Optional string column with the file path or file name of each "
            "image, e.g. the output of the Path to String node. The file name "
            "(without extension) is used to name the exported image and label "
            "files, so that they can be traced back to the original images. If "
            "no column is selected, the row ID is used instead."
        ),
        port_index=0,
        column_filter=_is_string_column,
        include_row_key=False,
    )

    output_folder = knext.StringParameter(
        label="Output folder (absolute path)",
        description=(
            "Full path of the folder where the dataset is written, for example "
            "/Users/name/Desktop/malaria_export on macOS and Linux or "
            "C:\\Users\\name\\Desktop\\malaria_export on Windows. Relative paths "
            "(such as 'export') are not accepted, because it would not be clear "
            "where the folder is created. The folder is created if it does not "
            "exist, and the 'images' and 'labels' subfolders are created inside it."
        ),
        default_value="",
    )

    class_names = knext.StringParameter(
        label="Class names (in model order)",
        description=(
            "Names of all the classes of the model, separated by commas, in the "
            "order of their class ID: the first name is class 0, the second class "
            "1, and so on. YOLO label files contain only the class number, so "
            "this list is written to classes.txt and data.yaml to say which "
            "parasite stage each number stands for.\n\n"
            "- Write all the classes of the model, even those that never appear "
            "in the exported detections.\n"
            "- The order must be the same as in the model. For the reference "
            "malaria model (M5 dataset) it is: schizont, gametocyte, ring, "
            "trophozoite, which is the default value.\n"
            "- If you use another model, the names and their order are shown in "
            "the Class ID and Class columns of the Malaria Detector output.\n\n"
            "The node checks the list against the Class ID and Class columns of "
            "the detections and stops with an error if they do not match, so that "
            "labels are never saved with swapped classes."
        ),
        default_value="schizont, gametocyte, ring, trophozoite",
    )

    include_empty = knext.BoolParameter(
        label="Export images without detections",
        description=(
            "Also export the images with no confirmed detection, with an empty "
            "label file. For training, an empty label file means that the image "
            "contains no parasites: enable this option only if that is true, "
            "otherwise add the missing parasites in an annotation tool first."
        ),
        default_value=True,
    )

    overwrite = knext.BoolParameter(
        label="Overwrite existing files",
        description=(
            "If disabled, the node stops with an error instead of overwriting "
            "files that already exist in the output folder. Enable it to "
            "re-export into the same folder."
        ),
        default_value=False,
    )

    def configure(self, configure_context, images_schema, detections_schema):
        image_columns = [c.name for c in images_schema if kutil.is_png(c)]
        if not image_columns:
            raise knext.InvalidParametersError("No image column in the Images table.")
        if self.image_column is None:
            # The original image is usually the first one; preprocessed and
            # annotated copies are appended later in the workflow.
            self.image_column = image_columns[0]

        if self.name_column is None:
            candidates = [
                c.name
                for c in images_schema
                if _is_string_column(c)
                and any(k in c.name.lower() for k in ("path", "file", "url", "name", "location"))
            ]
            if candidates:
                self.name_column = candidates[-1]

        missing = [
            name for name in REQUIRED_DETECTION_COLUMNS
            if name not in detections_schema.column_names
        ]
        if missing:
            raise knext.InvalidParametersError(
                "The Confirmed Detections table is missing these columns: "
                + ", ".join(missing)
                + ". Connect the output of the Malaria Detector or of the Detection Review component."
            )

        if not [n for n in self.class_names.split(",") if n.strip()]:
            raise knext.InvalidParametersError("Class names cannot be empty.")

        if not self.output_folder.strip():
            raise knext.InvalidParametersError("Output folder cannot be empty.")
        if not os.path.isabs(os.path.expanduser(self.output_folder.strip())):
            raise knext.InvalidParametersError(
                "Output folder must be an absolute path, e.g. /Users/name/Desktop/export "
                "or C:\\Users\\name\\Desktop\\export."
            )

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Image ID"),
                knext.Column(knext.string(), "Image File"),
                knext.Column(knext.string(), "Label File"),
                knext.Column(knext.int64(), "Box Count"),
            ]
        )

    def execute(self, execute_context, images_table, detections_table):
        images_df = images_table.to_pandas()
        # Only the needed columns are converted: image columns such as
        # 'Detection Crop' may come back from the Table View (JavaScript) in a
        # format that cannot be converted to pandas.
        try:
            detections_df = detections_table[list(REQUIRED_DETECTION_COLUMNS)].to_pandas()
        except Exception:
            detections_df = detections_table.to_pandas()

        root = os.path.abspath(os.path.expanduser(self.output_folder.strip()))
        LOGGER.info(f"Exporting YOLO dataset to {root}")
        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Detections grouped by source image.
        detections_df["Image Row ID"] = detections_df["Image Row ID"].astype(str)
        by_image = {key: group for key, group in detections_df.groupby("Image Row ID")}

        unknown = set(by_image) - set(map(str, images_df.index))
        if unknown:
            raise ValueError(
                f"{len(unknown)} detection(s) refer to images not present in the "
                f"Images table (e.g. row '{sorted(unknown)[0]}'). Connect the same "
                "table that enters the Malaria Detector."
            )

        # Class names: the list given by the user, checked against the detections.
        names = [n.strip() for n in self.class_names.split(",") if n.strip()]
        for class_id, class_name in set(zip(detections_df["Class ID"], detections_df["Class"])):
            class_id = int(class_id)
            if class_id >= len(names) or names[class_id] != str(class_name):
                expected = names[class_id] if class_id < len(names) else "none"
                raise ValueError(
                    f"Class ID {class_id} is '{class_name}' in the detections but "
                    f"'{expected}' in the 'Class names' setting. Write the class "
                    "names in the same order as the model."
                )

        rows, keys, used_names = [], [], set()
        total = len(images_df)

        for i, idx in enumerate(images_df.index):
            if execute_context.is_canceled():
                raise RuntimeError("Execution canceled by the user.")
            execute_context.set_progress(i / max(total, 1), f"Exporting image {i + 1} of {total}...")

            boxes = by_image.get(str(idx))
            if boxes is None and not self.include_empty:
                continue

            image = images_df.at[idx, self.image_column]
            if not isinstance(image, Image.Image):
                LOGGER.warning(f"Row {idx}: not a valid image, skipped.")
                continue

            # File name: original name without extension, made unique.
            file_name = (
                _file_name(images_df.at[idx, self.name_column])
                if self.name_column and self.name_column in images_df.columns
                else None
            )
            base = _safe(os.path.splitext(file_name)[0] if file_name else str(idx))
            name, n = base, 2
            while name in used_names:
                name, n = f"{base}_{n}", n + 1
            used_names.add(name)

            image_path = os.path.join(images_dir, f"{name}.png")
            label_path = os.path.join(labels_dir, f"{name}.txt")
            if not self.overwrite:
                for path in (image_path, label_path):
                    if os.path.exists(path):
                        raise FileExistsError(
                            f"'{path}' already exists. Choose another folder or "
                            "enable 'Overwrite existing files'."
                        )

            img_w, img_h = image.size
            lines = []
            if boxes is not None:
                for _, b in boxes.iterrows():
                    lines.append(
                        _yolo_line(
                            b["Class ID"], b["X Center"], b["Y Center"],
                            b["Width"], b["Height"], img_w, img_h,
                        )
                    )

            (image if image.mode in ("RGB", "L") else image.convert("RGB")).save(image_path, "PNG")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            rows.append([file_name or str(idx), image_path, label_path, len(lines)])
            keys.append(str(idx))

        # Class list for annotation tools and Ultralytics.
        with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(root)}\n")
            f.write("train: images\nval: images\n")
            f.write(f"nc: {len(names)}\n")
            f.write("names:\n")
            for k, class_name in enumerate(names):
                f.write(f"  {k}: {class_name}\n")

        execute_context.set_progress(1.0)
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS, index=pd.Index(keys, dtype=str))
        out = out.astype(
            {"Image ID": "string", "Image File": "string", "Label File": "string", "Box Count": "int64"}
        )
        return knext.Table.from_pandas(out, row_ids="keep")