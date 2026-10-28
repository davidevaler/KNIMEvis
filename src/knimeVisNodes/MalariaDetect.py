import logging
import os
import re
from urllib.parse import unquote

import knime.extension as knext
import pandas as pd
import torch
from PIL import Image
from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)

knimeVis_category = kutil.get_knimeVis_category()

# Column names, defined once so that configure() and execute() cannot diverge.
SUMMARY_COLUMNS = [
    "Image ID",
    "Verdict",
    "Max Confidence",
    "Detection Count",
    "Needs Review",
    "Annotated Image",
]
DETAIL_COLUMNS = [
    "Detection ID",
    "Image ID",
    "Image Row ID",
    "Class ID",
    "Class",
    "Confidence",
    "X Center",
    "Y Center",
    "Width",
    "Height",
    "Detection Crop",
]

# Drawing settings
BOX_COLOR_CONFIDENT = (220, 30, 30)    # red: confidence >= threshold
BOX_COLOR_UNCERTAIN = (255, 165, 0)    # orange: confidence < threshold
CROP_MARGIN = 0.5                      # margin around the box, as a fraction of its size
CROP_SIZE = 160                        # longer side of the crop, in pixels


def _is_string_column(column) -> bool:
    return column.ktype == knext.string()


def _file_name(value):
    """
    Extracts the file name from a path or URL string, e.g.
    '/data/lcm/img_042.png' or 'file:/C:/data/img_042.png' -> 'img_042.png'.
    Returns None if the value is missing or empty.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    name = re.split(r"[\\/]", unquote(str(value)).rstrip("/\\"))[-1]
    return name or None


def _to_rgb(image: Image.Image) -> Image.Image:
    """YOLO expects 3-channel images: convert RGBA, grayscale, palette, etc."""
    return image if image.mode == "RGB" else image.convert("RGB")


def _font(size):
    """Default PIL font; Pillow >= 10.1 accepts a size, older versions do not."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _annotate(image, detections, threshold):
    """
    Draws every box with its Detection ID and confidence, so that the reviewer
    can match each row of the Detections Detail table with a box in the image.
    Boxes below the confidence threshold are drawn in a different colour.
    """
    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(image.size) / 300))
    font = _font(max(14, round(min(image.size) / 40)))

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        color = BOX_COLOR_CONFIDENT if det["confidence"] >= threshold else BOX_COLOR_UNCERTAIN
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{det['number']}  {det['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        label_y = y1 - text_h - 6 if y1 - text_h - 6 >= 0 else y2 + 2
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 6], fill=color)
        draw.text((x1 + 3 - left, label_y + 3 - top), label, fill=(255, 255, 255), font=font)

    return annotated


def _shrink(image, max_size):
    """Reduces the image so that its longer side is at most max_size pixels."""
    if not max_size or max(image.size) <= max_size:
        return image
    reduced = image.copy()
    reduced.thumbnail((max_size, max_size), Image.LANCZOS)
    return reduced


def _crop(image, xyxy, color):
    """
    Cuts out the detected cell with some margin around it, draws the box on the
    crop and enlarges it, so that a single cell can be inspected closely.
    """
    from PIL import ImageDraw

    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    left = max(0, int(x1 - CROP_MARGIN * w))
    top = max(0, int(y1 - CROP_MARGIN * h))
    right = min(image.width, int(x2 + CROP_MARGIN * w))
    bottom = min(image.height, int(y2 + CROP_MARGIN * h))

    crop = image.crop((left, top, right, bottom))
    ImageDraw.Draw(crop).rectangle(
        [x1 - left, y1 - top, x2 - left, y2 - top], outline=color, width=2
    )

    scale = CROP_SIZE / max(crop.size)
    new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    return crop.resize(new_size, Image.LANCZOS)


@knext.node(
    name="Malaria Detector",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/malaria_detector.png",
    category=knimeVis_category,
    id="dl-malaria-detector",
)
@knext.input_table(
    name="Input Images",
    description="Table containing loaded images (PIL Image objects) to run malaria detection on.",
)
@knext.output_table(
    name="Diagnosis Summary",
    description=(
        "One row per image, with the aggregated verdict, the highest detection "
        "confidence found, whether the image needs clinical review, and the "
        "image with the detected bounding boxes drawn on it."
    ),
)
@knext.output_table(
    name="Detections Detail",
    description=(
        "One row per detected bounding box (the row ID is the Detection ID), "
        "with a unique Detection ID, class, "
        "confidence, pixel coordinates and an enlarged crop of the detected cell. "
        "The number after the dash in the Detection ID is the one drawn next to "
        "the box in the annotated image."
    ),
)
class MalariaDetectorNode:
    """
    Malaria Detector Node

    Runs a YOLO object detection model on a column of loaded images and
    reports, for each image, whether malaria parasites were detected and
    with how much confidence.

    **How it works**
    Point "Model file path" to a local YOLO weights file (.pt). If you don't
    have one, you can download the pre-trained malaria detection model
    from the project release:
    https://github.com/davidevaler/filter_learning_framework/releases/tag/tesi-v1

    Images whose highest detection confidence falls below the confidence
    threshold are flagged as "needs review", so that a clinician can confirm
    or correct the automatic verdict further down the pipeline. Each image is
    also returned with the detected bounding boxes drawn on it: every box is
    numbered (1 = most confident), red if above the confidence threshold and
    orange if below.
    """

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column that contains the loaded images.",
        port_index=0,
        column_filter=kutil.is_png,
        include_row_key=False,
    )

    name_column = knext.ColumnParameter(
        label="Image name column",
        description=(
            "Optional string column with the file path or name of each image, "
            "e.g. the output of the Path to String node. The file name is used "
            "as Image ID and in the Detection IDs. If no column is selected, "
            "the row ID of the input table is used instead."
        ),
        port_index=0,
        column_filter=_is_string_column,
        include_row_key=False,
        since_version="1.2.0",
    )

    model_path = knext.LocalPathParameter(
        label="Model file path",
        description=(
            "Path to the local YOLO weights file (.pt). See the node "
            "description for a link to the pre-trained model."
        ),
    )

    confidence_threshold = knext.DoubleParameter(
        label="Confidence threshold",
        description=(
            "Minimum detection confidence for the verdict to be accepted "
            "automatically. Images whose highest confidence is below this "
            "value are flagged as needing clinical review."
        ),
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
    )

    min_raw_confidence = knext.DoubleParameter(
        label="Minimum raw detection confidence",
        description=(
            "Confidence threshold passed to the model itself when running "
            "inference. Keep this lower than the confidence threshold above, "
            "so that weak detections are still captured and shown to the "
            "clinician during review, instead of being discarded before the "
            "review step ever sees them."
        ),
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True,
    )

    iou_threshold = knext.DoubleParameter(
        label="IoU threshold (NMS)",
        description="Intersection-over-union threshold used for non-maximum suppression.",
        default_value=0.45,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True,
    )

    preview_max_size = knext.IntParameter(
        label="Annotated image maximum size (pixels)",
        description=(
            "Longer side of the annotated image shown in the Diagnosis Summary "
            "table. Large microscope images may not open in the KNIME image "
            "viewer, so the annotated copy is reduced to this size. Boxes are "
            "drawn before resizing, and detection coordinates and crops always "
            "refer to the full-resolution image. Set to 0 to keep the original size."
        ),
        default_value=1600,
        min_value=0,
        is_advanced=True,
    )

    class DeviceOptions(knext.EnumParameterOptions):
        CPU = ("cpu", "Use CPU for inference")
        GPU = ("gpu", "Use GPU for inference if CUDA is available")

    device: str = knext.EnumParameter(
        label="Computation Device",
        description="Select the device for running model inference: CPU or GPU (CUDA).",
        default_value=DeviceOptions.CPU.name,
        enum=DeviceOptions,
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]
        if not image_columns:
            raise knext.InvalidParametersError("No image columns available in the input table.")

        if self.image_column is None:
            self.image_column = image_columns[-1][0]

        if self.name_column is None:
            # Guess a column that looks like a path or a file name.
            candidates = [
                c.name
                for c in input_schema_1
                if _is_string_column(c)
                and any(k in c.name.lower() for k in ("path", "file", "url", "name", "location"))
            ]
            if candidates:
                self.name_column = candidates[-1]
        elif self.name_column not in input_schema_1.column_names:
            raise knext.InvalidParametersError(
                f"Image name column '{self.name_column}' not found in the input table."
            )

        if not self.model_path:
            raise knext.InvalidParametersError("Model file path cannot be empty.")

        if not os.path.isfile(self.model_path):
            raise knext.InvalidParametersError(
                f"Model file not found at the specified path: '{self.model_path}'."
            )

        if not self.model_path.lower().endswith(".pt"):
            raise knext.InvalidParametersError(
                f"Expected a '.pt' weights file, got: '{self.model_path}'."
            )

        # The image output column reuses the type of the input image column,
        # so it is exactly the type the rest of the extension already uses.
        image_ktype = input_schema_1[self.image_column].ktype

        summary_schema = knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Image ID"),
                knext.Column(knext.string(), "Verdict"),
                knext.Column(knext.double(), "Max Confidence"),
                # int64, not int32: pandas creates Python ints as int64, and a
                # mismatch here causes the "DataSpec ... does not match" warning.
                knext.Column(knext.int64(), "Detection Count"),
                knext.Column(knext.bool_(), "Needs Review"),
                knext.Column(image_ktype, "Annotated Image"),
            ]
        )

        detail_schema = knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Detection ID"),
                knext.Column(knext.string(), "Image ID"),
                # Row ID of the source image, used by the export node to match
                # each detection with its original image.
                knext.Column(knext.string(), "Image Row ID"),
                # Class index of the model, needed to write YOLO label files.
                knext.Column(knext.int64(), "Class ID"),
                knext.Column(knext.string(), "Class"),
                knext.Column(knext.double(), "Confidence"),
                knext.Column(knext.int64(), "X Center"),
                knext.Column(knext.int64(), "Y Center"),
                knext.Column(knext.int64(), "Width"),
                knext.Column(knext.int64(), "Height"),
                knext.Column(image_ktype, "Detection Crop"),
            ]
        )

        return summary_schema, detail_schema

    def execute(
        self,
        execute_context: knext.ExecutionContext,
        input_table: knext.Table,
    ):
        from ultralytics import YOLO

        if self.device == "GPU":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                execute_context.set_warning("GPU not available, falling back to CPU")
        else:
            device = "cpu"

        LOGGER.info(f"Loading model from: {self.model_path}")
        model = YOLO(self.model_path)

        input_df = input_table.to_pandas()

        if self.image_column not in input_df.columns:
            raise ValueError(f"Image column '{self.image_column}' not found.")

        summary_rows = []
        summary_keys = []
        detail_rows = []
        detail_keys = []
        used_keys = set()
        duplicated_names = False

        total_rows = len(input_df)

        for i, idx in enumerate(input_df.index):
            if execute_context.is_canceled():
                LOGGER.warning("Execution canceled by the user.")
                break

            execute_context.set_progress(
                i / total_rows, f"Running inference on image {i + 1} of {total_rows}..."
            )

            # Image ID shown to the user: the file name if available, else the row ID.
            file_name = (
                _file_name(input_df.at[idx, self.name_column])
                if self.name_column and self.name_column in input_df.columns
                else None
            )
            image_id = file_name or str(idx)

            # Key used to build Detection IDs: the file name without extension,
            # made unique if two images share the same name (e.g. HCM/LCM pairs).
            base_key = os.path.splitext(image_id)[0] if file_name else image_id
            image_key, n = base_key, 2
            while image_key in used_keys:
                image_key, n = f"{base_key}_{n}", n + 1
                duplicated_names = True
            used_keys.add(image_key)

            summary_keys.append(str(idx))
            image = input_df.at[idx, self.image_column]

            if not isinstance(image, Image.Image):
                LOGGER.warning(f"Skipping row {idx} - not a valid image")
                summary_rows.append([image_id, "Error - invalid image", 0.0, 0, False, None])
                continue

            image = _to_rgb(image)

            results = model.predict(
                source=image,
                conf=self.min_raw_confidence,
                iou=self.iou_threshold,
                device=device,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            detection_count = len(boxes)

            # Boxes sorted by confidence, so that number 1 is the most confident.
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_id": int(box.cls[0]),
                        "class_name": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "xyxy": (x1, y1, x2, y2),
                    }
                )
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            for number, det in enumerate(detections, start=1):
                det["number"] = number

            annotated_image = _shrink(
                _annotate(image, detections, self.confidence_threshold),
                self.preview_max_size,
            )

            if detection_count == 0:
                summary_rows.append(
                    [image_id, "Malaria not detected", 0.0, 0, False, annotated_image]
                )
                continue

            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                color = (
                    BOX_COLOR_CONFIDENT
                    if det["confidence"] >= self.confidence_threshold
                    else BOX_COLOR_UNCERTAIN
                )
                detection_id = f"{image_key}-{det['number']}"
                detail_keys.append(detection_id)
                detail_rows.append(
                    [
                        detection_id,
                        image_id,
                        str(idx),
                        det["class_id"],
                        det["class_name"],
                        det["confidence"],
                        int(round((x1 + x2) / 2)),
                        int(round((y1 + y2) / 2)),
                        int(round(x2 - x1)),
                        int(round(y2 - y1)),
                        _crop(image, det["xyxy"], color),
                    ]
                )

            max_confidence = detections[0]["confidence"]
            needs_review = max_confidence < self.confidence_threshold
            verdict = "Uncertain - review needed" if needs_review else "Malaria detected"

            summary_rows.append(
                [
                    image_id,
                    verdict,
                    max_confidence,
                    detection_count,
                    needs_review,
                    annotated_image,
                ]
            )

        execute_context.set_progress(1.0)

        if duplicated_names:
            execute_context.set_warning(
                "Some images share the same file name: a suffix (_2, _3, ...) was "
                "added to their Detection IDs to keep them unique."
            )

        # Row IDs: the input row ID for images, the Detection ID for detections.
        summary_df = pd.DataFrame(
            summary_rows, columns=SUMMARY_COLUMNS, index=pd.Index(summary_keys, dtype=str)
        )
        detail_df = pd.DataFrame(
            detail_rows, columns=DETAIL_COLUMNS, index=pd.Index(detail_keys, dtype=str)
        )

        # Explicit dtypes: they must match configure() also when a table is
        # empty, otherwise pandas falls back to generic "object" columns.
        summary_df = summary_df.astype(
            {
                "Image ID": "string",
                "Verdict": "string",
                "Max Confidence": "float64",
                "Detection Count": "int64",
                "Needs Review": "bool",
            }
        )
        detail_df = detail_df.astype(
            {
                "Detection ID": "string",
                "Image ID": "string",
                "Image Row ID": "string",
                "Class ID": "int64",
                "Class": "string",
                "Confidence": "float64",
                "X Center": "int64",
                "Y Center": "int64",
                "Width": "int64",
                "Height": "int64",
            }
        )

        return (
            knext.Table.from_pandas(summary_df, row_ids="keep"),
            knext.Table.from_pandas(detail_df, row_ids="keep"),
        )