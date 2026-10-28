"""
Malaria Preprocessing node.

Applies to a column of images the same sequence of filters that was selected
by the filter learning framework, so that images acquired
with a low-cost microscope (LCM) can be prepared before running the Malaria
Detector node.

The filter implementations reproduce exactly those of the framework's
filters.py. The framework reads images with OpenCV, i.e. in BGR channel order,
and some filters (saturation boost, CLAHE) convert colour spaces assuming BGR:
images are therefore converted to BGR before filtering and back to RGB after.
"""

import json
import logging
import os

import knime.extension as knext
import numpy as np
from PIL import Image
from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)

knimeVis_category = kutil.get_knimeVis_category()


# ---------------------------------------------------------------------------
# Filters (same maths as filters.py of the framework)
# ---------------------------------------------------------------------------


def _median_blur(img, kernel_size):
    import cv2

    return cv2.medianBlur(img, int(kernel_size))


def _bilateral(img, diameter, sigmaColor, sigmaSpace):
    import cv2

    return cv2.bilateralFilter(
        src=img, d=int(diameter), sigmaColor=float(sigmaColor), sigmaSpace=float(sigmaSpace)
    )


def _saturation_boost(img, boost_factor):
    import cv2

    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * float(boost_factor), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def _gamma(img, gamma):
    import cv2

    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def _clahe(img, clip_limit, tile_grid_size):
    import cv2

    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size))
    )
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a, b)), cv2.COLOR_LAB2BGR)


def _unsharp_mask(img, sigma, strength):
    import cv2

    blurred = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return cv2.addWeighted(img, 1.0 + float(strength), blurred, -float(strength), 0)


def _laplacian_sharpen(img, ksize, scale, delta=0.0):
    import cv2

    laplacian = cv2.Laplacian(
        img, cv2.CV_64F, ksize=int(ksize), scale=float(scale), delta=float(delta)
    )
    sharpened = cv2.subtract(img.astype(np.float64), laplacian)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# Keys are the filter names used by the framework when it saves a pipeline.
FILTERS = {
    "NoOpFilter": lambda img: img,
    "MedianBlur": _median_blur,
    "BilateralFilter": _bilateral,
    "SaturationBoost": _saturation_boost,
    "GammaCorrection": _gamma,
    "CLAHE": _clahe,
    "UnsharpMask": _unsharp_mask,
    "LaplacianSharpen": _laplacian_sharpen,
}

## Pipeline found by the filter search for the reference malaria model
# (YOLOv3u trained with targeted data augmentation, model with_da_s2),
# searched on 200 LCM training images (runs/search/with_da_s2_train200).
DEFAULT_PIPELINE = [
    {"name": "MedianBlur", "parameters": {"kernel_size": 5}},
    {"name": "SaturationBoost", "parameters": {"boost_factor": 1.4}},
    {"name": "CLAHE", "parameters": {"clip_limit": 1.5, "tile_grid_size": 4}},
]


def load_pipeline(path):
    """
    Reads a pipeline from a JSON file. Accepted formats:
      - the orchestrator checkpoint, which has a "best_filters_combination" key;
      - a plain list of {"name": ..., "parameters": {...}}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("best_filters_combination")
    if not isinstance(data, list):
        raise ValueError(
            "The file must contain a list of filters or a 'best_filters_combination' key."
        )
    return data


def validate_pipeline(pipeline):
    for step in pipeline:
        if not isinstance(step, dict) or "name" not in step:
            raise ValueError(f"Invalid filter entry: {step!r}")
        if step["name"] not in FILTERS:
            raise ValueError(
                f"Unknown filter '{step['name']}'. Supported: {', '.join(FILTERS)}."
            )


def apply_pipeline(image, pipeline):
    """PIL image in, PIL image out. Filtering happens in BGR, as in training."""
    rgb = np.array(image.convert("RGB"))
    bgr = rgb[..., ::-1].copy()
    for step in pipeline:
        bgr = FILTERS[step["name"]](bgr, **step.get("parameters", {}))
    return Image.fromarray(bgr[..., ::-1].copy())


def describe_pipeline(pipeline):
    if not pipeline:
        return "no filters"
    parts = []
    for step in pipeline:
        params = ", ".join(f"{k}={v}" for k, v in step.get("parameters", {}).items())
        parts.append(f"{step['name']}({params})")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@knext.node(
    name="Malaria Preprocessing",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/malaria_preprocessing.png",
    category=knimeVis_category,
    id="dl-malaria-preprocessing",
)
@knext.input_table(
    name="Input Images",
    description="Table containing loaded images (PIL Image objects) to be preprocessed.",
)
@knext.output_table(
    name="Preprocessed Images",
    description=(
        "The input table with an additional column containing the preprocessed "
        "images, ready to be passed to the Malaria Detector node."
    ),
)
class MalariaPreprocessingNode:
    """
    Malaria Preprocessing Node

    Applies a sequence of image filters (noise reduction, colour correction,
    contrast enhancement, sharpening) to microscope images before malaria
    detection.

    **When to use it**
    Use this node only with a detection model that was trained with the same
    filters (targeted data augmentation). A model trained on unfiltered images
    usually performs *worse* on filtered images.

    **Pipeline source**
    - *Default pipeline*: the filters found for the reference malaria model
      (median blur with window 5, saturation boost 1.4, CLAHE with clip limit
      1.5 and 4x4 grid). Use it with that model. With this model the filters
      bring little or no measurable gain, because the model has already been
      trained on filtered images: you can also connect the images directly to
      the Malaria Detector.
    - *From file*: a JSON file produced by the filter learning framework, saved
      together with the model. Use it with any other model, so that the
      filters always match the ones selected for that model.

    The applied pipeline is reported in the node's flow variable
    "preprocessing_pipeline" so that it remains visible in the workflow.
    """

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column that contains the images to preprocess.",
        port_index=0,
        column_filter=kutil.is_png,
        include_row_key=False,
    )

    class PipelineSource(knext.EnumParameterOptions):
        DEFAULT = (
            "Default pipeline",
            "Pipeline found for the reference malaria detection model distributed with the extension.",
        )
        FILE = ("From file", "Pipeline saved by the framework in a JSON file, for any other model.")

    pipeline_source = knext.EnumParameter(
        label="Pipeline source",
        description="Where to read the sequence of filters from.",
        default_value=PipelineSource.DEFAULT.name,
        enum=PipelineSource,
    )

    pipeline_path = knext.LocalPathParameter(
        label="Pipeline file",
        description=(
            "JSON file with the sequence of filters, as saved by the filter "
            "learning framework. Used only when the pipeline source is 'From file'."
        ),
    ).rule(knext.OneOf(pipeline_source, [PipelineSource.FILE.name]), knext.Effect.SHOW)

    output_column = knext.StringParameter(
        label="Output column name",
        description="Name of the new column containing the preprocessed images.",
        default_value="Preprocessed Image",
    )

    def _get_pipeline(self):
        if self.pipeline_source == self.PipelineSource.DEFAULT.name:
            return DEFAULT_PIPELINE
        if not self.pipeline_path:
            raise knext.InvalidParametersError("Pipeline file path cannot be empty.")
        if not os.path.isfile(self.pipeline_path):
            raise knext.InvalidParametersError(
                f"Pipeline file not found: '{self.pipeline_path}'."
            )
        try:
            pipeline = load_pipeline(self.pipeline_path)
            validate_pipeline(pipeline)
        except (ValueError, json.JSONDecodeError) as e:
            raise knext.InvalidParametersError(f"Invalid pipeline file: {e}")
        return pipeline

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        image_columns = [c.name for c in input_schema if kutil.is_png(c)]
        if not image_columns:
            raise knext.InvalidParametersError("No image columns available in the input table.")

        if self.image_column is None:
            self.image_column = image_columns[-1]

        if not self.output_column.strip():
            raise knext.InvalidParametersError("Output column name cannot be empty.")
        if self.output_column in input_schema.column_names:
            raise knext.InvalidParametersError(
                f"A column named '{self.output_column}' already exists in the input table."
            )

        pipeline = self._get_pipeline()
        configure_context.flow_variables["preprocessing_pipeline"] = describe_pipeline(pipeline)

        image_ktype = input_schema[self.image_column].ktype
        return input_schema.append(knext.Column(image_ktype, self.output_column))

    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table):
        pipeline = self._get_pipeline()
        LOGGER.info(f"Applying preprocessing pipeline: {describe_pipeline(pipeline)}")
        execute_context.flow_variables["preprocessing_pipeline"] = describe_pipeline(pipeline)

        df = input_table.to_pandas()
        total = len(df)
        outputs = []

        for i, idx in enumerate(df.index):
            if execute_context.is_canceled():
                raise RuntimeError("Execution canceled by the user.")
            execute_context.set_progress(i / max(total, 1), f"Preprocessing image {i + 1} of {total}...")

            image = df.at[idx, self.image_column]
            if not isinstance(image, Image.Image):
                LOGGER.warning(f"Row {idx}: not a valid image, left empty.")
                outputs.append(None)
                continue
            outputs.append(apply_pipeline(image, pipeline))

        df[self.output_column] = outputs
        execute_context.set_progress(1.0)
        return knext.Table.from_pandas(df)