import knime.extension as knext

main_category = knext.category(
    path="/community",
    level_id="image_processing",
    name="Image Processing",
    description="Community extension for image processing nodes",
    icon="icons/unibz_icon64.png",
)


from knimeVisNodes import imageLoader,Denoising,EdgeDetection,Equalization,KnimeYOLO,SAM,MalariaDetect,Preprocessing,YOLO_DatasetExport
