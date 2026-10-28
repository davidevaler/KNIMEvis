# knimeVizLab

![alt text](LogoProject.png)

## Description
This repository serves as the official [UNIBZ](https://www.unibz.it) project for creating custom KNIME nodes dedicated to image processing using Python. The goal is to provide seamless integration between KNIME's workflow analytics platform and Python's advanced capabilities for image analysis and processing.

With this project, users can:

- Develop custom KNIME nodes tailored to specific image processing requirements.
- Leverage Python libraries and tools for advanced image manipulation and analysis.
- Simplify workflows by combining KNIME's no-code/low-code environment with Python's scripting flexibility.

## Malaria detection nodes (fork)

This fork adds three nodes and a component for the automatic detection of
malaria parasites in thin blood smear images, developed by Davide Valer for
the bachelor thesis *"Machine Learning No-Code per Immagini Biomediche:
Rilevamento della Malaria con KNIME"* (University of Trento, 2026). They work
together with the existing nodes, in particular the **Image Loader**.

| Node | What it does |
|------|--------------|
| **Malaria Preprocessing** | Applies the filter pipeline found by the [filter learning framework](https://github.com/davidevaler/filter_learning_framework) (default pipeline of the reference model, or a pipeline loaded from a JSON file). |
| **Malaria Detector** | Runs a YOLO model on each image and returns one table per image (verdict, highest confidence, annotated image) and one table per detection (class, confidence, box, crop of the cell). |
| **YOLO Dataset Export** | Saves the original images and the confirmed detections in YOLO format, ready to update the model or to be opened in an annotation tool. |

The detections can be reviewed with the **Detection Review** component, included in the example workflow. It is not a new node, but a group of three standard KNIME nodes: **Sorter**, which orders the detections by increasing confidence, **Table View (JavaScript)**, where the user selects the false positives, and **Row Filter**, which removes the selected rows and returns only the confirmed detections.

The verdicts describe what the model detected, not the state of the patient:
the nodes are a decision-support tool and do not replace the examination by
an expert microscopist.

**Reference model and pipeline.** The trained model (`malaria_yolov3u_da.pt`)
and its filter pipeline are attached to the
[release of the framework](https://github.com/davidevaler/filter_learning_framework/releases/tag/tesi-v1).

**Example workflow.** `demo/malaria_detection.knwf` shows the full analysis.

**Installation.** The installable extension (update site) is attached to the
[releases of this repository](https://github.com/davidevaler/KNIMEvis/releases).
In KNIME: *Preferences → Install/Update → Available Software Sites → Add →
Archive*, select the zip file, then install the extension from
*File → Install KNIME Extensions*.


## Installation

To start developing KNIME nodes with Python, follow these steps:
1. **Clone this repository**:
    ```bash
    git clone ...
    ```

2. **Install the KNIME Analytics Platform**
    Install [KNIME Analytics Platform](https://docs.knime.com/ap/latest/analytics_platform_installation_guide/#install-knime-analytics-platform) version 5.5.0 or higher.
    Follow the offical guide line on how to develope pure [python nod in KNIME](https://docs.knime.com/developers/latest/create_a_node_with_python/)



## Usage
Once you have completed the installation, you can begin developing and using custom KNIME nodes for image processing. To integrate your Python-based node into KNIME, follow these steps:

1. Create your custom node in Python.
2. Integrate the node into your KNIME workflow by using the custom node created in the previous step.
3. Execute the workflow, and the image processing will be handled by Python scripts through KNIME's interface.

## Support
For any request about this software, please refer to the authors.

## Roadmap
This is a preliminary version. Further improvements and nodes will come soon. Stay tuned.

## Contributing
We welcome contributions to this project! To contribute:

- Fork this repository.
- Create a new branch.
- Make your changes and commit them.
- Submit a pull request.

Please make sure your code follows the existing style and includes appropriate tests.

## Authors and acknowledgment
Thanks to the collaboration with KNIME.

## License
The nodes are released under GPLv3.

## Project status
We are currently developing new nodes and improving the existing ones. Please feel free to contribute.
