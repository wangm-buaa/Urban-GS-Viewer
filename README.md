# Urban-GS Viewer & Renderer

A lightweight web viewer and offline renderer for [Urban-GS: A Unified 3D Gaussian Splatting Framework for Compact and High-Fidelity Aerial-to-Street Reconstruction](https://github.com/wangm-buaa/Urban-GS) models, built on top of [nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

The core codebase is derived from [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The Gaussian-based rendering framework also follows design ideas from [yzslab/nerfstudio](https://github.com/yzslab/nerfstudio). We appreciate the open-source work and contributions from both projects.
<p align="center">
  <video width="80%" controls autoplay loop muted>
    <source src="assert/Visualization_example.mp4" type="video/mp4">
  </video>
  <br>
  <em>Note: The web viewer preview may exhibit lower visual quality than the actual rendered output video.</em>
</p>

## Notes

* The camera control logic has been updated to align with SIBR-style navigation. Use `WASDQE` and `IJKLUO
* ` to control the camera.
* Dynamic blur control has been disabled so the rendering resolution stays stable while the camera is moving.
* This project is focused on rendering only. Training is not supported, the implementation has not been extensively tested, and it is not as fast as the SIBR viewers.
* The nerfstudio web viewer can reach similar rendering quality to SIBR viewers. Recommended settings:
  * Increase the render resolution with the `Max Res` option under the `CONTROLS` tab.
  * Increase JPEG quality with `--config.viewer.jpeg-quality 100`, or switch to PNG output with `--config.viewer.image-format png`.
* The default scene orientation may be incorrect. You can adjust it in one of the following ways:
  * Add `--no-auto-reorient` to `run_viewer.py` or `render.py` if you want to keep the same coordinate system as the input dataset.
  * Click `RESET UP DIRECTION` under the `SCENE` tab to use the current viewpoint as the orientation.
  * Use `--ref-orientation IMAGE_NAME` to specify an image as the reference orientation.
* Hold the right mouse button and move the camera slightly before pressing `W`; otherwise, the camera may freeze after moving a short distance.
  * A fix has been submitted upstream: https://github.com/nerfstudio-project/nerfstudio/pull/2404
  * Alternatively, use this viewer: https://nsv.cslab.pro/23-09-15-1/?websocket_url=ws://localhost:7007
* Press `F5` to refresh the page if the web viewer stops updating. If you are creating a camera path, click `EXPORT PATH` to download it before refreshing, then use `LOAD PATH` to restore it afterward.

## Installation

1. Install the nerfstudio dependencies.
2. Run `pip install -e .` in this repository. This is required if you are reusing a virtual environment from another nerfstudio checkout.
3. Install the additional dependencies:

```bash
pip install plyfile==0.8.1
pip install ./submodules/gsplat-urbangs
pip install ./submodules/simple-knn
```

4. Install the viewer frontend dependencies:

```bash
cd nerfstudio/viewer/app
npm install
# or
yarn install
```

## Usage

### Viewer

Start the viewer frontend:

```bash
cd nerfstudio/viewer/app
yarn start
```

In a separate terminal, launch the Urban-GS viewer backend:

```bash
python nerfstudio/scripts/urban_gs/run_viewer.py --model-path GAUSSIAN_TRAINING_OUTPUT_MODEL_DIR
```

### Render

Render a camera path to a video:

```bash
python nerfstudio/scripts/urban_gs/render.py camera-path \
    --model-path GAUSSIAN_TRAINING_OUTPUT_MODEL_DIR \
    --camera-path-filename YOUR_CAMERA_PATH_FILE.json \
    --output-path YOUR_OUTPUT_MP4_FILE.mp4
```

## License

Code derived from [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) is licensed under the Apache-2.0 license.
