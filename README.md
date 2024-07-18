# VGU-Rangers-2023

## Testing
To run the real-time detection module, use the `main_api.py` file. The options for running from the CLI:

|    Option    |    Default    |    Description   |
|     :---:    |    :----:     |       :---      |
| `-c`, `--config`  | `"last_tiny.json"` | The config file specifying classes and settings, must be `json` |
| `-m`, `--model`   | `"last_tiny.blob"` | The model file, must be converted to `.blob` |
| `-r`, `--record`   | `""` | Record the video during running model, must be `.avi` format. Does not record if not specified. |

For example, running the pipeline with another model and record the detection video would be:
```bash
python main_api.py -m another_model.blob -c another_model.json -r demo.avi
```
Testing with **the provided config and model files** and without recording, would be simple as
```bash
python main_api.py
```
