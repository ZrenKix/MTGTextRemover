import os
import sys
import argparse
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from text_removal.config import (
    DEFAULT_PHRASES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PAD_WIDTH,
    DEFAULT_PAD_HEIGHT,
    DEFAULT_INPAINT_RADIUS,
    DEFAULT_INPAINT_METHOD,
    DEFAULT_DILATE,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_COMBINE_THRESHOLD
)
from text_removal.remove import remove_phrases
from text_removal.tesseract_utils import configure_tesseract

"""
Command-line interface to remove specified text from images using Tesseract OCR.
Allows configuration via a YAML file or CLI arguments.
"""

def load_config_from_yaml(file_path):
    """
    Loads a YAML file and returns its contents as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def merge_config(cli_args, yaml_config):
    """
    Merges defaults, YAML config, and CLI args into a single dictionary.
    Priority: CLI > YAML > defaults.
    """
    final_config = {
        "texts": DEFAULT_PHRASES,
        "tesseract_cmd": None,
        "debug": False,
        "max_workers": DEFAULT_MAX_WORKERS,
        "pad_width": DEFAULT_PAD_WIDTH,
        "pad_height": DEFAULT_PAD_HEIGHT,
        "combine_threshold": DEFAULT_COMBINE_THRESHOLD,
        "dilate": DEFAULT_DILATE,
        "kernel_size": DEFAULT_KERNEL_SIZE,
        "output_format": None,
        "input_dir": "input",
        "output_dir": "output"
    }
    for key in final_config.keys():
        if key in yaml_config:
            final_config[key] = yaml_config[key]
    if "phrases" in yaml_config:
        final_config["texts"] = yaml_config["phrases"]
    if cli_args.texts != DEFAULT_PHRASES:
        final_config["texts"] = cli_args.texts
    if cli_args.tesseract_cmd is not None:
        final_config["tesseract_cmd"] = cli_args.tesseract_cmd
    if cli_args.debug:
        final_config["debug"] = True
    if cli_args.max_workers != DEFAULT_MAX_WORKERS:
        final_config["max_workers"] = cli_args.max_workers
    if cli_args.pad_width != DEFAULT_PAD_WIDTH:
        final_config["pad_width"] = cli_args.pad_width
    if cli_args.pad_height != DEFAULT_PAD_HEIGHT:
        final_config["pad_height"] = cli_args.pad_height
    if cli_args.combine_threshold != DEFAULT_COMBINE_THRESHOLD:
        final_config["combine_threshold"] = cli_args.combine_threshold
    if cli_args.dilate != DEFAULT_DILATE:
        final_config["dilate"] = cli_args.dilate
    if cli_args.kernel_size != DEFAULT_KERNEL_SIZE:
        final_config["kernel_size"] = cli_args.kernel_size
    if cli_args.output_format is not None:
        final_config["output_format"] = cli_args.output_format
    if cli_args.input_dir != "input":
        final_config["input_dir"] = cli_args.input_dir
    if cli_args.output_dir != "output":
        final_config["output_dir"] = cli_args.output_dir
    return final_config

def main():
    """
    Entry point for the CLI. Parses arguments, loads YAML config if provided,
    merges settings, and processes images to remove specified text.
    """
    parser = argparse.ArgumentParser(description="Remove specified text from images using Tesseract OCR.")
    parser.add_argument("--config-file", type=str, help="Path to a YAML config file.")
    parser.add_argument("--texts", nargs="+", default=DEFAULT_PHRASES)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--pad-width", type=int, default=DEFAULT_PAD_WIDTH)
    parser.add_argument("--pad-height", type=int, default=DEFAULT_PAD_HEIGHT)
    parser.add_argument("--combine-threshold", type=int, default=DEFAULT_COMBINE_THRESHOLD)
    parser.add_argument("--dilate", action="store_true", default=DEFAULT_DILATE)
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument("--output-format", default=None)
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    yaml_config = {}
    if args.config_file and os.path.exists(args.config_file):
        logging.info("Loading config from YAML: %s", args.config_file)
        yaml_config = load_config_from_yaml(args.config_file)
    else:
        if args.config_file:
            logging.warning("Config file not found: %s", args.config_file)
    final_config = merge_config(args, yaml_config)
    logging.getLogger().setLevel(logging.DEBUG if final_config["debug"] else logging.INFO)
    configure_tesseract(final_config["tesseract_cmd"])
    input_dir = final_config["input_dir"]
    output_dir = final_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    all_files = os.listdir(input_dir)
    files = [f for f in all_files if f.lower().endswith(exts)]
    if not files:
        logging.warning("No images found in '%s'.", input_dir)
        return
    total_files = len(files)
    completed = 0
    debug_dir = None
    if final_config["debug"]:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=final_config["max_workers"]) as executor:
            future_to_file = {}
            for f in files:
                input_path = os.path.join(input_dir, f)
                future = executor.submit(
                    remove_phrases,
                    image_path=input_path,
                    phrases=final_config["texts"],
                    tesseract_cmd=final_config["tesseract_cmd"],
                    debug=final_config["debug"],
                    pad_width=final_config["pad_width"],
                    pad_height=final_config["pad_height"],
                    inpaint_radius=DEFAULT_INPAINT_RADIUS,
                    inpaint_method=DEFAULT_INPAINT_METHOD,
                    do_dilate=final_config["dilate"],
                    dilate_kernel_size=final_config["kernel_size"],
                    combine_threshold=final_config["combine_threshold"]
                )
                future_to_file[future] = f
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    logging.error("Error processing '%s': %s", f, e)
                    continue
                if result is None:
                    logging.error("Could not process '%s'.", f)
                else:
                    if final_config["debug"] and isinstance(result, tuple):
                        final_img, dbg_img = result
                    else:
                        final_img, dbg_img = result, None
                    basename, orig_ext = os.path.splitext(f)
                    if final_config["output_format"] is None:
                        out_ext = orig_ext
                    else:
                        out_ext = f".{final_config['output_format'].lower()}"
                    out_fname = basename + out_ext
                    out_path = os.path.join(output_dir, out_fname)
                    cv2.imwrite(out_path, final_img)
                    if dbg_img is not None and debug_dir:
                        debug_fname = f"debug_{basename}{out_ext}"
                        debug_path = os.path.join(debug_dir, debug_fname)
                        cv2.imwrite(debug_path, dbg_img)
                progress_msg = f"Progress: {completed}/{total_files} ({(completed/total_files)*100:.1f}%)"
                print(progress_msg, end='\r')
        print()
        logging.info("Processing complete!")
    except KeyboardInterrupt:
        logging.info("[INFO] KeyboardInterrupt detected. Canceling tasks...")
        executor.shutdown(wait=True, cancel_futures=True)
        sys.exit(1)

if __name__ == "__main__":
    main()