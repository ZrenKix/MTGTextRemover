import os
import sys
import time
import yaml
import cv2
import shutil
import logging
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed

from text_removal.config import (
    DEFAULT_PHRASES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PAD_WIDTH,
    DEFAULT_PAD_HEIGHT,
    DEFAULT_COMBINE_THRESHOLD,
    DEFAULT_DILATE,
    DEFAULT_KERNEL_SIZE
)
from text_removal.remove import remove_phrases, logger as remove_logger
from text_removal.tesseract_utils import configure_tesseract

"""
A CustomTkinter GUI for text removal. Features:
- Single 'Run/Cancel' button that toggles between run state and cancel state.
- Progress bar with percentage text and estimated time remaining.
- Inpainting method/radius and 'original' format options.
- Cancels ongoing tasks on user request.

No inline comments, only docstrings.
"""

DEFAULT_INPAINT_METHOD = cv2.INPAINT_TELEA
DEFAULT_INPAINT_RADIUS = 3
executor = None
canceled = False
is_running = False

def load_yaml_config(file_path):
    """
    Loads YAML data from file_path and returns it as a dict.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml_config(config_data, file_path):
    """
    Saves config_data to a YAML file at file_path.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_data, f)

def validate_tesseract_path(tesseract_cmd):
    """
    Validates Tesseract presence in PATH or at tesseract_cmd. Raises RuntimeError if missing.
    """
    if tesseract_cmd:
        if shutil.which(tesseract_cmd) is None:
            raise RuntimeError(f"Tesseract not found at path: {tesseract_cmd}")
    else:
        if shutil.which("tesseract") is None:
            raise RuntimeError("Tesseract is not installed or not in PATH.")

def merge_defaults():
    """
    Returns default configuration values.
    """
    return {
        "phrases": list(DEFAULT_PHRASES),
        "tesseract_cmd": None,
        "debug": False,
        "max_workers": DEFAULT_MAX_WORKERS,
        "pad_width": DEFAULT_PAD_WIDTH,
        "pad_height": DEFAULT_PAD_HEIGHT,
        "combine_threshold": DEFAULT_COMBINE_THRESHOLD,
        "dilate": DEFAULT_DILATE,
        "kernel_size": DEFAULT_KERNEL_SIZE,
        "output_format": "original",
        "input_dir": "input",
        "output_dir": "output",
        "inpaint_method": "TELEA",
        "inpaint_radius": str(DEFAULT_INPAINT_RADIUS),
        "config_file": ""
    }

def merge_config_into_gui(config_dict, gui_settings):
    """
    Copies config_dict data into gui_settings for display in the GUI.
    """
    gui_settings["config_file"] = config_dict.get("config_file", "")
    gui_settings["phrases"] = config_dict.get("phrases", list(DEFAULT_PHRASES))
    gui_settings["tesseract_cmd"] = config_dict.get("tesseract_cmd", None)
    gui_settings["debug"] = bool(config_dict.get("debug", False))
    gui_settings["max_workers"] = str(config_dict.get("max_workers", DEFAULT_MAX_WORKERS))
    gui_settings["pad_width"] = str(config_dict.get("pad_width", DEFAULT_PAD_WIDTH))
    gui_settings["pad_height"] = str(config_dict.get("pad_height", DEFAULT_PAD_HEIGHT))
    gui_settings["combine_threshold"] = str(config_dict.get("combine_threshold", DEFAULT_COMBINE_THRESHOLD))
    gui_settings["dilate"] = bool(config_dict.get("dilate", DEFAULT_DILATE))
    gui_settings["kernel_size"] = str(config_dict.get("kernel_size", DEFAULT_KERNEL_SIZE))
    gui_settings["output_format"] = config_dict.get("output_format", "original") or "original"
    gui_settings["input_dir"] = config_dict.get("input_dir", "input")
    gui_settings["output_dir"] = config_dict.get("output_dir", "output")
    im = config_dict.get("inpaint_method", "TELEA")
    gui_settings["inpaint_method"] = "NS" if im == "NS" else "TELEA"
    gui_settings["inpaint_radius"] = str(config_dict.get("inpaint_radius", DEFAULT_INPAINT_RADIUS))

def build_config_from_gui(gui_settings):
    """
    Creates a final config dict from gui_settings. 
    'original' is mapped to None for output_format.
    """
    fmt_choice = gui_settings["output_format"]
    if fmt_choice == "original":
        fmt_choice = None
    method_str = gui_settings["inpaint_method"]
    if method_str not in ("TELEA", "NS"):
        method_str = "TELEA"
    return {
        "phrases": gui_settings["phrases"],
        "tesseract_cmd": gui_settings["tesseract_cmd"] or None,
        "debug": gui_settings["debug"],
        "max_workers": int(gui_settings["max_workers"]),
        "pad_width": int(gui_settings["pad_width"]),
        "pad_height": int(gui_settings["pad_height"]),
        "combine_threshold": int(gui_settings["combine_threshold"]),
        "dilate": bool(gui_settings["dilate"]),
        "kernel_size": int(gui_settings["kernel_size"]),
        "output_format": fmt_choice,
        "input_dir": gui_settings["input_dir"],
        "output_dir": gui_settings["output_dir"],
        "config_file": gui_settings["config_file"],
        "inpaint_method": method_str,
        "inpaint_radius": int(gui_settings["inpaint_radius"])
    }

def inpaint_method_to_cv2(method_str):
    """
    Converts a string (TELEA/NS) to cv2.INPAINT_TELEA or cv2.INPAINT_NS.
    """
    if method_str == "NS":
        return cv2.INPAINT_NS
    return cv2.INPAINT_TELEA

def cancel_processing(status_label, run_cancel_button):
    """
    Sets a global canceled flag and updates the UI state to canceled.
    """
    global canceled
    canceled = True
    status_label.configure(text="Canceling...")
    run_cancel_button.configure(text="Run")

def run_removal_in_thread(config_data, progress_bar, progress_label, status_label, run_cancel_button):
    """
    Runs text removal in a background thread, with progress tracking. 
    Cancels if user requests. Toggles 'Run/Cancel' button text appropriately.
    """
    global executor
    global canceled
    global is_running

    try:
        logging.getLogger().setLevel(logging.DEBUG if config_data["debug"] else logging.INFO)
        validate_tesseract_path(config_data["tesseract_cmd"])
        configure_tesseract(config_data["tesseract_cmd"])
        input_dir = config_data["input_dir"]
        output_dir = config_data["output_dir"]
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        if not files:
            messagebox.showwarning("No Files", f"No images found in '{input_dir}'.")
            run_cancel_button.configure(text="Run")
            is_running = False
            return
        os.makedirs(output_dir, exist_ok=True)
        debug_dir = None
        if config_data["debug"]:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

        total_files = len(files)
        progress_bar.set(0.0)
        progress_label.configure(text="0%")
        completed = 0
        start_time = time.time()
        executor = ThreadPoolExecutor(max_workers=config_data["max_workers"])
        futures = {}

        for f in files:
            path_in = os.path.join(input_dir, f)
            fut = executor.submit(
                remove_phrases,
                image_path=path_in,
                phrases=config_data["phrases"],
                tesseract_cmd=config_data["tesseract_cmd"],
                debug=config_data["debug"],
                pad_width=config_data["pad_width"],
                pad_height=config_data["pad_height"],
                inpaint_radius=config_data["inpaint_radius"],
                inpaint_method=inpaint_method_to_cv2(config_data["inpaint_method"]),
                do_dilate=config_data["dilate"],
                dilate_kernel_size=config_data["kernel_size"],
                combine_threshold=config_data["combine_threshold"]
            )
            futures[fut] = f

        for fut in as_completed(futures):
            if canceled:
                break
            completed += 1
            fraction = completed / total_files
            progress_bar.set(fraction)
            progress_label.configure(text=f"{int(fraction*100)}%")
            elapsed = time.time() - start_time
            avg_per_file = elapsed / completed if completed else 0
            remain = (total_files - completed) * avg_per_file
            remain_str = time.strftime("%M:%S", time.gmtime(remain))
            status_label.configure(text=f"Processed {completed}/{total_files} | ~{remain_str} left")

            try:
                result = fut.result()
            except Exception as e:
                remove_logger.error("Error processing '%s': %s", futures[fut], e)
                continue

            if result is None:
                remove_logger.error("Could not process '%s'.", futures[fut])
            else:
                basename, orig_ext = os.path.splitext(futures[fut])
                if config_data["debug"] and isinstance(result, tuple):
                    final_img, dbg_img = result
                else:
                    final_img, dbg_img = result, None
                chosen_fmt = config_data["output_format"]
                if chosen_fmt is None:
                    out_ext = orig_ext
                else:
                    out_ext = f".{chosen_fmt.lower()}"
                out_name = basename + out_ext
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, final_img)
                if dbg_img is not None and debug_dir:
                    dbg_path = os.path.join(debug_dir, "debug_" + out_name)
                    cv2.imwrite(dbg_path, dbg_img)

        executor.shutdown(wait=False, cancel_futures=True)
        if canceled:
            status_label.configure(text="Canceled.")
            messagebox.showinfo("Canceled", "Processing was canceled.")
        else:
            messagebox.showinfo("Done", "Processing Complete!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

    is_running = False
    canceled = False
    run_cancel_button.configure(text="Run")

def start_removal(gui_settings, progress_bar, progress_label, status_label, run_cancel_button):
    """
    Builds final config from gui_settings and launches run_removal_in_thread 
    in a separate thread if not already running. If running, triggers cancel.
    """
    global is_running
    global canceled
    if not is_running:
        is_running = True
        canceled = False
        run_cancel_button.configure(text="Cancel")
        config_data = build_config_from_gui(gui_settings)
        t = threading.Thread(
            target=run_removal_in_thread,
            args=(config_data, progress_bar, progress_label, status_label, run_cancel_button),
            daemon=True
        )
        t.start()
    else:
        cancel_processing(status_label, run_cancel_button)

def refresh_gui_from_settings(gui_settings, basic_tab, adv_tab, phrases_widget):
    """
    Updates the GUI fields from gui_settings.
    """
    basic_tab["config_var"].set(gui_settings.get("config_file", ""))
    basic_tab["input_var"].set(gui_settings.get("input_dir", ""))
    basic_tab["output_var"].set(gui_settings.get("output_dir", ""))
    basic_tab["debug_var"].set(gui_settings.get("debug", False))
    adv_tab["tesseract_var"].set(gui_settings.get("tesseract_cmd", "") or "")
    adv_tab["workers_var"].set(gui_settings.get("max_workers", ""))
    adv_tab["padw_var"].set(gui_settings.get("pad_width", ""))
    adv_tab["padh_var"].set(gui_settings.get("pad_height", ""))
    adv_tab["comb_var"].set(gui_settings.get("combine_threshold", ""))
    adv_tab["dilate_var"].set(gui_settings.get("dilate", False))
    adv_tab["k_size_var"].set(gui_settings.get("kernel_size", ""))
    adv_tab["format_var"].set(gui_settings.get("output_format", "original"))
    adv_tab["method_var"].set("NS" if gui_settings.get("inpaint_method") == "NS" else "TELEA")
    adv_tab["radius_var"].set(gui_settings.get("inpaint_radius", str(DEFAULT_INPAINT_RADIUS)))
    phrases_widget.delete("0.0", "end")
    for phrase in gui_settings.get("phrases", []):
        phrases_widget.insert("end", phrase + "\n")

def update_settings_from_gui(gui_settings, basic_tab, adv_tab, phrases_widget):
    """
    Reads the current GUI values and stores them in gui_settings.
    """
    gui_settings["config_file"] = basic_tab["config_var"].get()
    gui_settings["input_dir"] = basic_tab["input_var"].get()
    gui_settings["output_dir"] = basic_tab["output_var"].get()
    gui_settings["debug"] = bool(basic_tab["debug_var"].get())
    gui_settings["tesseract_cmd"] = adv_tab["tesseract_var"].get() or None
    gui_settings["max_workers"] = adv_tab["workers_var"].get()
    gui_settings["pad_width"] = adv_tab["padw_var"].get()
    gui_settings["pad_height"] = adv_tab["padh_var"].get()
    gui_settings["combine_threshold"] = adv_tab["comb_var"].get()
    gui_settings["dilate"] = bool(adv_tab["dilate_var"].get())
    gui_settings["kernel_size"] = adv_tab["k_size_var"].get()
    gui_settings["output_format"] = adv_tab["format_var"].get() or "original"
    gui_settings["inpaint_method"] = adv_tab["method_var"].get() or "TELEA"
    gui_settings["inpaint_radius"] = adv_tab["radius_var"].get()
    all_text = phrases_widget.get("0.0", "end").strip()
    lines = [line for line in all_text.split("\n") if line.strip()]
    gui_settings["phrases"] = lines

def open_config_file(gui_settings, basic_tab, adv_tab, phrases_widget):
    """
    Opens a YAML config and refreshes the GUI.
    """
    path = filedialog.askopenfilename(
        title="Open Config File",
        filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")]
    )
    if path:
        try:
            data = load_yaml_config(path)
            data["config_file"] = path
            merge_config_into_gui(data, gui_settings)
            refresh_gui_from_settings(gui_settings, basic_tab, adv_tab, phrases_widget)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")

def save_config_as(gui_settings, basic_tab, adv_tab, phrases_widget):
    """
    Prompts for a file path and saves current GUI settings as YAML.
    """
    update_settings_from_gui(gui_settings, basic_tab, adv_tab, phrases_widget)
    path = filedialog.asksaveasfilename(
        title="Save Config As",
        defaultextension=".yaml",
        filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")]
    )
    if path:
        try:
            cfg_data = build_config_from_gui(gui_settings)
            cfg_data["config_file"] = path
            save_yaml_config(cfg_data, path)
            gui_settings["config_file"] = path
            messagebox.showinfo("Saved", f"Config saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def save_config_file(gui_settings, basic_tab, adv_tab, phrases_widget):
    """
    Saves GUI settings to the existing config path or prompts for a new path if none exist.
    """
    update_settings_from_gui(gui_settings, basic_tab, adv_tab, phrases_widget)
    curr_path = gui_settings.get("config_file", "")
    if not curr_path:
        save_config_as(gui_settings, basic_tab, adv_tab, phrases_widget)
        return
    try:
        cfg_data = build_config_from_gui(gui_settings)
        save_yaml_config(cfg_data, curr_path)
        messagebox.showinfo("Saved", f"Config saved to {curr_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def build_gui():
    """
    Builds a CustomTkinter window with:
    - A top bar for Open/Save/Save As.
    - Tabs for Basic, Advanced, and a text box for phrases.
    - One toggle button that says "Run" (switches to "Cancel" while running).
    - Progress bar with percentage text and a status label for time estimation.
    """
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("MTG Text Remover")
    try:
        root.iconbitmap("my_icon.ico")
    except Exception:
        pass

    gui_settings = merge_defaults()

    top_bar = ctk.CTkFrame(root, corner_radius=0)
    top_bar.pack(side="top", fill="x", padx=5, pady=5)

    def open_cb():
        open_config_file(gui_settings, basic_tab, adv_tab, phrases_box)
    def save_cb():
        save_config_file(gui_settings, basic_tab, adv_tab, phrases_box)
    def save_as_cb():
        save_config_as(gui_settings, basic_tab, adv_tab, phrases_box)

    ctk.CTkButton(top_bar, text="Open Config", command=open_cb).pack(side="left", padx=5)
    ctk.CTkButton(top_bar, text="Save Config", command=save_cb).pack(side="left", padx=5)
    ctk.CTkButton(top_bar, text="Save Config As...", command=save_as_cb).pack(side="left", padx=5)

    tabview = ctk.CTkTabview(root, corner_radius=5)
    tabview.pack(fill="both", expand=True, padx=10, pady=10)
    tabview.add("Basic")
    tabview.add("Advanced")
    tabview.add("Phrases")

    basic_frame = tabview.tab("Basic")
    adv_frame = tabview.tab("Advanced")
    phr_frame = tabview.tab("Phrases")

    basic_tab = {}
    adv_tab = {}

    basic_tab["config_var"] = ctk.StringVar()
    basic_tab["input_var"] = ctk.StringVar()
    basic_tab["output_var"] = ctk.StringVar()
    basic_tab["debug_var"] = ctk.BooleanVar()

    ctk.CTkLabel(basic_frame, text="Config File:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(basic_frame, textvariable=basic_tab["config_var"], width=220).grid(row=0, column=1, padx=5, pady=5)

    ctk.CTkLabel(basic_frame, text="Input Dir:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    in_e = ctk.CTkEntry(basic_frame, textvariable=basic_tab["input_var"], width=220)
    in_e.grid(row=1, column=1, padx=5, pady=5)

    def browse_input():
        d = filedialog.askdirectory(title="Select Input Directory")
        if d:
            basic_tab["input_var"].set(d)
    ctk.CTkButton(basic_frame, text="Browse", command=browse_input, width=80).grid(row=1, column=2, padx=5, pady=5)

    ctk.CTkLabel(basic_frame, text="Output Dir:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    out_e = ctk.CTkEntry(basic_frame, textvariable=basic_tab["output_var"], width=220)
    out_e.grid(row=2, column=1, padx=5, pady=5)

    def browse_output():
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            basic_tab["output_var"].set(d)
    ctk.CTkButton(basic_frame, text="Browse", command=browse_output, width=80).grid(row=2, column=2, padx=5, pady=5)

    ctk.CTkCheckBox(basic_frame, text="Debug Mode", variable=basic_tab["debug_var"]).grid(row=3, column=1, sticky="w", padx=5, pady=5)

    adv_tab["tesseract_var"] = ctk.StringVar()
    adv_tab["workers_var"] = ctk.StringVar()
    adv_tab["padw_var"] = ctk.StringVar()
    adv_tab["padh_var"] = ctk.StringVar()
    adv_tab["comb_var"] = ctk.StringVar()
    adv_tab["dilate_var"] = ctk.BooleanVar()
    adv_tab["k_size_var"] = ctk.StringVar()
    adv_tab["format_var"] = ctk.StringVar()
    adv_tab["method_var"] = ctk.StringVar(value="TELEA")
    adv_tab["radius_var"] = ctk.StringVar()

    ctk.CTkLabel(adv_frame, text="Tesseract Cmd:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["tesseract_var"], width=150).grid(row=0, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Max Workers:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["workers_var"], width=80).grid(row=1, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Pad Width:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["padw_var"], width=80).grid(row=2, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Pad Height:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["padh_var"], width=80).grid(row=3, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Combine Threshold:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["comb_var"], width=80).grid(row=4, column=1, padx=5, pady=5)

    ctk.CTkCheckBox(adv_frame, text="Dilate Mask", variable=adv_tab["dilate_var"]).grid(row=5, column=1, sticky="w", padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Kernel Size:").grid(row=6, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["k_size_var"], width=80).grid(row=6, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Output Format:").grid(row=7, column=0, sticky="e", padx=5, pady=5)
    fmt_values = ["original", "png", "jpg", "jpeg", "bmp", "webp"]
    ctk.CTkOptionMenu(adv_frame, values=fmt_values, variable=adv_tab["format_var"], width=100).grid(row=7, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Inpaint Method:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
    method_values = ["TELEA", "NS"]
    ctk.CTkOptionMenu(adv_frame, values=method_values, variable=adv_tab["method_var"], width=80).grid(row=8, column=1, padx=5, pady=5)

    ctk.CTkLabel(adv_frame, text="Inpaint Radius:").grid(row=9, column=0, sticky="e", padx=5, pady=5)
    ctk.CTkEntry(adv_frame, textvariable=adv_tab["radius_var"], width=80).grid(row=9, column=1, padx=5, pady=5)

    phrases_box = ctk.CTkTextbox(phr_frame, width=400, height=200)
    phrases_box.pack(fill="both", expand=True, padx=10, pady=10)

    progress_bar = ctk.CTkProgressBar(root)
    progress_bar.set(0.0)
    progress_bar.pack(fill="x", padx=10, pady=5)

    progress_label = ctk.CTkLabel(root, text="0%")
    progress_label.pack()

    status_label = ctk.CTkLabel(root, text="Ready...")
    status_label.pack(pady=5)

    def run_cancel_callback():
        update_settings_from_gui(gui_settings, basic_tab, adv_tab, phrases_box)
        start_removal(gui_settings, progress_bar, progress_label, status_label, run_cancel_button)

    run_cancel_button = ctk.CTkButton(root, text="Run", command=run_cancel_callback, width=120)
    run_cancel_button.pack(pady=5)

    refresh_gui_from_settings(gui_settings, basic_tab, adv_tab, phrases_box)
    return root, gui_settings, basic_tab, adv_tab, phrases_box

def main():
    """
    Initializes logging, creates the CustomTkinter window, and starts the event loop.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    root, gui_settings, basic_tab, adv_tab, phrases_box = build_gui()
    root.mainloop()

if __name__ == "__main__":
    main()