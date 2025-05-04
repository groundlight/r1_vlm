import json
import os

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from datasets import load_dataset
from PIL import Image

OUTPUT_FILE = "zoom_demos.jsonl"


# --- Image Resizing Logic (copied from text_vqa_tool_use_r1.py) ---
def resize_image(image):
    """
    Resizes the image if its longer side exceeds 1024 pixels,
    maintaining aspect ratio.
    """
    # if the longer side is greater than 1024, resize it so the longer side is 1024
    if image.mode == "L":  # Handle grayscale images
        image = image.convert("RGB")

    if image.size[0] > 1024 or image.size[1] > 1024:
        longer_side = max(image.size[0], image.size[1])
        image = image.resize(
            (
                int(1024 * image.size[0] / longer_side),
                int(1024 * image.size[1] / longer_side),
            ),
            Image.Resampling.LANCZOS,  # Use LANCZOS for better quality
        )
    return image


# --- Global State (simpler for this script) ---
current_example = None
original_image = None
original_size = None
displayed_image = None
displayed_size = None
selected_keypoint_original = None
selected_keypoint_display = None
marker_plot = None
dataset_iterator = None
demos_saved_count = 0

# --- Matplotlib UI Elements ---
fig, ax = plt.subplots(figsize=(10, 8))  # Make figure a bit larger
plt.subplots_adjust(bottom=0.25)  # More space for controls and title
ax.axis("off")

ax_save = plt.axes([0.81, 0.05, 0.15, 0.075])
btn_save = widgets.Button(ax_save, "Save & Next")
btn_save.active = False  # Initially disabled

ax_undo = plt.axes([0.65, 0.05, 0.15, 0.075])
btn_undo = widgets.Button(ax_undo, "Undo Click")
btn_undo.active = False  # Initially disabled

ax_skip = plt.axes([0.49, 0.05, 0.15, 0.075])
btn_skip = widgets.Button(ax_skip, "Skip")

# Text to show saved count
ax_count_text = plt.axes([0.05, 0.05, 0.4, 0.075])
ax_count_text.axis("off")
count_text_obj = ax_count_text.text(
    0.5,
    0.5,
    f"Saved: {demos_saved_count}",
    ha="center",
    va="center",
    transform=ax_count_text.transAxes,
)


# --- Core Logic Functions ---
def update_saved_count_display():
    """Updates the saved count text on the plot."""
    global demos_saved_count
    count_text_obj.set_text(f"Saved: {demos_saved_count}")
    plt.draw()


def remove_marker():
    """Removes the keypoint marker from the plot if it exists."""
    global marker_plot
    if marker_plot:
        try:
            marker_plot.remove()
        except ValueError:  # May have already been removed if plot cleared
            pass
        marker_plot = None


def display_current_example():
    """Displays the current image and question."""
    global displayed_image, displayed_size, marker_plot
    if current_example is None:
        return

    remove_marker()  # Clear marker from previous example

    # Store original and prepare display image
    original_image_pil = current_example["image"]
    if original_image_pil.mode == "L":  # Ensure RGB
        original_image_pil = original_image_pil.convert("RGB")

    displayed_image = resize_image(original_image_pil.copy())  # Use resized for display
    displayed_size = displayed_image.size

    ax.cla()  # Clear axes before drawing new image
    ax.imshow(displayed_image)
    question_text = (
        f"Q ({current_example['question_id']}): {current_example['question']}"
    )
    ax.set_title(question_text, wrap=True, fontsize=10)
    ax.axis("off")
    plt.draw()


def load_next_example():
    """Loads the next example from the dataset iterator and updates the UI."""
    global current_example, original_image, original_size
    global selected_keypoint_original, selected_keypoint_display, marker_plot
    global dataset_iterator

    if dataset_iterator is None:
        print("Error: Dataset iterator not initialized.")
        plt.close(fig)
        return

    try:
        current_example = next(dataset_iterator)
        original_image = current_example["image"]
        if original_image.mode == "L":  # Ensure RGB for original too
            original_image = original_image.convert("RGB")
        original_size = original_image.size

        # Reset state for the new example
        selected_keypoint_original = None
        selected_keypoint_display = None
        remove_marker()  # Explicitly remove marker here too

        display_current_example()

        btn_save.active = False
        btn_undo.active = False
        plt.draw()  # Update button states visually

    except StopIteration:
        print("Finished processing all examples in the dataset split.")
        plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred loading the next example: {e}")
        plt.close(fig)


def on_click(event):
    """Handles mouse clicks on the image axes to select a keypoint."""
    global selected_keypoint_original, selected_keypoint_display, marker_plot
    global original_size, displayed_size

    if event.inaxes != ax or original_size is None or displayed_size is None:
        return  # Click outside image axes or data not loaded

    # Get click coordinates relative to displayed image
    x_disp, y_disp = int(event.xdata), int(event.ydata)

    # Map coordinates back to original image dimensions
    scale_w = original_size[0] / displayed_size[0]
    scale_h = original_size[1] / displayed_size[1]
    x_orig = int(round(x_disp * scale_w))
    y_orig = int(round(y_disp * scale_h))

    # Clamp coordinates to ensure they are within the original image bounds
    x_orig = max(0, min(original_size[0] - 1, x_orig))
    y_orig = max(0, min(original_size[1] - 1, y_orig))

    selected_keypoint_original = [x_orig, y_orig]
    selected_keypoint_display = [x_disp, y_disp]

    # Update marker
    remove_marker()
    (marker_plot,) = ax.plot(x_disp, y_disp, "r+", markersize=12, markeredgewidth=2)

    # Enable Save and Undo buttons
    btn_save.active = True
    btn_undo.active = True
    plt.draw()


def save_callback(event):
    """Saves the current example's info and keypoint, then loads the next."""
    global demos_saved_count
    if current_example is None or selected_keypoint_original is None:
        print("Warning: Cannot save - no example loaded or keypoint selected.")
        return

    data = {
        "image_id": current_example["image_id"],
        "question_id": current_example["question_id"],
        "keypoint": selected_keypoint_original,
    }

    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
        print(f"Saved: QID {data['question_id']} -> {data['keypoint']}")
        demos_saved_count += 1
        update_saved_count_display()
        load_next_example()
    except IOError as e:
        print(f"Error saving data to {OUTPUT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during save: {e}")


def undo_callback(event):
    """Clears the selected keypoint and removes the marker."""
    global selected_keypoint_original, selected_keypoint_display
    remove_marker()
    selected_keypoint_original = None
    selected_keypoint_display = None
    btn_save.active = False
    btn_undo.active = False
    plt.draw()


def skip_callback(event):
    """Loads the next example without saving."""
    print("Skipped.")
    load_next_example()


# --- Initialization and Execution ---
def main():
    global dataset_iterator, demos_saved_count

    print("Loading TextVQA 'train' split...")
    try:
        dataset = load_dataset("lmms-lab/textvqa", split="train")
        dataset_iterator = iter(dataset)
        print("Dataset loaded.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Count existing demos if file exists
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                demos_saved_count = sum(1 for line in f)
            print(
                f"Found {demos_saved_count} existing demonstrations in {OUTPUT_FILE}."
            )
        except Exception as e:
            print(
                f"Warning: Could not read existing demo count from {OUTPUT_FILE}: {e}"
            )

    # Connect callbacks
    btn_save.on_clicked(save_callback)
    btn_undo.on_clicked(undo_callback)
    btn_skip.on_clicked(skip_callback)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Load the first example and show the plot
    print("Loading first example...")
    update_saved_count_display()  # Show initial count
    load_next_example()

    if current_example:  # Only show plot if first load succeeded
        print("Starting labeling interface. Close the plot window to exit.")
        plt.show()
    else:
        print("Could not load the first example. Exiting.")


if __name__ == "__main__":
    main()
