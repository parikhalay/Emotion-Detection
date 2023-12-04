import os
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def clear_frame(frame):
    """ Clear all widgets from a frame """
    for widget in frame.winfo_children():
        widget.destroy()

def get_user_input(root, frame, image_path, input_var):
    # Clear previous widgets
    clear_frame(frame)

    # Display the image
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(frame, image=photo)
    label.image = photo
    label.pack()

    # Entry widgets for gender and age
    entry_gender = tk.Entry(frame)
    entry_age = tk.Entry(frame)

    # Pack entry widgets and labels
    tk.Label(frame, text="Enter gender (m/f):").pack()
    entry_gender.pack()
    tk.Label(frame, text="Enter age category (1 for young, 2 for middle_age, 3 for senior):").pack()
    entry_age.pack()

    # Button to submit input
    submit_button = tk.Button(frame, text="Submit", command=lambda: input_var.set(1))
    submit_button.pack()

    # Bind Enter key to submit button
    root.bind('<Return>', lambda event=None: submit_button.invoke())

    # Wait for input
    root.wait_variable(input_var)

    # Unbind Enter key after submission
    root.unbind('<Return>')

    # Get values from entries
    gender = entry_gender.get().lower()
    age = entry_age.get()

    # Validation
    if gender not in ['m', 'f'] or age not in ['1', '2', '3']:
        raise ValueError("Invalid input")

    return gender, age


def save_progress(last_processed):
    with open("progress.txt", "w") as file:
        file.write(last_processed)

def load_progress():
    try:
        with open("progress.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

# Initialize main window
root = tk.Tk()
root.title("Image Annotation")
input_var = tk.IntVar()

# Frame to hold widgets
frame = tk.Frame(root)
frame.pack()

# Path to the main directory
main_directory = "dataset/datasplit/"

# Output CSV file
csv_file_path = "labels.csv"

# Load progress
last_processed = load_progress()
resume = False

# Open CSV file for appending or writing
csvfile = open(csv_file_path, 'a' if last_processed else 'w', newline='')
csvwriter = csv.writer(csvfile)

if not last_processed:
    # Write header if starting fresh
    csvwriter.writerow(['Image Path', 'Gender', 'Age Category'])

# Iterate through each category folder in the train directory
for category_folder in os.listdir(os.path.join(main_directory, 'train')):
    category_folder_path = os.path.join(main_directory, 'train', category_folder)

    # Iterate through each image in the category folder
    for image_file in os.listdir(category_folder_path):
        # Get the full path of the image
        image_path = os.path.join(category_folder_path, image_file)

        if last_processed and not resume:
            if image_path == last_processed:
                resume = True
            continue

        # Get user input for gender and age with error handling
        gender, age = get_user_input(root, frame, image_path, input_var)
        if gender is None or age is None:
            continue

        # Write the information to the CSV file
        csvwriter.writerow([image_path, gender, age])
        csvfile.flush()
        os.fsync(csvfile.fileno())
        save_progress(image_path)

# Close the file and main window
csvfile.close()
root.mainloop()