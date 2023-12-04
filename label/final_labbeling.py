import os
import csv
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


def faceBox(faceNet, cv_frame):
    frameHeight = cv_frame.shape[0]
    frameWidth = cv_frame.shape[1]
    blob = cv2.dnn.blobFromImage(cv_frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
    return cv_frame, bboxs


def clear_frame(tk_frame):
    for widget in tk_frame.winfo_children():
        widget.destroy()


def get_user_input(root, tk_frame, image_path, input_var, faceNet, ageNet, genderNet):
    clear_frame(tk_frame)

    cv_frame = cv2.imread(image_path)
    cv_frame, bboxs = faceBox(faceNet, cv_frame)

    # Initialize default values for gender and age
    gender, age = "", ""

    # If a face is detected, use the model to predict gender and age
    if bboxs:
        bbox = bboxs[0]
        face = cv_frame[max(0, bbox[1]):min(bbox[3], cv_frame.shape[0]),
               max(0, bbox[0]):min(bbox[2], cv_frame.shape[1])]

        # Ensure the face region is not empty
        if face.size != 0:
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            agePred = ageList[agePreds[0].argmax()]

            age = '1' if agePred <= '(25-32)' else ('2' if agePred <= '(48-53)' else '3')

    # Convert cv_frame to a PIL Image and display in Tkinter window
    img = Image.fromarray(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB))
    img = img.resize((250, 250), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(tk_frame, image=photo)
    label.image = photo  # Keep a reference to avoid garbage collection
    label.pack()

    # Entry widgets for gender and age
    entry_gender = tk.Entry(tk_frame)
    entry_gender.insert(0, gender)
    entry_age = tk.Entry(tk_frame)
    entry_age.insert(0, age)

    # Pack entry widgets and labels
    tk.Label(tk_frame, text="Enter gender (m/f):").pack()
    entry_gender.pack()
    tk.Label(tk_frame, text="Enter age category (1 for young, 2 for middle_age, 3 for senior):").pack()
    entry_age.pack()

    # Button to submit input
    submit_button = tk.Button(tk_frame, text="Submit", command=lambda: input_var.set(1))
    submit_button.pack()

    # Bind Enter key to submit_button
    root.bind('<Return>', lambda event=None: submit_button.invoke())

    # Wait for input
    root.wait_variable(input_var)

    # Unbind Enter key after submission
    root.unbind('<Return>')

    return entry_gender.get().lower(), entry_age.get()


def save_progress(last_processed_path):
    with open("progress.txt", "w") as file:
        file.write(last_processed_path)

def load_progress():
    try:
        with open("progress.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['m', 'f']

root = tk.Tk()
root.title("Image Annotation")
input_var = tk.IntVar()
tk_frame = tk.Frame(root)
tk_frame.pack()

main_directory = "../dataset/datasplit/"
csv_file_path = "labels.csv"

# Load the last processed image path
last_processed = load_progress()
resume = False

# File operations
csvfile = open(csv_file_path, 'a', newline='')
csvwriter = csv.writer(csvfile)

if not last_processed:
    csvwriter.writerow(['Image Path', 'Gender', 'Age Category'])

for category_folder in os.listdir(os.path.join(main_directory, 'test')):
    category_folder_path = os.path.join(main_directory, 'test', category_folder)
    for image_file in os.listdir(category_folder_path):
        image_path = os.path.join(category_folder_path, image_file)
        if last_processed and not resume:
            if image_path == last_processed:
                resume = True
            continue
        gender, age = get_user_input(root, tk_frame, image_path, input_var, faceNet, ageNet, genderNet)
        if gender and age:
            csvwriter.writerow([image_path, gender, age])
            csvfile.flush()
            save_progress(image_path)  # Save progress after each successful entry

csvfile.close()
root.mainloop()
