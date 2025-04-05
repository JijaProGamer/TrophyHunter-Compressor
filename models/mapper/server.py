from flask import Flask, send_from_directory, request, jsonify
import os
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "./unannotated_images/"
ANNOTATED_IMAGES_FOLDER = "./annotated_images/images/"
ANNOTATED_TEXT_FOLDER = "./annotated_images/annotations/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_TEXT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return "No files uploaded", 400

    files = request.files.getlist("files")
    saved_files = []

    for file in files:
        if file.filename == "":
            continue
        
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        saved_files.append(filename)

    return jsonify({"message": "Files uploaded successfully", "files": saved_files})

@app.route("/get_next_image", methods=["GET"])
def get_next_image():
    images = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if images:
        return jsonify({"image": images[0]})

    return jsonify({"image": None})

@app.route("/get_annotated_images", methods=["GET"])
def get_annotated_images():
    images = [f for f in os.listdir(ANNOTATED_IMAGES_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    valid_images = {}

    for image in images:
        annotation_file_path = os.path.join(ANNOTATED_TEXT_FOLDER, image.rsplit(".", 1)[0] + ".txt")

        if os.path.exists(annotation_file_path):
            valid_images[len(valid_images)] = image
        else:
            shutil.move(os.path.join(ANNOTATED_IMAGES_FOLDER, image), os.path.join(UPLOAD_FOLDER, image))

    return jsonify({"images": valid_images})



@app.route("/get_annotation/<filename>", methods=["GET"])
def get_annotation(filename):
    annotation_file_path = os.path.join(ANNOTATED_TEXT_FOLDER, filename.rsplit(".", 1)[0] + ".txt")
    
    if not os.path.exists(annotation_file_path):
        return jsonify({"annotations": None})
    
    with open(annotation_file_path, "r") as file:
        annotation_text = file.read()

    return jsonify({"annotations": annotation_text})

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    annotation_text = request.form.get("annotations")
    filename = request.form.get("filename")

    if not filename or not annotation_text:
        return "Missing data", 400

    annotation_file_path = os.path.join(ANNOTATED_TEXT_FOLDER, filename.rsplit(".", 1)[0] + ".txt")
    with open(annotation_file_path, "w") as json_file:
        json_file.write(annotation_text)

    original_image_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(original_image_path):
        shutil.move(original_image_path, os.path.join(ANNOTATED_IMAGES_FOLDER, filename))

    return "Annotation saved!"

@app.route("/unannotated_images/<filename>")
def serve_unannotated_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/annotated_images/<filename>")
def serve_annotated_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
