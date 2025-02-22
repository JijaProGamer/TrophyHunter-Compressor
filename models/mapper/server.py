from flask import Flask, send_from_directory, request, jsonify
import os
import base64
import shutil
import json

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
    if not images:
        return jsonify({"image": None})

    return jsonify({"image": images[0]})

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    annotation_text = request.form.get("annotations")
    filename = request.form.get("filename")

    if not filename or not annotation_text:
        return "Missing data", 400

    color_grid = json.loads(annotation_text)
    
    annotation_file_path = os.path.join(ANNOTATED_TEXT_FOLDER, filename.rsplit(".", 1)[0] + ".json")
    
    with open(annotation_file_path, "w") as txt_file:
        json.dump(color_grid, txt_file)

    return "Annotation saved!"

@app.route("/unannotated_images/<filename>")
def serve_unannotated_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()