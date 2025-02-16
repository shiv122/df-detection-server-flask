import os ,uuid
import boto3
from flask import Flask, jsonify, request
from detector import detect
from werkzeug.utils import secure_filename
import json
app = Flask(__name__)


S3_BUCKET = "df-detection"
S3_REGION = "blr1"
S3_ENDPOINT = f"https://{S3_REGION}.digitaloceanspaces.com"
S3_ACCESS_KEY = "DO801M42WP46ZDZ9EGMR"
S3_SECRET_KEY = "h2DQGOY75hpgbQrNuZnNPbQ3pfWvr8bc3QNV09JsLxM"

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"


@app.route('/detect', methods = ['GET', 'POST'])
def home():
    
    #check if header has secret key
    if(request.headers.get('secret') != 'secret'):
        return jsonify({'error': 'invalid secret key'})

    if(request.method == 'GET'):
        image =  detect('static/images/input/1.jpg')
        return jsonify({
            'image': image
        })
        
    if(request.method == 'POST'):
        #return error if no image is sent
        if('image' not in request.files):
            return jsonify({'error': 'no image sent'})
        
        UPLOAD_FOLDER = 'static/images/input'
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        
        #upload image
        file = request.files['image']
        file_name = uuid.uuid4().hex
        file_ext = file.filename.split('.')[-1]
        uploaded_path = os.path.join(UPLOAD_FOLDER,file_name+'.'+file_ext)
        uploaded_file = file.save(uploaded_path)
        
        output = detect(uploaded_path)
        
        return jsonify({'data': output})
     



@app.route('/detect-new', methods=['POST'])
def detect_new():
    try:
        if request.headers.get('secret') != 'secret':
            return jsonify({'error': 'invalid secret key'})

        if request.method == 'GET':
            try:
                image = detect('static/images/input/1.jpg')
                return jsonify({'image': image})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        if request.method == 'POST':
            if 'image' not in request.files:
                return jsonify({'error': 'no image sent'})
            
            file = request.files['image']
            file_ext = file.filename.split('.')[-1]
            file_name = secure_filename(uuid.uuid4().hex + '.' + file_ext)
            uploaded_path = os.path.join('static/images/input', file_name)
            
            try:
                file.save(uploaded_path)
            except Exception as e:
                return jsonify({'error': f'Failed to save file: {str(e)}'})
            
            try:
                output = detect(uploaded_path)
                output_data = json.loads(output)
                output_image_name = output_data.get('image')
                output_image_path = output_image_name
            except Exception as e:
                return jsonify({'error': f'Error during detection: {str(e)}'})
            
           #  try:
           #     s3_key_input = f'input/{file_name}'
           #     s3_client.upload_file(uploaded_path, S3_BUCKET, s3_key_input, ExtraArgs={'ACL': 'public-read'})
           #     input_file_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{s3_key_input}"
           # except Exception as e:
           #     return jsonify({'error': f'Failed to upload input file to S3: {str(e)}'})
            
            try:
                s3_key_output = f'output/{output_image_name}'
                s3_client.upload_file(output_image_path, S3_BUCKET, s3_key_output, ExtraArgs={'ACL': 'public-read'})
                output_file_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{s3_key_output}"
            except Exception as e:
                return jsonify({'error': f'Failed to upload output file to S3: {str(e)}'})
            
            try:
                os.remove(uploaded_path)
                os.remove(output_image_path)
            except Exception as e:
                return jsonify({'error': f'Failed to delete local files: {str(e)}'})
            
            output_data['image'] = output_file_url
            return jsonify(output_data)
    except Exception as e:
        return jsonify({'error': str(e)})        



@app.route('/output-images',methods=['GET'])
def images():
    if(request.headers.get('secret') != 'secret'):
        return jsonify({'error': 'invalid secret key'})
    
    path = 'static/images/output'
    dir_list = os.listdir(path)     
    return jsonify(dir_list)


@app.route('/input-images',methods=['GET'])
def input_images():
    if(request.headers.get('secret') != 'secret'):
        return jsonify({'error': 'invalid secret key'})
    
    path = 'static/images/input'
    dir_list = os.listdir(path)     
    return jsonify(dir_list)

@app.route('/test',methods=['GET'])
def test():
    return jsonify({'res':'test'})





if __name__ == "__main__":
    app.run(host='0.0.0.0')
