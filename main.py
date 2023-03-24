
import os ,uuid
from flask import Flask, jsonify, request
from detector import detect
  
# creating a Flask app
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
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
        
  
  

@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):
  
    return jsonify({'data': num**2})
  
  
# driver function
if __name__ == '__main__':
  
    app.run(debug = True)
    

app.run()