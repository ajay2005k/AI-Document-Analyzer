from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from sample1 import DocumentQA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize QA system
qa_system = DocumentQA()
current_file = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            qa_system.load_document(filepath)
            current_file = filename
            return jsonify({'success': f'File {filename} uploaded and processed successfully'})
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

@app.route('/ask', methods=['POST'])
def ask_question():
    if not current_file:
        return jsonify({'error': 'Please upload a document first'})
    
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'})
    
    try:
        result = qa_system.answer_question(question)
        return jsonify({
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources']
        })
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
