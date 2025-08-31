from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from gemini_chat import GeminiChat
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'nervous-about-the-up-coming-semester'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize the chatbot
document_folder = "documents"
if not os.path.exists(document_folder):
    os.makedirs(document_folder)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDGJ8R2jU2lRRbcjN0rhvzUvTgK4v6DLwo')
bot = GeminiChat(document_folder, api_key=GEMINI_API_KEY)


@app.route('/')
def index():
    doc_count = len(bot.processor.documents) if bot and hasattr(bot.processor, 'documents') else 0
    return render_template('index.html', doc_count=doc_count)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Debug: Print file info
        print(f"📁 Uploading file: {filename}")
        print(f"📁 Destination: {filepath}")

        # Save the file
        file.save(filepath)

        # Debug: Check if file exists
        if os.path.exists(filepath):
            print(f"✅ File saved successfully: {filepath}")
        else:
            print(f"❌ File failed to save: {filepath}")
            return jsonify({'status': 'error', 'message': 'File failed to save'})

        # Reload documents to include the new file
        try:
            print("🔄 Reloading documents...")
            bot.processor.load_documents(document_folder)
            print(f"✅ Documents reloaded. Total: {len(bot.processor.documents)}")
            print(f"✅ Chunks created: {len(bot.processor.document_chunks)}")

            return jsonify({
                'status': 'success',
                'message': f'File {filename} uploaded successfully!',
                'new_count': len(bot.processor.documents)
            })
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'})

    return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload TXT, PDF, or DOCX files.'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        response = bot.generate_response(user_message)
        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        return jsonify({'response': f'I encountered an error: {str(e)}', 'status': 'error'})


@app.route('/get_doc_count')
def get_doc_count():
    if not bot:
        return jsonify({'count': 0})
    return jsonify({
        'count': len(bot.processor.documents),
        'chunks': len(bot.processor.document_chunks) if hasattr(bot.processor, 'document_chunks') else 0
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)