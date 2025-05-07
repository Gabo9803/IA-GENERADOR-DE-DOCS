from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
import openai
import os
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime
import re
import json
import sqlite3
import bcrypt

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Clave secreta para sesiones

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")

# Configurar cliente OpenAI
client = openai.OpenAI(api_key=openai_api_key)

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Modelo de usuario
class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    
    # Tabla de usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Tabla de perfiles de estilo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS style_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            font_name TEXT,
            font_size INTEGER,
            tone TEXT,
            margins TEXT,
            structure TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Ejecutar la inicialización de la base de datos al iniciar la aplicación
init_db()

# Cache para respuestas, sesiones y vistas previas HTML
response_cache = {}
session_messages = {}  # Almacena mensajes por sesión
html_previews = {}  # Almacena contenido HTML temporal para vistas previas

def get_db_connection():
    conn = sqlite3.connect('profiles.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_default_style_profile():
    """Devuelve un estilo por defecto para la generación de documentos."""
    return {
        'font_name': 'Helvetica',
        'font_size': 12,
        'tone': 'neutral',
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'structure': ['paragraphs']
    }

def parse_markdown_to_reportlab(text, style_profile=None):
    """Convierte Markdown a elementos de reportlab con soporte para estilos personalizados."""
    styles = getSampleStyleSheet()
    style_profile = style_profile or get_default_style_profile()
    font_name = style_profile['font_name']
    font_size = style_profile['font_size']
    body_style = ParagraphStyle(
        name='Body', fontSize=font_size, leading=font_size * 1.33, spaceAfter=8, fontName=font_name
    )
    bold_style = ParagraphStyle(
        name='Bold', fontSize=font_size, leading=font_size * 1.33, spaceAfter=8, fontName=f'{font_name}-Bold'
    )
    elements = []

    # Soporte para tablas Markdown
    table_pattern = re.compile(r'\|(.+?)\|\n\|[-|:\s]+\|\n((?:\|.+?\|(?:\n|$))+)')
    text_parts = table_pattern.split(text)
    
    for i, part in enumerate(text_parts):
        if i % 3 == 2:  # Contenido de la tabla
            rows = [row.strip().split('|')[1:-1] for row in part.strip().split('\n')]
            table_data = [[cell.strip() for cell in row] for row in rows]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), font_size),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))
        elif i % 3 != 1:  # Texto normal
            lines = part.split('\n')
            for line in lines:
                line = line.replace('\n', '<br />')
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                if line.strip().startswith('- '):
                    line = f'• {line[2:]}'
                    elements.append(Paragraph(line, body_style))
                else:
                    elements.append(Paragraph(line, bold_style if '**' in line else body_style))
                elements.append(Spacer(1, 0.1 * inch))
    
    return elements

def analyze_document(file):
    """Analiza un documento subido para extraer estilo, formato y tipografía."""
    style_profile = {
        'font_name': 'Helvetica',
        'font_size': 12,
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'tone': 'neutral',
        'structure': []
    }
    
    content = ""
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content += page.extract_text() or ""
                chars = page.chars
                if chars:
                    style_profile['font_name'] = chars[0].get('fontname', 'Helvetica').split('-')[0]
                    style_profile['font_size'] = round(chars[0].get('size', 12))
                style_profile['margins'] = {
                    'top': page.bbox[3] - page.chars[-1]['y1'] if page.chars else 1,
                    'bottom': page.chars[0]['y0'] - page.bbox[1] if page.chars else 1,
                    'left': page.chars[0]['x0'] - page.bbox[0] if page.chars else 1,
                    'right': page.bbox[2] - page.chars[-1]['x1'] if page.chars else 1
                }
    elif file.filename.endswith('.txt'):
        content = file.read().decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analiza el tono (formal, informal, técnico) y la estructura (encabezados, párrafos, listas, tablas) del siguiente texto. Devuelve un JSON con 'tone' y 'structure'."},
                {"role": "user", "content": content[:4000]}
            ],
            max_tokens=500
        )
        analysis = json.loads(response.choices[0].message.content)
        style_profile['tone'] = analysis.get('tone', 'neutral')
        style_profile['structure'] = analysis.get('structure', [])
    except Exception as e:
        print(f"Error en análisis de tono: {e}")
    
    return style_profile, content

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, password FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            login_user(User(user['id'], user['email']))
            return redirect(url_for('index'))
        else:
            flash('Correo o contraseña incorrectos', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registro exitoso. Por favor, inicia sesión.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('El correo ya está registrado', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/generate', methods=['POST'])
@login_required
def generate_document():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        doc_type = data.get('doc_type', 'general')
        tone = data.get('tone', 'neutral')
        length = data.get('length', 'medium')
        language = data.get('language', 'es')
        style_profile_id = data.get('style_profile_id', None)
        message_history = data.get('message_history', [])

        if not prompt:
            return jsonify({'error': 'El prompt está vacío'}), 400

        cache_key = f"{current_user.id}:{session_id}:{prompt}:{doc_type}:{tone}:{length}:{language}:{style_profile_id}"
        if cache_key in response_cache:
            return jsonify({
                'document': response_cache[cache_key],
                'content_type': 'html' if doc_type == 'html' or 'html' in prompt.lower() else 'pdf'
            })

        if session_id not in session_messages:
            session_messages[session_id] = []

        style_profile = get_style_profile(style_profile_id)
        
        # Detectar si el prompt requiere una explicación estructurada
        is_explanatory = re.search(r'^(¿Quién es|¿Qué es|explicar|detallar)\b', prompt, re.IGNORECASE) is not None
        
        # Detectar si el prompt solicita HTML
        is_html = doc_type == 'html' or re.search(r'\b(html|página web|sitio web)\b', prompt, re.IGNORECASE) is not None
        
        if is_explanatory:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown con una estructura clara para explicaciones.
            - Tipo de documento: {doc_type} (e.g., explicación, biografía, informe).
            - Tono: {tone} (formal, informal, técnico).
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - Usa la siguiente estructura en Markdown:
              ```markdown
              # [Tema o Nombre]
              
              ## Introducción
              [Párrafo breve presentando el tema o persona.]
              
              ## Detalles Principales
              - [Clave 1]: [Valor o descripción]
              - [Clave 2]: [Valor o descripción]
              - ...
              
              ## Contexto Adicional
              [Párrafos detallando información relevante, como antecedentes, logros, o impacto.]
              
              ## Conclusión
              [Resumen de la relevancia o importancia del tema.]
              ```
            - Usa encabezados (#, ##), listas (-), negritas (**), y tablas (|...|) cuando sea apropiado.
            - Estilo: {style_profile}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia en la conversación.
            """
        elif is_html:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera páginas web en formato HTML con CSS y JavaScript si es necesario.
            - Tipo de documento: página web (HTML).
            - Tono: {tone} (formal, informal, técnico).
            - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - Genera código HTML completo con:
              - Estructura semántica (<header>, <main>, <footer>, etc.).
              - Estilos CSS internos (en <style>) adaptados al tono y propósito.
              - JavaScript opcional (en <script>) si el prompt lo requiere.
              - Usa un diseño responsivo y moderno.
            - Estilo: {style_profile}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia.
            """
        else:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown.
            - Tipo de documento: {doc_type} (e.g., carta formal, informe, correo, contrato, currículum).
            - Tono: {tone} (formal, informal, técnico).
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - Usa encabezados (#, ##), listas (-), negritas (**), y tablas (|...|) cuando sea apropiado.
            - Estilo: {style_profile}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia en la conversación.
            """
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(message_history[-10:])
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        document = response.choices[0].message.content
        response_cache[cache_key] = document

        session_messages[session_id].append({"role": "user", "content": prompt})
        session_messages[session_id].append({"role": "assistant", "content": document})

        return jsonify({
            'document': document,
            'content_type': 'html' if is_html else 'pdf'
        })
    except openai.AuthenticationError:
        return jsonify({'error': 'Error de autenticación con OpenAI. Verifica la clave API.'}), 401
    except openai.RateLimitError:
        return jsonify({'error': 'Límite de solicitudes alcanzado. Intenta de nuevo más tarde.'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_style_profile(profile_id):
    """Obtiene un perfil de estilo o devuelve el estilo por defecto."""
    if not profile_id:
        return get_default_style_profile()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM style_profiles WHERE id = ? AND user_id = ?', (profile_id, current_user.id))
    profile = cursor.fetchone()
    conn.close()
    if profile:
        return {
            'font_name': profile['font_name'],
            'font_size': profile['font_size'],
            'tone': profile['tone'],
            'margins': json.loads(profile['margins']),
            'structure': json.loads(profile['structure'])
        }
    return get_default_style_profile()

@app.route('/generate_preview', methods=['POST'])
@login_required
def generate_preview():
    try:
        data = request.json
        content = data.get('content', '')
        content_type = data.get('content_type', 'pdf')
        style_profile_id = data.get('style_profile_id', None)
        if not content:
            return jsonify({'error': 'El contenido está vacío'}), 400

        if content_type == 'html':
            # Almacenar el contenido HTML temporalmente
            preview_id = f"{current_user.id}:{datetime.now().timestamp()}"
            html_previews[preview_id] = content
            return jsonify({'preview_id': preview_id, 'content_type': 'html'})
        
        # Generar PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch, 
            topMargin=inch, 
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, spaceAfter=12, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle(name='Subtitle', fontSize=10, leading=14, spaceAfter=10, fontName='Helvetica-Oblique')

        story = []
        story.append(Paragraph("Documento Generado por GarBotGPT", title_style))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))
        
        style_profile = get_style_profile(style_profile_id)
        story.extend(parse_markdown_to_reportlab(content, style_profile))
        
        doc.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='documento_generado.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview_html/<preview_id>', methods=['GET'])
@login_required
def preview_html(preview_id):
    """Sirve el contenido HTML para la vista previa."""
    if preview_id in html_previews and preview_id.startswith(f"{current_user.id}:"):
        return Response(html_previews[preview_id], mimetype='text/html')
    return jsonify({'error': 'Vista previa no encontrada'}), 404

@app.route('/analyze_document', methods=['POST'])
@login_required
def analyze_document_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file']
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            return jsonify({'error': 'Formato no soportado. Usa PDF o TXT.'}), 400
        
        style_profile, content = analyze_document(file)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO style_profiles (user_id, font_name, font_size, tone, margins, structure)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            current_user.id,
            style_profile['font_name'],
            style_profile['font_size'],
            style_profile['tone'],
            json.dumps(style_profile['margins']),
            json.dumps(style_profile['structure'])
        ))
        profile_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'style_profile_id': str(profile_id),
            'style_profile': style_profile,
            'content_summary': content[:500]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/style_profiles', methods=['GET'])
@login_required
def get_style_profiles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM style_profiles WHERE user_id = ?', (current_user.id,))
    profiles = cursor.fetchall()
    conn.close()
    
    result = {}
    for profile in profiles:
        result[str(profile['id'])] = {
            'font_name': profile['font_name'],
            'font_size': profile['font_size'],
            'tone': profile['tone'],
            'margins': json.loads(profile['margins']),
            'structure': json.loads(profile['structure'])
        }
    
    return jsonify(result)

@app.route('/style_profiles/<profile_id>', methods=['DELETE'])
@login_required
def delete_style_profile(profile_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM style_profiles WHERE id = ? AND user_id = ?', (profile_id, current_user.id))
    conn.commit()
    conn.close()
    
    if cursor.rowcount > 0:
        return jsonify({'message': 'Perfil de estilo eliminado'})
    return jsonify({'error': 'Perfil no encontrado'}), 404

@app.route('/purge_style_profiles', methods=['DELETE'])
@login_required
def purge_style_profiles():
    """Elimina todos los perfiles de estilo del usuario autenticado."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM style_profiles WHERE user_id = ?', (current_user.id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Todos los perfiles de estilo han sido eliminados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))