from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
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
from datetime import datetime, timedelta
import re
import json
import bcrypt
from collections import Counter
import statistics
import uuid
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")

# Configurar base de datos PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configurar cliente OpenAI
client = openai.OpenAI(api_key=openai_api_key)

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Modelos de la base de datos
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)

class StyleProfile(db.Model):
    __tablename__ = 'style_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    font_name = db.Column(db.String(100))
    font_size = db.Column(db.Integer)
    tone = db.Column(db.String(50))
    margins = db.Column(db.JSON)
    structure = db.Column(db.JSON)
    text_color = db.Column(db.String(7), default="#000000")
    background_color = db.Column(db.String(7), default="#FFFFFF")
    alignment = db.Column(db.String(20), default="left")
    line_spacing = db.Column(db.Float, default=1.33)
    document_purpose = db.Column(db.String(100), default="general")
    confidence_scores = db.Column(db.JSON, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.String(50), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    messages = db.Column(db.JSON, nullable=False, default=[])
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalyzedDocument(db.Model):
    __tablename__ = 'analyzed_documents'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    style_profile_id = db.Column(db.Integer, db.ForeignKey('style_profiles.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Crear tablas en la base de datos
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    logger.error(f"Error al crear tablas en la base de datos: {str(e)}")
    raise

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error al cargar usuario {user_id}: {str(e)}")
        return None

# Cache para respuestas y vistas previas HTML
response_cache = {}
html_previews = {}

def clean_cache():
    """Limpia elementos de caché más antiguos que 1 hora."""
    try:
        current_time = datetime.now().timestamp()
        keys_to_delete = [key for key, value in html_previews.items() if float(key.split(':')[1]) < current_time - 3600]
        for key in keys_to_delete:
            del html_previews[key]
        keys_to_delete = [key for key in response_cache if response_cache[key]['timestamp'] < current_time - 3600]
        for key in keys_to_delete:
            del response_cache[key]
    except Exception as e:
        logger.error(f"Error al limpiar caché: {str(e)}")

def get_default_style_profile():
    return {
        'font_name': 'Helvetica',
        'font_size': 12,
        'tone': 'neutral',
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'structure': ['paragraphs'],
        'text_color': '#000000',
        'background_color': '#FFFFFF',
        'alignment': 'left',
        'line_spacing': 1.33,
        'document_purpose': 'general',
        'confidence_scores': {
            'font_name': 1.0, 'font_size': 1.0, 'tone': 0.8, 'margins': 1.0, 'structure': 0.8,
            'text_color': 1.0, 'background_color': 1.0, 'alignment': 0.9, 'line_spacing': 0.9, 'document_purpose': 0.7
        }
    }

def parse_markdown_to_reportlab(text, style_profile=None):
    try:
        styles = getSampleStyleSheet()
        style_profile = style_profile or get_default_style_profile()
        font_name = style_profile['font_name']
        font_size = style_profile['font_size']
        alignment = {'left': 0, 'center': 1, 'right': 2, 'justified': 4}.get(style_profile['alignment'], 0)
        line_spacing = style_profile['line_spacing']
        text_color = colors.HexColor(style_profile['text_color'])
        background_color = colors.HexColor(style_profile['background_color'])
        
        body_style = ParagraphStyle(
            name='Body',
            fontSize=font_size,
            leading=font_size * line_spacing,
            spaceAfter=8,
            fontName=font_name,
            alignment=alignment,
            textColor=text_color,
            backColor=background_color
        )
        bold_style = ParagraphStyle(
            name='Bold',
            fontSize=font_size,
            leading=font_size * line_spacing,
            spaceAfter=8,
            fontName=f'{font_name}-Bold',
            alignment=alignment,
            textColor=text_color,
            backColor=background_color
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
                    ('LEADING', (0, 0), (-1, -1), font_size * line_spacing),
                    ('TEXTCOLOR', (0, 0), (-1, -1), text_color),
                    ('BACKCOLOR', (0, 0), (-1, -1), background_color),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.2 * inch))
            elif i % 3 != 1:  # Texto normal
                lines = part.split('\n')
                for line in lines:
                    line = line.replace('\n', '<br />')
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    if re.match(r'^\s*[-*]\s+', line):
                        line = f'• {line.lstrip("-* ").strip()}'
                        elements.append(Paragraph(line, body_style))
                    elif re.match(r'^\s*\d+\.\s+', line):
                        line = f'{line.lstrip("0123456789. ").strip()}'
                        elements.append(Paragraph(line, body_style))
                    else:
                        elements.append(Paragraph(line, bold_style if '**' in line else body_style))
                    elements.append(Spacer(1, 0.1 * inch))
        
        return elements
    except Exception as e:
        logger.error(f"Error en parse_markdown_to_reportlab: {str(e)}")
        raise

def analyze_document(file):
    try:
        style_profile = {
            'font_name': 'Helvetica',
            'font_size': 12,
            'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
            'tone': 'neutral',
            'structure': [],
            'text_color': '#000000',
            'background_color': '#FFFFFF',
            'alignment': 'left',
            'line_spacing': 1.33,
            'document_purpose': 'general',
            'confidence_scores': {
                'font_name': 0.8, 'font_size': 0.8, 'tone': 0.7, 'margins': 0.8, 'structure': 0.7,
                'text_color': 0.8, 'background_color': 0.8, 'alignment': 0.7, 'line_spacing': 0.7, 'document_purpose': 0.6
            }
        }
        
        content = ""
        if file.filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                font_names = []
                font_sizes = []
                margins = {'top': [], 'bottom': [], 'left': [], 'right': []}
                text_colors = []
                alignments = []
                line_spacings = []
                structures = set()
                
                for page in pdf.pages:
                    content += page.extract_text() or ""
                    chars = page.chars
                    if chars:
                        font_names.extend([char.get('fontname', 'Helvetica').split('-')[0] for char in chars])
                        font_sizes.extend([round(char.get('size', 12)) for char in chars])
                        text_colors.extend([char.get('non_stroking_color', (0, 0, 0)) for char in chars])
                    
                    if page.chars:
                        margins['top'].append(page.bbox[3] - max(c['y1'] for c in page.chars))
                        margins['bottom'].append(min(c['y0'] for c in page.chars) - page.bbox[1])
                        margins['left'].append(min(c['x0'] for c in page.chars) - page.bbox[0])
                        margins['right'].append(page.bbox[2] - max(c['x1'] for c in page.chars))
                    
                    if page.extract_text():
                        lines = page.extract_text().split('\n')
                        for i in range(len(lines)-1):
                            try:
                                y0 = page.chars[i]['y0']
                                y1 = page.chars[i+1]['y0']
                                spacing = (y0 - y1) / page.chars[i].get('size', 12)
                                if spacing > 0:
                                    line_spacings.append(spacing)
                            except IndexError:
                                continue
                        
                        x_positions = [c['x0'] for c in page.chars]
                        if x_positions:
                            x_variance = statistics.variance(x_positions) if len(x_positions) > 1 else 0
                            if x_variance < 10:
                                alignments.append('justified' if max(x_positions) - min(x_positions) > page.width * 0.8 else 'left')
                            elif max(x_positions) > page.width * 0.9:
                                alignments.append('right')
                            elif abs(max(x_positions) + min(x_positions) - page.width) < page.width * 0.1:
                                alignments.append('center')
                    
                    if page.extract_tables():
                        structures.add('tables')
                    if page.extract_text():
                        text = page.extract_text()
                        if re.search(r'^\s*[-*]\s+', text, re.MULTILINE):
                            structures.add('bullet_lists')
                        if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
                            structures.add('numbered_lists')
                        if re.search(r'^(#+)\s+', text, re.MULTILINE):
                            structures.add('headings')
                        if re.search(r'\n\n+', text):
                            structures.add('paragraphs')
                
                if font_names:
                    font_counter = Counter(font_names)
                    style_profile['font_name'] = font_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['font_name'] = font_counter.most_common(1)[0][1] / len(font_names)
                
                if font_sizes:
                    size_counter = Counter(font_sizes)
                    style_profile['font_size'] = size_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['font_size'] = size_counter.most_common(1)[0][1] / len(font_sizes)
                
                if margins['top']:
                    for key in margins:
                        style_profile['margins'][key] = statistics.mean(margins[key]) / 72
                        style_profile['confidence_scores']['margins'] = min(1.0, len(margins[key]) / len(pdf.pages))
                
                if text_colors:
                    color_counter = Counter([tuple(c) if isinstance(c, (list, tuple)) else c for c in text_colors])
                    dominant_color = color_counter.most_common(1)[0][0]
                    if isinstance(dominant_color, (list, tuple)):
                        style_profile['text_color'] = '#{:02x}{:02x}{:02x}'.format(
                            int(dominant_color[0] * 255), int(dominant_color[1] * 255), int(dominant_color[2] * 255)
                        )
                    style_profile['confidence_scores']['text_color'] = color_counter.most_common(1)[0][1] / len(text_colors)
                
                if alignments:
                    alignment_counter = Counter(alignments)
                    style_profile['alignment'] = alignment_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['alignment'] = alignment_counter.most_common(1)[0][1] / len(alignments)
                
                if line_spacings:
                    style_profile['line_spacing'] = min(max(statistics.mean(line_spacings), 1.0), 2.0)
                    style_profile['confidence_scores']['line_spacing'] = len(line_spacings) / len(pdf.pages)
                
                style_profile['structure'] = list(structures)
                style_profile['confidence_scores']['structure'] = 0.9 if structures else 0.7
                style_profile['background_color'] = '#FFFFFF'
                style_profile['confidence_scores']['background_color'] = 0.9
            
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8', errors='ignore')
            structures = set()
            
            if re.search(r'^\s*[-*]\s+', content, re.MULTILINE):
                structures.add('bullet_lists')
            if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
                structures.add('numbered_lists')
            if re.search(r'^\s*[A-Z].*\n[-=]+', content, re.MULTILINE):
                structures.add('headings')
            if re.search(r'\n\n+', content):
                structures.add('paragraphs')
            if re.search(r'^\s*\|.*\|\s*$', content, re.MULTILINE):
                structures.add('tables')
            
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.8 if structures else 0.6
        
        if content:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            Analiza el texto proporcionado y devuelve un JSON con:
                            - 'tone': tono del texto (formal, informal, técnico, académico, persuasivo, narrativo).
                            - 'document_purpose': propósito del documento (informe, carta, manual, artículo, presentación, contrato, currículum).
                            - 'confidence': confianza en la predicción (0.0 a 1.0).
                            Usa el contexto, vocabulario, y estructura para determinar el tono y propósito con alta precisión.
                            """
                        },
                        {"role": "user", "content": content[:4000]}
                    ],
                    max_tokens=500
                )
                analysis = json.loads(response.choices[0].message.content)
                style_profile['tone'] = analysis.get('tone', 'neutral')
                style_profile['document_purpose'] = analysis.get('document_purpose', 'general')
                style_profile['confidence_scores']['tone'] = analysis.get('confidence', 0.7)
                style_profile['confidence_scores']['document_purpose'] = analysis.get('confidence', 0.7)
            except Exception as e:
                logger.error(f"Error en análisis de tono: {str(e)}")
        
        font_mapping = {
            'Times New Roman': 'Times-Roman',
            'Arial': 'Helvetica',
            'Courier New': 'Courier',
        }
        style_profile['font_name'] = font_mapping.get(style_profile['font_name'], style_profile['font_name'])
        
        return style_profile, content
    except Exception as e:
        logger.error(f"Error en analyze_document: {str(e)}")
        raise

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
        
        try:
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                login_user(user)
                logger.info(f"Usuario {email} autenticado exitosamente")
                return redirect(url_for('index'))
            else:
                flash('Correo o contraseña incorrectos', 'error')
                logger.warning(f"Intento de login fallido para {email}")
        except Exception as e:
            logger.error(f"Error en login: {str(e)}")
            flash('Error interno al iniciar sesión', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if not email or not password:
            flash('Correo y contraseña son obligatorios', 'error')
            logger.warning("Intento de registro con campos vacíos")
            return render_template('register.html')
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            user = User(email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Registro exitoso. Por favor, inicia sesión.', 'success')
            logger.info(f"Usuario {email} registrado exitosamente")
            return redirect(url_for('login'))
        except db.IntegrityError:
            db.session.rollback()
            flash('El correo ya está registrado', 'error')
            logger.warning(f"Intento de registro con correo duplicado: {email}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error en registro: {str(e)}")
            flash('Error interno al registrarse', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        logger.info("Usuario deslogueado exitosamente")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error en logout: {str(e)}")
        return jsonify({'error': 'Error interno al cerrar sesión'}), 500

@app.route('/generate', methods=['POST'])
@login_required
def generate_document():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        session_id = data.get('session_id')
        doc_type = data.get('doc_type', 'general')
        tone = data.get('tone', 'neutral')
        length = data.get('length', 'medium')
        language = data.get('language', 'es')
        style_profile_id = data.get('style_profile_id')
        message_history = data.get('message_history', [])

        # Validaciones
        if not prompt:
            logger.warning("Prompt vacío recibido en /generate")
            return jsonify({'error': 'El prompt está vacío'}), 400
        if len(prompt) > 10000:
            logger.warning("Prompt excede límite de 10,000 caracteres")
            return jsonify({'error': 'El prompt excede el límite de 10,000 caracteres'}), 400
        valid_doc_types = ['general', 'formal_letter', 'report', 'email', 'contract', 'resume', 'html']
        valid_tones = ['neutral', 'formal', 'informal', 'technical']
        valid_lengths = ['short', 'medium', 'long']
        valid_languages = ['es', 'en', 'fr']
        if doc_type not in valid_doc_types:
            logger.warning(f"Tipo de documento inválido: {doc_type}")
            return jsonify({'error': f'Tipo de documento inválido. Opciones: {valid_doc_types}'}), 400
        if tone not in valid_tones:
            logger.warning(f"Tono inválido: {tone}")
            return jsonify({'error': f'Tono inválido. Opciones: {valid_tones}'}), 400
        if length not in valid_lengths:
            logger.warning(f"Longitud inválida: {length}")
            return jsonify({'error': f'Longitud inválida. Opciones: {valid_lengths}'}), 400
        if language not in valid_languages:
            logger.warning(f"Idioma inválido: {language}")
            return jsonify({'error': f'Idioma inválido. Opciones: {valid_languages}'}), 400

        # Validar session_id
        if session_id:
            session = Session.query.filter_by(id=session_id, user_id=current_user.id).first()
            if not session:
                logger.warning(f"Sesión no encontrada o no autorizada: {session_id}")
                return jsonify({'error': 'Sesión no encontrada o no autorizada'}), 404
        else:
            session_id = str(uuid.uuid4())
            session = Session(
                id=session_id,
                user_id=current_user.id,
                name=f"Sesión {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                messages=[]
            )
            db.session.add(session)

        clean_cache()
        cache_key = f"{current_user.id}:{session_id}:{prompt}:{doc_type}:{tone}:{length}:{language}:{style_profile_id}"
        if cache_key in response_cache:
            logger.info(f"Respuesta obtenida desde caché: {cache_key}")
            return jsonify({
                'document': response_cache[cache_key]['document'],
                'content_type': response_cache[cache_key]['content_type'],
                'session_id': session_id
            })

        style_profile = get_style_profile(style_profile_id)
        
        is_explanatory = re.search(r'^(¿Quién es|¿Qué es|explicar|detallar)\b', prompt, re.IGNORECASE) is not None
        is_html = doc_type == 'html' or re.search(r'\b(html|página web|sitio web)\b', prompt, re.IGNORECASE) is not None
        
        if is_explanatory:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown con una estructura clara para explicaciones.
            - Tipo de documento: {doc_type}.
            - Tono: {tone}.
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language}.
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
            - Estilo: {json.dumps(style_profile)}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia.
            """
        elif is_html:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera páginas web en formato HTML con CSS y JavaScript si es necesario.
            - Tipo de documento: página web (HTML).
            - Tono: {tone}.
            - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
            - Idioma: {language}.
            - Genera código HTML completo con:
              - Estructura semántica (<header>, <main>, <footer>, etc.).
              - Estilos CSS internos (en <style>) adaptados al tono y propósito.
              - JavaScript opcional (en <script>) si el prompt lo requiere.
              - Usa un diseño responsivo y moderno.
            - Estilo: {json.dumps(style_profile)}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia.
            """
        else:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown.
            - Tipo de documento: {doc_type}.
            - Tono: {tone}.
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language}.
            - Usa encabezados (#, ##), listas (-), negritas (**), y tablas (|...|) cuando sea apropiado.
            - Estilo: {json.dumps(style_profile)}.
            - Considera el contexto de los mensajes anteriores para mantener coherencia.
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
        content_type = 'html' if is_html else 'pdf'
        response_cache[cache_key] = {'document': document, 'content_type': content_type, 'timestamp': datetime.now().timestamp()}

        # Guardar mensajes en la sesión
        session.messages.append({"role": "user", "content": prompt, "content_type": doc_type})
        session.messages.append({"role": "assistant", "content": document, "content_type": content_type})
        db.session.commit()

        logger.info(f"Documento generado para sesión {session_id}")
        return jsonify({
            'document': document,
            'content_type': content_type,
            'session_id': session_id
        })
    except openai.AuthenticationError:
        logger.error("Error de autenticación con OpenAI")
        return jsonify({'error': 'Error de autenticación con OpenAI. Verifica la clave API.'}), 401
    except openai.RateLimitError:
        logger.error("Límite de solicitudes alcanzado en OpenAI")
        return jsonify({'error': 'Límite de solicitudes alcanzado. Intenta de nuevo más tarde.'}), 429
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /generate: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /generate: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

def get_style_profile(profile_id):
    try:
        if not profile_id:
            return get_default_style_profile()
        profile = StyleProfile.query.filter_by(id=profile_id, user_id=current_user.id).first()
        if profile:
            return {
                'font_name': profile.font_name,
                'font_size': profile.font_size,
                'tone': profile.tone,
                'margins': profile.margins,
                'structure': profile.structure,
                'text_color': profile.text_color,
                'background_color': profile.background_color,
                'alignment': profile.alignment,
                'line_spacing': profile.line_spacing,
                'document_purpose': profile.document_purpose,
                'confidence_scores': profile.confidence_scores
            }
        logger.warning(f"Perfil de estilo {profile_id} no encontrado para usuario {current_user.id}")
        return get_default_style_profile()
    except Exception as e:
        logger.error(f"Error en get_style_profile: {str(e)}")
        return get_default_style_profile()

@app.route('/generate_preview', methods=['POST'])
@login_required
def generate_preview():
    try:
        data = request.json
        content = data.get('content', '').strip()
        content_type = data.get('content_type', 'pdf')
        style_profile_id = data.get('style_profile_id')
        
        if not content:
            logger.warning("Contenido vacío recibido en /generate_preview")
            return jsonify({'error': 'El contenido está vacío'}), 400
        if content_type not in ['pdf', 'html']:
            logger.warning(f"Tipo de contenido inválido: {content_type}")
            return jsonify({'error': 'Tipo de contenido inválido. Opciones: pdf, html'}), 400

        clean_cache()
        
        if content_type == 'html':
            preview_id = f"{current_user.id}:{datetime.now().timestamp()}"
            html_previews[preview_id] = content
            logger.info(f"Vista previa HTML generada: {preview_id}")
            return jsonify({'preview_id': preview_id, 'content_type': 'html'})
        
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
        logger.info("Vista previa PDF generada")
        return send_file(
            buffer,
            as_attachment=True,
            download_name='documento_generado.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error en /generate_preview: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/preview_html/<preview_id>', methods=['GET'])
@login_required
def preview_html(preview_id):
    try:
        clean_cache()
        if preview_id in html_previews and preview_id.startswith(f"{current_user.id}:"):
            logger.info(f"Sirviendo vista previa HTML: {preview_id}")
            return Response(html_previews[preview_id], mimetype='text/html')
        logger.warning(f"Vista previa HTML no encontrada: {preview_id}")
        return jsonify({'error': 'Vista previa no encontrada'}), 404
    except Exception as e:
        logger.error(f"Error en /preview_html/{preview_id}: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/analyze_document', methods=['POST'])
@login_required
def analyze_document_endpoint():
    try:
        if 'file' not in request.files:
            logger.warning("No se proporcionó archivo en /analyze_document")
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file']
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            logger.warning(f"Formato de archivo no soportado: {file.filename}")
            return jsonify({'error': 'Formato no soportado. Usa PDF o TXT.'}), 400
        
        if file.content_length > 10 * 1024 * 1024:  # Límite de 10MB
            logger.warning(f"Archivo excede límite de 10MB: {file.filename}")
            return jsonify({'error': 'El archivo excede el límite de 10MB'}), 400

        style_profile, content = analyze_document(file)
        
        profile = StyleProfile(
            user_id=current_user.id,
            font_name=style_profile['font_name'],
            font_size=style_profile['font_size'],
            tone=style_profile['tone'],
            margins=style_profile['margins'],
            structure=style_profile['structure'],
            text_color=style_profile['text_color'],
            background_color=style_profile['background_color'],
            alignment=style_profile['alignment'],
            line_spacing=style_profile['line_spacing'],
            document_purpose=style_profile['document_purpose'],
            confidence_scores=style_profile['confidence_scores']
        )
        db.session.add(profile)
        db.session.flush()
        
        analyzed_doc = AnalyzedDocument(
            user_id=current_user.id,
            filename=file.filename,
            content=content,
            style_profile_id=profile.id
        )
        db.session.add(analyzed_doc)
        db.session.commit()
        
        logger.info(f"Documento analizado: {file.filename}, perfil de estilo ID: {profile.id}")
        return jsonify({
            'style_profile_id': str(profile.id),
            'analyzed_document_id': str(analyzed_doc.id),
            'style_profile': style_profile,
            'content_summary': content[:500]
        })
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /analyze_document: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /analyze_document: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/style_profiles', methods=['GET'])
@login_required
def get_style_profiles():
    try:
        if not current_user.is_authenticated:
            logger.warning("Usuario no autenticado intentó acceder a /style_profiles")
            return jsonify({'error': 'Usuario no autenticado'}), 401

        profiles = StyleProfile.query.filter_by(user_id=current_user.id).all()
        result = {}
        for profile in profiles:
            try:
                # Validar campos JSON
                margins = profile.margins if profile.margins and isinstance(profile.margins, dict) else {}
                structure = profile.structure if profile.structure and isinstance(profile.structure, list) else []
                confidence_scores = profile.confidence_scores if profile.confidence_scores and isinstance(profile.confidence_scores, dict) else {}

                result[str(profile.id)] = {
                    'font_name': profile.font_name or 'Helvetica',
                    'font_size': profile.font_size or 12,
                    'tone': profile.tone or 'neutral',
                    'margins': margins,
                    'structure': structure,
                    'text_color': profile.text_color or '#000000',
                    'background_color': profile.background_color or '#FFFFFF',
                    'alignment': profile.alignment or 'left',
                    'line_spacing': profile.line_spacing or 1.33,
                    'document_purpose': profile.document_purpose or 'general',
                    'confidence_scores': confidence_scores,
                    'created_at': profile.created_at.isoformat() if profile.created_at else datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error al procesar perfil de estilo ID {profile.id}: {str(e)}")
                continue
        
        logger.info(f"Devolviendo {len(result)} perfiles de estilo para usuario {current_user.id}")
        return jsonify(result)
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /style_profiles: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /style_profiles: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/style_profiles/<profile_id>', methods=['DELETE'])
@login_required
def delete_style_profile(profile_id):
    try:
        profile = StyleProfile.query.filter_by(id=profile_id, user_id=current_user.id).first()
        if profile:
            db.session.delete(profile)
            db.session.commit()
            logger.info(f"Perfil de estilo {profile_id} eliminado para usuario {current_user.id}")
            return jsonify({'message': 'Perfil de estilo eliminado'})
        logger.warning(f"Perfil de estilo {profile_id} no encontrado para usuario {current_user.id}")
        return jsonify({'error': 'Perfil no encontrado'}), 404
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /style_profiles/{profile_id}: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /style_profiles/{profile_id}: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/purge_style_profiles', methods=['DELETE'])
@login_required
def purge_style_profiles():
    try:
        num_deleted = StyleProfile.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        logger.info(f"Eliminados {num_deleted} perfiles de estilo para usuario {current_user.id}")
        return jsonify({'message': f'Todos los perfiles de estilo han sido eliminados ({num_deleted})'})
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /purge_style_profiles: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /purge_style_profiles: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/sessions', methods=['GET'])
@login_required
def get_sessions():
    try:
        sessions = Session.query.filter_by(user_id=current_user.id).all()
        result = {
            session.id: {
                'name': session.name,
                'messages': session.messages,
                'created_at': session.created_at.isoformat()
            } for session in sessions
        }
        logger.info(f"Devolviendo {len(result)} sesiones para usuario {current_user.id}")
        return jsonify(result)
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /sessions: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /sessions: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/session/<session_id>', methods=['GET'])
@login_required
def get_session(session_id):
    try:
        session = Session.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            logger.warning(f"Sesión {session_id} no encontrada para usuario {current_user.id}")
            return jsonify({'error': 'Sesión no encontrada o no autorizada'}), 404
        logger.info(f"Devolviendo sesión {session_id} para usuario {current_user.id}")
        return jsonify({
            'id': session.id,
            'name': session.name,
            'messages': session.messages,
            'created_at': session.created_at.isoformat()
        })
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /session/{session_id}: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /session/{session_id}: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/session', methods=['POST'])
@login_required
def create_session():
    try:
        data = request.json
        name = data.get('name', f"Sesión {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if not name or len(name) > 100:
            logger.warning(f"Nombre de sesión inválido: {name}")
            return jsonify({'error': 'El nombre debe tener entre 1 y 100 caracteres'}), 400
        
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            user_id=current_user.id,
            name=name,
            messages=[]
        )
        db.session.add(session)
        db.session.commit()
        logger.info(f"Sesión creada: {session_id} para usuario {current_user.id}")
        return jsonify({
            'id': session.id,
            'name': session.name,
            'messages': session.messages,
            'created_at': session.created_at.isoformat()
        })
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /session POST: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /session POST: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/session/<session_id>', methods=['PUT'])
@login_required
def update_session(session_id):
    try:
        session = Session.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            logger.warning(f"Sesión {session_id} no encontrada para usuario {current_user.id}")
            return jsonify({'error': 'Sesión no encontrada o no autorizada'}), 404
        
        data = request.json
        name = data.get('name')
        if not name or len(name) > 100:
            logger.warning(f"Nombre de sesión inválido en actualización: {name}")
            return jsonify({'error': 'El nombre debe tener entre 1 y 100 caracteres'}), 400
        
        session.name = name
        db.session.commit()
        logger.info(f"Sesión {session_id} renombrada para usuario {current_user.id}")
        return jsonify({
            'id': session.id,
            'name': session.name,
            'messages': session.messages,
            'created_at': session.created_at.isoformat()
        })
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /session/{session_id} PUT: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /session/{session_id} PUT: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/session/<session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    try:
        session = Session.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            logger.warning(f"Sesión {session_id} no encontrada para usuario {current_user.id}")
            return jsonify({'error': 'Sesión no encontrada o no autorizada'}), 404
        
        db.session.delete(session)
        db.session.commit()
        logger.info(f"Sesión {session_id} eliminada para usuario {current_user.id}")
        return jsonify({'message': 'Sesión eliminada'})
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /session/{session_id} DELETE: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /session/{session_id} DELETE: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/session/<session_id>/clear', methods=['POST'])
@login_required
def clear_session(session_id):
    try:
        session = Session.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            logger.warning(f"Sesión {session_id} no encontrada para usuario {current_user.id}")
            return jsonify({'error': 'Sesión no encontrada o no autorizada'}), 404
        
        session.messages = []
        db.session.commit()
        logger.info(f"Mensajes de sesión {session_id} limpiados para usuario {current_user.id}")
        return jsonify({'message': 'Mensajes de la sesión limpiados'})
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /session/{session_id}/clear: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /session/{session_id}/clear: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/analyzed_documents', methods=['GET'])
@login_required
def get_analyzed_documents():
    try:
        documents = AnalyzedDocument.query.filter_by(user_id=current_user.id).all()
        result = {}
        for doc in documents:
            result[str(doc.id)] = {
                'filename': doc.filename,
                'content_summary': doc.content[:500],
                'style_profile_id': str(doc.style_profile_id) if doc.style_profile_id else None,
                'created_at': doc.created_at.isoformat()
            }
        logger.info(f"Devolviendo {len(result)} documentos analizados para usuario {current_user.id}")
        return jsonify(result)
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /analyzed_documents: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /analyzed_documents: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/analyzed_documents/<doc_id>', methods=['DELETE'])
@login_required
def delete_analyzed_document(doc_id):
    try:
        doc = AnalyzedDocument.query.filter_by(id=doc_id, user_id=current_user.id).first()
        if not doc:
            logger.warning(f"Documento analizado {doc_id} no encontrado para usuario {current_user.id}")
            return jsonify({'error': 'Documento no encontrado o no autorizado'}), 404
        
        db.session.delete(doc)
        db.session.commit()
        logger.info(f"Documento analizado {doc_id} eliminado para usuario {current_user.id}")
        return jsonify({'message': 'Documento analizado eliminado'})
    except db.OperationalError as e:
        logger.error(f"Error de base de datos en /analyzed_documents/{doc_id}: {str(e)}")
        return jsonify({'error': 'Error de base de datos. Por favor, intenta de nuevo.'}), 500
    except Exception as e:
        logger.error(f"Error en /analyzed_documents/{doc_id}: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Iniciando servidor en puerto {port}")
    app.run(host='0.0.0.0', port=port)