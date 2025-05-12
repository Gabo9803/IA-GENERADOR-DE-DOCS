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
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from datetime import datetime
import re
import json
import sqlite3
import bcrypt
from collections import Counter
import statistics
import unicodedata

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

# Definir la versión actual del esquema
CURRENT_SCHEMA_VERSION = 2

def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
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
    
    # Tabla para rastrear la versión del esquema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL
        )
    ''')
    
    # Tabla de perfiles de estilo (estructura base)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS style_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            font_name TEXT,
            font_size INTEGER,
            tone TEXT,
            margins TEXT,
            structure TEXT,
            text_color TEXT,
            background_color TEXT,
            alignment TEXT,
            line_spacing REAL,
            document_purpose TEXT,
            confidence_scores TEXT,
            font_styles TEXT,
            heading_levels INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def migrate_db():
    """Verifica y migra la base de datos a la versión actual."""
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    
    # Obtener la versión actual del esquema
    cursor.execute('SELECT version FROM schema_version WHERE id = 1')
    result = cursor.fetchone()
    if result:
        current_version = result[0]
    else:
        # Si no existe, asumir versión 0 e insertar
        current_version = 0
        cursor.execute('INSERT INTO schema_version (id, version) VALUES (1, ?)', (current_version,))
        conn.commit()
    
    # Migrar desde la versión actual hasta CURRENT_SCHEMA_VERSION
    if current_version < 1:
        # Migración a versión 1: Estructura inicial ya creada en init_db
        current_version = 1
    
    if current_version < 2:
        # Migración a versión 2: Añadir nuevos campos a style_profiles
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN text_color TEXT DEFAULT "#000000"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN background_color TEXT DEFAULT "#FFFFFF"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN alignment TEXT DEFAULT "left"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN line_spacing REAL DEFAULT 1.33')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN document_purpose TEXT DEFAULT "general"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN confidence_scores TEXT DEFAULT "{}"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN font_styles TEXT DEFAULT "[]"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN heading_levels INTEGER DEFAULT 1')
        current_version = 2
    
    # Actualizar la versión del esquema
    cursor.execute('UPDATE schema_version SET version = ? WHERE id = 1', (CURRENT_SCHEMA_VERSION,))
    conn.commit()
    conn.close()

# Ejecutar la inicialización y migración de la base de datos al iniciar la aplicación
init_db()
migrate_db()

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
        'structure': ['paragraphs'],
        'text_color': '#000000',
        'background_color': '#FFFFFF',
        'alignment': 'left',
        'line_spacing': 1.33,
        'document_purpose': 'general',
        'confidence_scores': {
            'font_name': 1.0, 'font_size': 1.0, 'tone': 0.8, 'margins': 1.0, 'structure': 0.8,
            'text_color': 1.0, 'background_color': 1.0, 'alignment': 0.9, 'line_spacing': 0.9,
            'document_purpose': 0.7, 'font_styles': 0.8, 'heading_levels': 0.8
        },
        'font_styles': ['regular'],
        'heading_levels': 1
    }

def parse_markdown_to_reportlab(text, style_profile=None):
    """Convierte Markdown a elementos de reportlab con soporte para estilos personalizados."""
    styles = getSampleStyleSheet()
    style_profile = style_profile or get_default_style_profile()
    font_name = style_profile['font_name']
    font_size = style_profile['font_size']
    alignment = {'left': 0, 'center': 1, 'right': 2, 'justified': 4}.get(style_profile['alignment'], 0)
    line_spacing = style_profile['line_spacing']
    text_color = colors.HexColor(style_profile['text_color'])
    background_color = colors.HexColor(style_profile['background_color'])
    
    # Validar la fuente
    available_fonts = pdfmetrics.getRegisteredFontNames()
    if font_name not in available_fonts:
        print(f"Fuente no válida '{font_name}', usando 'Helvetica' como respaldo")  # Depuración
        font_name = 'Helvetica'
    bold_font = f'{font_name}-Bold'
    if bold_font not in available_fonts:
        print(f"Fuente negrita no válida '{bold_font}', usando 'Helvetica-Bold' como respaldo")  # Depuración
        bold_font = 'Helvetica-Bold'
    
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
        fontName=bold_font,
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
                if line.strip().startswith('- '):
                    line = f'• {line[2:]}'
                    elements.append(Paragraph(line, body_style))
                else:
                    elements.append(Paragraph(line, bold_style if '**' in line else body_style))
                elements.append(Spacer(1, 0.1 * inch))
    
    return elements

def analyze_document(file):
    """Analiza un documento subido para extraer estilo, formato y tipografía con mayor precisión."""
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
            'text_color': 0.8, 'background_color': 0.8, 'alignment': 0.7, 'line_spacing': 0.7,
            'document_purpose': 0.6, 'font_styles': 0.7, 'heading_levels': 0.7
        },
        'font_styles': ['regular'],
        'heading_levels': 1
    }
    
    content = ""
    try:
        if file.filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                font_names = []
                font_sizes = []
                font_styles = []
                margins = {'top': [], 'bottom': [], 'left': [], 'right': []}
                text_colors = []
                alignments = []
                line_spacings = []
                structures = set()
                heading_sizes = []
                
                for page in pdf.pages:
                    # Extraer texto
                    page_text = page.extract_text() or ""
                    content += page_text + "\n"
                    
                    # Analizar caracteres para fuentes, tamaños, colores y estilos
                    chars = page.chars
                    if chars:
                        raw_fonts = [char.get('fontname', 'Helvetica') for char in chars]
                        # Limpiar y normalizar nombres de fuente
                        cleaned_fonts = []
                        for f in raw_fonts:
                            # Eliminar prefijos no alfabéticos y normalizar
                            f_clean = re.sub(r'^[^a-zA-Z]+', '', f).lower()
                            f_clean = re.sub(r'[-_].*$', '', f_clean)  # Eliminar sufijos de estilo
                            cleaned_fonts.append(f_clean)
                            # Detectar estilos (negrita, cursiva)
                            if 'bold' in f.lower():
                                font_styles.append('bold')
                            elif 'italic' in f.lower() or 'oblique' in f.lower():
                                font_styles.append('italic')
                            else:
                                font_styles.append('regular')
                        
                        font_names.extend(cleaned_fonts)
                        font_sizes.extend([round(char.get('size', 12)) for char in chars])
                        text_colors.extend([char.get('non_stroking_color', (0, 0, 0)) for char in chars])
                        
                        # Detectar posibles encabezados basados en tamaño de fuente
                        char_sizes = [char.get('size', 12) for char in chars]
                        if char_sizes:
                            median_size = statistics.median(char_sizes)
                            heading_sizes.extend([s for s in char_sizes if s > median_size * 1.2])
                    
                    # Calcular márgenes
                    if page.chars:
                        margins['top'].append(page.bbox[3] - max(c['y1'] for c in page.chars))
                        margins['bottom'].append(min(c['y0'] for c in page.chars) - page.bbox[1])
                        margins['left'].append(min(c['x0'] for c in page.chars) - page.bbox[0])
                        margins['right'].append(page.bbox[2] - max(c['x1'] for c in page.chars))
                    
                    # Detectar alineación y espaciado
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
                        
                        # Estimar alineación basada en posiciones x
                        x_positions = [c['x0'] for c in page.chars]
                        if x_positions:
                            x_variance = statistics.variance(x_positions) if len(x_positions) > 1 else 0
                            if x_variance < 10:
                                alignments.append('justified' if max(x_positions) - min(x_positions) > page.width * 0.8 else 'left')
                            elif max(x_positions) > page.width * 0.9:
                                alignments.append('right')
                            elif abs(max(x_positions) + min(x_positions) - page.width) < page.width * 0.1:
                                alignments.append('center')
                    
                    # Detectar estructuras avanzadas
                    if page.extract_tables():
                        structures.add('tables')
                    if page_text:
                        # Listas
                        if re.search(r'^\s*([-*]|\d+\.)\s+', page_text, re.MULTILINE):
                            structures.add('lists')
                        # Encabezados
                        if re.search(r'^(#{1,3})\s+', page_text, re.MULTILINE):
                            structures.add('headings')
                        # Párrafos
                        if re.search(r'\n\n+', page_text):
                            structures.add('paragraphs')
                        # Notas al pie
                        if re.search(r'^\[\d+\]\s+', page_text, re.MULTILINE):
                            structures.add('footnotes')
                        # Citas
                        if re.search(r'^\s*>\s+', page_text, re.MULTILINE):
                            structures.add('quotes')
                
                # Normalizar y calcular valores dominantes
                if font_names:
                    font_counter = Counter(font_names)
                    raw_font = font_counter.most_common(1)[0][0]
                    print(f"Fuente detectada: {raw_font}")  # Depuración
                    
                    # Mapeo de fuentes a las soportadas por ReportLab
                    font_mapping = {
                        'timesnewroman': 'Times-Roman',
                        'times new roman': 'Times-Roman',
                        'times': 'Times-Roman',
                        'arial': 'Helvetica',
                        'arialm': 'Helvetica',
                        'helvetica': 'Helvetica',
                        'couriernew': 'Courier',
                        'courier new': 'Courier',
                        'courier': 'Courier',
                        '': 'Helvetica'
                    }
                    normalized_font = font_mapping.get(raw_font.lower(), 'Helvetica')
                    style_profile['font_name'] = normalized_font
                    style_profile['confidence_scores']['font_name'] = font_counter.most_common(1)[0][1] / len(font_names)
                    print(f"Fuente normalizada: {style_profile['font_name']}")  # Depuración
                
                if font_sizes:
                    size_counter = Counter(font_sizes)
                    style_profile['font_size'] = size_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['font_size'] = size_counter.most_common(1)[0][1] / len(font_sizes)
                    # Validar font_size
                    style_profile['font_size'] = max(6, min(24, style_profile['font_size']))
                
                if font_styles:
                    style_counter = Counter(font_styles)
                    style_profile['font_styles'] = list(set(style_counter.keys()))
                    style_profile['confidence_scores']['font_styles'] = style_counter.most_common(1)[0][1] / len(font_styles)
                
                if margins['top']:
                    for key in margins:
                        style_profile['margins'][key] = statistics.mean(margins[key]) / 72  # Convertir a pulgadas
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
                    style_profile['line_spacing'] = min(max(statistics.mean(line_spacings), 1.0), 3.0)
                    style_profile['confidence_scores']['line_spacing'] = len(line_spacings) / len(pdf.pages)
                
                if heading_sizes:
                    # Estimar niveles de encabezado basados en tamaños de fuente
                    unique_sizes = sorted(set(heading_sizes), reverse=True)
                    style_profile['heading_levels'] = min(len(unique_sizes), 3)  # Máximo 3 niveles
                    style_profile['confidence_scores']['heading_levels'] = len(heading_sizes) / len(font_sizes)
                
                style_profile['structure'] = list(structures)
                style_profile['confidence_scores']['structure'] = 0.9 if structures else 0.7
                
                # Asumir fondo blanco
                style_profile['background_color'] = '#FFFFFF'
                style_profile['confidence_scores']['background_color'] = 0.9
                
                print(f"Estructuras detectadas: {style_profile['structure']}")  # Depuración
                print(f"Estilos de fuente detectados: {style_profile['font_styles']}")  # Depuración
                print(f"Niveles de encabezado: {style_profile['heading_levels']}")  # Depuración
        
        elif file.filename.endswith('.txt'):
            try:
                content = file.read().decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                content = file.read().decode('latin-1', errors='ignore')
                print("Codificación no UTF-8 detectada, usando latin-1 como respaldo")  # Depuración
            
            structures = set()
            alignments = []
            line_spacings = []
            
            # Normalizar texto para análisis
            content = unicodedata.normalize('NFKD', content)
            lines = content.split('\n')
            
            # Estimar alineación basada en sangría
            indentations = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            if indentations:
                indent_variance = statistics.variance(indentations) if len(indentations) > 1 else 0
                if indent_variance < 2:
                    alignments.append('left')
                elif any(indent > 10 for indent in indentations):
                    alignments.append('justified')
            
            # Estimar espaciado entre líneas basado en líneas vacías
            empty_line_count = sum(1 for line in lines if not line.strip())
            if empty_line_count > len(lines) * 0.1:
                line_spacings.append(1.5)  # Estimar espaciado mayor
            else:
                line_spacings.append(1.2)
            
            # Detectar estructuras avanzadas
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                # Listas
                if re.match(r'^([-*]|\d+\.)\s+', line):
                    structures.add('lists')
                # Encabezados
                if re.match(r'^[A-Z].*\n[-=]+$', content, re.MULTILINE) or re.match(r'^#{1,3}\s+', line):
                    structures.add('headings')
                # Párrafos
                if len(line) > 50 and not re.match(r'^[-*]|\d+\.', line):
                    structures.add('paragraphs')
                # Tablas
                if re.match(r'^\|.*\|\s*$', line):
                    structures.add('tables')
                # Citas
                if line.startswith('>'):
                    structures.add('quotes')
                # Bloques de código
                if re.match(r'^```|^    ', line):
                    structures.add('code')
            
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.8 if structures else 0.6
            
            if alignments:
                alignment_counter = Counter(alignments)
                style_profile['alignment'] = alignment_counter.most_common(1)[0][0]
                style_profile['confidence_scores']['alignment'] = alignment_counter.most_common(1)[0][1] / len(alignments)
            
            if line_spacings:
                style_profile['line_spacing'] = min(max(statistics.mean(line_spacings), 1.0), 3.0)
                style_profile['confidence_scores']['line_spacing'] = 0.7
            
            print(f"Estructuras detectadas en TXT: {style_profile['structure']}")  # Depuración
        
        # Analizar tono y propósito con OpenAI
        if content:
            # Extraer palabras clave para mejorar el análisis
            words = re.findall(r'\b\w+\b', content.lower())
            word_counter = Counter(words)
            keywords = [word for word, count in word_counter.most_common(20) if len(word) > 3]
            print(f"Palabras clave extraídas: {keywords}")  # Depuración
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            Analiza el texto proporcionado y las palabras clave para devolver un JSON con:
                            - 'tone': tono del texto (e.g., formal, informal, técnico, académico, persuasivo, instructivo, analítico, narrativo, descriptivo).
                            - 'document_purpose': propósito del documento (e.g., informe, carta, manual, artículo, presentación, guía, propuesta, resumen).
                            - 'confidence': confianza en la predicción (0.0 a 1.0).
                            Usa el contexto, vocabulario y palabras clave para determinar el tono y propósito con alta precisión.
                            """
                        },
                        {
                            "role": "user",
                            "content": f"Texto: {content[:4000]}\nPalabras clave: {', '.join(keywords)}"
                        }
                    ],
                    max_tokens=500
                )
                analysis = json.loads(response.choices[0].message.content)
                style_profile['tone'] = analysis.get('tone', 'neutral')
                style_profile['document_purpose'] = analysis.get('document_purpose', 'general')
                style_profile['confidence_scores']['tone'] = analysis.get('confidence', 0.7)
                style_profile['confidence_scores']['document_purpose'] = analysis.get('confidence', 0.7)
                print(f"Análisis de tono: {style_profile['tone']}, propósito: {style_profile['document_purpose']}")  # Depuración
            except Exception as e:
                print(f"Error en análisis de tono: {e}")
                style_profile['tone'] = 'neutral'
                style_profile['document_purpose'] = 'general'
                style_profile['confidence_scores']['tone'] = 0.5
                style_profile['confidence_scores']['document_purpose'] = 0.5
        
        return style_profile, content
    
    except Exception as e:
        print(f"Error al analizar el documento: {e}")
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
        print("style_profile en /generate:", style_profile)  # Depuración

        # Usar el tono y la estructura del style_profile si están disponibles
        effective_tone = style_profile['tone'] if style_profile.get('tone') else tone
        structure = style_profile['structure'] if style_profile.get('structure') else ['paragraphs']
        structure_instruction = f"Usa una estructura que incluya: {', '.join(structure)} (e.g., headings, lists, paragraphs, tables, quotes, footnotes)."

        # Detectar si el prompt requiere una explicación estructurada
        is_explanatory = re.search(r'^(¿Quién es|¿Qué es|explicar|detallar)\b', prompt, re.IGNORECASE) is not None
        
        # Detectar si el prompt solicita HTML
        is_html = doc_type == 'html' or re.search(r'\b(html|página web|sitio web)\b', prompt, re.IGNORECASE) is not None
        
        if is_explanatory:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown con una estructura clara para explicaciones.
            - Tipo de documento: {doc_type} (e.g., explicación, biografía, informe).
            - Tono: {effective_tone} (formal, informal, técnico, académico, persuasivo, instructivo, analítico, narrativo, descriptivo).
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - {structure_instruction}
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
            - Usa encabezados (#, ##, ###), listas (-), negritas (**), tablas (|...|), citas (>), y notas al pie ([1]) cuando sea apropiado.
            - Considera el contexto de los mensajes anteriores para mantener coherencia en la conversación.
            """
        elif is_html:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera páginas web en formato HTML con CSS y JavaScript si es necesario.
            - Tipo de documento: página web (HTML).
            - Tono: {effective_tone} (formal, informal, técnico, académico, persuasivo, instructivo, analítico, narrativo, descriptivo).
            - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - Genera código HTML completo con:
              - Estructura semántica (<header>, <main>, <footer>, etc.).
              - Estilos CSS internos (en <style>) adaptados al tono y propósito.
              - JavaScript opcional (en <script>) si el prompt lo requiere.
              - Usa un diseño responsivo y moderno.
            - Considera el contexto de los mensajes anteriores para mantener coherencia.
            """
        else:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown.
            - Tipo de documento: {doc_type} (e.g., carta formal, informe, correo, contrato, currículum, guía, propuesta).
            - Tono: {effective_tone} (formal, informal, técnico, académico, persuasivo, instructivo, analítico, narrativo, descriptivo).
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
            - {structure_instruction}
            - Usa encabezados (#, ##, ###), listas (-), negritas (**), tablas (|...|), citas (>), y notas al pie ([1]) cuando sea apropiado.
            - Considera el contexto de los mensajes anteriores para mantener coherencia en la conversación.
            """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(message_history[-10:])
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Nota: Cambiar a "gpt-4o-mini" si está disponible
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
            'structure': json.loads(profile['structure']),
            'text_color': profile['text_color'],
            'background_color': profile['background_color'],
            'alignment': profile['alignment'],
            'line_spacing': profile['line_spacing'],
            'document_purpose': profile['document_purpose'],
            'confidence_scores': json.loads(profile['confidence_scores']),
            'font_styles': json.loads(profile['font_styles']),
            'heading_levels': profile['heading_levels']
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
        print("Aplicando style_profile en generate_preview:", style_profile)  # Depuración
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
            INSERT INTO style_profiles (
                user_id, font_name, font_size, tone, margins, structure, 
                text_color, background_color, alignment, line_spacing, 
                document_purpose, confidence_scores, font_styles, heading_levels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user.id,
            style_profile['font_name'],
            style_profile['font_size'],
            style_profile['tone'],
            json.dumps(style_profile['margins']),
            json.dumps(style_profile['structure']),
            style_profile['text_color'],
            style_profile['background_color'],
            style_profile['alignment'],
            style_profile['line_spacing'],
            style_profile['document_purpose'],
            json.dumps(style_profile['confidence_scores']),
            json.dumps(style_profile['font_styles']),
            style_profile['heading_levels']
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
            'structure': json.loads(profile['structure']),
            'text_color': profile['text_color'],
            'background_color': profile['background_color'],
            'alignment': profile['alignment'],
            'line_spacing': profile['line_spacing'],
            'document_purpose': profile['document_purpose'],
            'confidence_scores': json.loads(profile['confidence_scores']),
            'font_styles': json.loads(profile['font_styles']),
            'heading_levels': profile['heading_levels']
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