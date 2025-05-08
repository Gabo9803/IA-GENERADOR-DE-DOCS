import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
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
from collections import Counter
import statistics
import uuid
import base64
import cv2
import numpy as np
from PIL import Image
import io
import stripe

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configurar Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Configurar Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY")

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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

# Definir la versión actual del esquema
CURRENT_SCHEMA_VERSION = 5

def get_db_connection():
    db_path = os.getenv('RENDER_DB_PATH', 'profiles.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    conn = get_db_connection()
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
            text_color TEXT DEFAULT "#000000",
            background_color TEXT DEFAULT "#FFFFFF",
            alignment TEXT DEFAULT "left",
            line_spacing REAL DEFAULT 1.33,
            document_purpose TEXT DEFAULT "general",
            confidence_scores TEXT DEFAULT "{}",
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Tabla para sesiones y mensajes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            is_user BOOLEAN NOT NULL,
            content_type TEXT DEFAULT "pdf",
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    # Tabla para suscripciones y uso
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            user_id INTEGER PRIMARY KEY,
            stripe_subscription_id TEXT,
            plan TEXT,
            usage_count INTEGER DEFAULT 0,
            last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT "pending",
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def migrate_db():
    """Verifica y migra la base de datos a la versión actual."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT version FROM schema_version WHERE id = 1')
    result = cursor.fetchone()
    if result:
        current_version = result[0]
    else:
        current_version = 0
        cursor.execute('INSERT INTO schema_version (id, version) VALUES (1, ?)', (current_version,))
        conn.commit()
    
    if current_version < 1:
        current_version = 1
    
    if current_version < 2:
        cursor.execute('PRAGMA table_info(style_profiles)')
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'text_color' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN text_color TEXT DEFAULT "#000000"')
        if 'background_color' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN background_color TEXT DEFAULT "#FFFFFF"')
        if 'alignment' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN alignment TEXT DEFAULT "left"')
        if 'line_spacing' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN line_spacing REAL DEFAULT 1.33')
        if 'document_purpose' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN document_purpose TEXT DEFAULT "general"')
        if 'confidence_scores' not in columns:
            cursor.execute('ALTER TABLE style_profiles ADD COLUMN confidence_scores TEXT DEFAULT "{}"')
        current_version = 2
    
    if current_version < 3:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                is_user BOOLEAN NOT NULL,
                content_type TEXT DEFAULT "pdf",
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        current_version = 3
    
    if current_version < 4:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                stripe_subscription_id TEXT,
                plan TEXT,
                usage_count INTEGER DEFAULT 0,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT "pending",
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        current_version = 4
    
    if current_version < 5:
        # Asegurar que todos los usuarios existentes tengan una suscripción gratuita
        cursor.execute('SELECT id FROM users')
        users = cursor.fetchall()
        for user in users:
            user_id = user['id']
            cursor.execute('SELECT user_id FROM subscriptions WHERE user_id = ?', (user_id,))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO subscriptions (user_id, plan, status, usage_count, last_reset) VALUES (?, ?, ?, ?, ?)',
                              (user_id, 'free', 'active', 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        current_version = 5
    
    cursor.execute('UPDATE schema_version SET version = ? WHERE id = 1', (CURRENT_SCHEMA_VERSION,))
    conn.commit()
    conn.close()

init_db()
migrate_db()

# Cache para respuestas y vistas previas HTML
response_cache = {}
html_previews = {}

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
        'confidence_scores': {'font_name': 1.0, 'font_size': 1.0, 'tone': 0.8, 'margins': 1.0, 'structure': 0.8}
    }

def parse_markdown_to_reportlab(text, style_profile=None):
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

    table_pattern = re.compile(r'\|(.+?)\|\n\|[-|:\s]+\|\n((?:\|.+?\|(?:\n|$))+)')
    text_parts = table_pattern.split(text)
    
    for i, part in enumerate(text_parts):
        if i % 3 == 2:
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
        elif i % 3 != 1:
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

def analyze_image(file):
    """Analiza una imagen para extraer texto usando OCR con OpenAI Vision y determinar estilo."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT plan FROM subscriptions WHERE user_id = ?', (current_user.id,))
    subscription = cursor.fetchone()
    conn.close()
    
    if not subscription or subscription['plan'] == 'free':
        raise ValueError("OCR en imágenes requiere una suscripción de pago (básica, media o premium).")
    
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
            'font_name': 0.7,
            'font_size': 0.7,
            'tone': 0.6,
            'margins': 0.7,
            'structure': 0.6,
            'text_color': 0.7,
            'background_color': 0.7,
            'alignment': 0.6,
            'line_spacing': 0.6,
            'document_purpose': 0.5
        }
    }

    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un asistente que extrae texto de imágenes y analiza su estilo.
                    - Extrae todo el texto visible en la imagen.
                    - Analiza el estilo y devuelve un JSON con:
                      - 'font_name': Nombre de la fuente aproximada (e.g., Helvetica, Times).
                      - 'font_size': Tamaño de fuente aproximado en puntos.
                      - 'text_color': Color del texto en formato hexadecimal.
                      - 'background_color': Color de fondo en formato hexadecimal.
                      - 'alignment': Alineación del texto (left, center, right, justified).
                      - 'line_spacing': Espaciado entre líneas (aproximado, como factor).
                      - 'tone': Tono del texto (formal, informal, técnico).
                      - 'document_purpose': Propósito del documento (informe, carta, etc.).
                      - 'confidence': Confianza en las predicciones (0.0 a 1.0).
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extrae el texto y analiza el estilo de esta imagen."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        if result.startswith("```json"):
            result = result[7:-3].strip()
        analysis = json.loads(result)
        
        content = analysis.get('text', '')
        style_profile['font_name'] = analysis.get('font_name', 'Helvetica')
        style_profile['font_size'] = analysis.get('font_size', 12)
        style_profile['text_color'] = analysis.get('text_color', '#000000')
        style_profile['background_color'] = analysis.get('background_color', '#FFFFFF')
        style_profile['alignment'] = analysis.get('alignment', 'left')
        style_profile['line_spacing'] = analysis.get('line_spacing', 1.33)
        style_profile['tone'] = analysis.get('tone', 'neutral')
        style_profile['document_purpose'] = analysis.get('document_purpose', 'general')
        
        confidence = analysis.get('confidence', {})
        for key in style_profile['confidence_scores']:
            style_profile['confidence_scores'][key] = confidence.get(key, style_profile['confidence_scores'][key])
        
        if content:
            structures = set()
            if re.search(r'^\s*[-*]\s+', content, re.MULTILINE):
                structures.add('lists')
            if re.search(r'^\s*[A-Z].*\n[-=]+', content, re.MULTILINE):
                structures.add('headings')
            if re.search(r'\n\n+', content):
                structures.add('paragraphs')
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.8 if structures else 0.6
        
        return style_profile, content
    
    except Exception as e:
        print(f"Error en OCR de imagen: {e}")
        return style_profile, ""

def analyze_document(file):
    """Analiza un documento subido para extraer estilo, formato y tipografía."""
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
            'font_name': 0.8,
            'font_size': 0.8,
            'tone': 0.7,
            'margins': 0.8,
            'structure': 0.7,
            'text_color': 0.8,
            'background_color': 0.8,
            'alignment': 0.7,
            'line_spacing': 0.7,
            'document_purpose': 0.6
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
                        structures.add('lists')
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
            structures.add('lists')
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
                        - 'tone': tono del texto (e.g., formal, informal, técnico, académico, persuasivo, narrativo).
                        - 'document_purpose': propósito del documento (e.g., informe, carta, manual, artículo, presentación).
                        - 'confidence': confianza en la predicción (0.0 a 1.0).
                        Usa el contexto y vocabulario para determinar el tono y propósito.
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
            print(f"Error en análisis de tono: {e}")
    
    font_mapping = {
        'Times New Roman': 'Times-Roman',
        'Arial': 'Helvetica',
        'Courier New': 'Courier',
    }
    style_profile['font_name'] = font_mapping.get(style_profile['font_name'], style_profile['font_name'])
    
    return style_profile, content

def check_usage_limit():
    """Verifica y actualiza el uso del usuario según su plan."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT plan, usage_count, last_reset, status FROM subscriptions WHERE user_id = ?', (current_user.id,))
    subscription = cursor.fetchone()
    
    if not subscription:
        conn.close()
        raise ValueError("No tienes una suscripción activa. Suscríbete para continuar.")
    
    if subscription['status'] != 'active':
        conn.close()
        raise ValueError("Tu suscripción no está activa. Completa el proceso de suscripción para continuar.")
    
    plan, usage_count, last_reset = subscription['plan'], subscription['usage_count'], subscription['last_reset']
    last_reset_date = datetime.strptime(last_reset, '%Y-%m-%d %H:%M:%S')
    current_date = datetime.now()
    
    # Reiniciar contador si ha pasado un mes
    if (current_date - last_reset_date).days >= 30:
        usage_count = 0
        cursor.execute('UPDATE subscriptions SET usage_count = ?, last_reset = ? WHERE user_id = ?',
                       (0, current_date.strftime('%Y-%m-%d %H:%M:%S'), current_user.id))
        conn.commit()
    
    limits = {
        'free': 50,
        'basic': 100,
        'medium': 500,
        'premium': float('inf')
    }
    limit = limits.get(plan, 0)
    
    if usage_count >= limit:
        conn.close()
        raise ValueError(f"Has alcanzado el límite de uso de tu plan ({plan}: {limit} documentos/mes).")
    
    cursor.execute('UPDATE subscriptions SET usage_count = ? WHERE user_id = ?',
                   (usage_count + 1, current_user.id))
    conn.commit()
    conn.close()
    return True

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
            user_id = cursor.lastrowid
            # Asignar suscripción gratuita al nuevo usuario
            cursor.execute('INSERT INTO subscriptions (user_id, plan, status, usage_count, last_reset) VALUES (?, ?, ?, ?, ?)',
                          (user_id, 'free', 'active', 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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

@app.route('/subscribe', methods=['GET'])
@login_required
def subscribe_page():
    return render_template('subscribe.html', stripe_public_key=STRIPE_PUBLIC_KEY)

@app.route('/create-subscription', methods=['POST'])
@login_required
def create_subscription():
    try:
        data = request.json
        plan = data['plan']
        
        price_map = {
            'basic': 'price_1RMPYvDKDJaukVa6VkwjKGPy',
            'medium': 'price_1RMPZEDKDJaukVa6EjrpeQX7',
            'premium': 'price_1RMPZODKDJaukVa6pDmChEZD'
        }
        price_id = price_map.get(plan)
        if not price_id:
            return jsonify({'error': 'Plan no válido'}), 400
        
        customer = stripe.Customer.create(
            email=current_user.email
        )
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:5000/subscription-success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:5000/subscribe',
            customer=customer.id,
        )
        
        # Guardar la sesión temporalmente, pero no activar la suscripción aún
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO subscriptions (user_id, stripe_subscription_id, plan, status) VALUES (?, ?, ?, ?)',
                       (current_user.id, checkout_session.id, plan, 'pending'))
        conn.commit()
        conn.close()
        
        return jsonify({'sessionId': checkout_session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/subscription-success')
@login_required
def subscription_success():
    session_id = request.args.get('session_id')
    if session_id:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == 'paid':
            subscription = stripe.Subscription.retrieve(session.subscription)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE subscriptions SET stripe_subscription_id = ?, status = ? WHERE user_id = ?',
                          (subscription.id, 'active', current_user.id))
            conn.commit()
            conn.close()
            return "¡Suscripción exitosa! Ahora puedes usar el servicio según tu plan."
        else:
            return "El pago no se completó correctamente.", 400
    return "Error al procesar la suscripción.", 400

@app.route('/webhook', methods=['POST'])
def webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError as e:
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        return jsonify({'error': 'Invalid signature'}), 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        if session['payment_status'] == 'paid':
            subscription = stripe.Subscription.retrieve(session['subscription'])
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE subscriptions SET stripe_subscription_id = ?, status = ?, last_reset = ? WHERE user_id = ?',
                          (subscription.id, 'active', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_user.id))
            conn.commit()
            conn.close()

    return jsonify({'status': 'success'}), 200

@app.route('/subscription-cancel')
@login_required
def subscription_cancel():
    return "Suscripción cancelada. Puedes intentarlo de nuevo."

@app.route('/check-usage', methods=['GET'])
@login_required
def check_usage():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT plan, usage_count, last_reset, status FROM subscriptions WHERE user_id = ?', (current_user.id,))
    subscription = cursor.fetchone()
    conn.close()
    
    if not subscription:
        return jsonify({'error': 'No tienes una suscripción activa.'}), 403
    
    if subscription['status'] != 'active':
        return jsonify({'error': 'Tu suscripción no está activa.'}), 403
    
    plan, usage_count, last_reset = subscription['plan'], subscription['usage_count'], subscription['last_reset']
    last_reset_date = datetime.strptime(last_reset, '%Y-%m-%d %H:%M:%S')
    current_date = datetime.now()
    
    if (current_date - last_reset_date).days >= 30:
        usage_count = 0
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE subscriptions SET usage_count = ?, last_reset = ? WHERE user_id = ?',
                       (0, current_date.strftime('%Y-%m-%d %H:%M:%S'), current_user.id))
        conn.commit()
        conn.close()
    
    limits = {
        'free': 50,
        'basic': 100,
        'medium': 500,
        'premium': float('inf')
    }
    limit = limits.get(plan, 0)
    return jsonify({'usage_count': usage_count, 'limit': limit, 'remaining': limit - usage_count})

@app.route('/generate', methods=['POST'])
@login_required
def generate_document():
    try:
        check_usage_limit()
        
        data = request.json
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        doc_type = data.get('doc_type', 'general')
        tone = data.get('tone', 'neutral')
        length = data.get('length', 'medium')
        language = data.get('language', 'es')
        style_profile_id = data.get('style_profile_id', None)
        message_history = data.get('message_history', [])
        model = data.get('model', 'gpt-3.5-turbo')

        allowed_models = ['gpt-3.5-turbo', 'gpt-4o']
        if model not in allowed_models:
            return jsonify({'error': f'Modelo no válido. Usa uno de: {allowed_models}'}), 400

        if not prompt:
            return jsonify({'error': 'El prompt está vacío'}), 400

        cache_key = f"{current_user.id}:{session_id}:{prompt}:{doc_type}:{tone}:{length}:{language}:{style_profile_id}:{model}"
        if cache_key in response_cache:
            return jsonify({
                'document': response_cache[cache_key]['document'],
                'content_type': response_cache[cache_key]['content_type'],
                'includes_css': response_cache[cache_key].get('includes_css', False)
            })

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM sessions WHERE id = ? AND user_id = ?', (session_id, current_user.id))
        if not cursor.fetchone():
            cursor.execute('INSERT INTO sessions (id, user_id, name) VALUES (?, ?, ?)', 
                          (session_id, current_user.id, f"Sesión {session_id}"))
            conn.commit()
        
        style_profile = get_style_profile(style_profile_id)
        
        is_explanatory = re.search(r'^(¿Quién es|¿Qué es|explicar|detallar)\b', prompt, re.IGNORECASE) is not None
        is_html = doc_type == 'html' or re.search(r'\b(html|página web|sitio web)\b', prompt, re.IGNORECASE) is not None
        requests_css = re.search(r'\b(css|estilo|diseño|con estilo)\b', prompt, re.IGNORECASE) is not None
        
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
            if requests_css:
                system_prompt = f"""
                Eres GarBotGPT, un asistente que genera páginas web completas en formato HTML con CSS embebido.
                - Tipo de documento: página web (HTML con CSS).
                - Tono: {tone} (formal, informal, técnico).
                - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
            - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
                - Genera un archivo HTML completo autocontenido con:
                  - Estructura semántica completa (<header>, <nav>, <main>, <section>, <footer>, etc.).
                  - Metaetiquetas esenciales (<meta charset="UTF-8">, viewport, title).
                  - Estilos CSS embebidos dentro de una etiqueta <style> (NO uses enlaces a archivos CSS externos).
                    - Diseño responsivo usando flexbox o grid.
                    - Paleta de colores moderna y coherente.
                    - Soporte para temas claro y oscuro usando variables CSS (--variable-name).
                    - Transiciones y animaciones suaves para interactividad (e.g., hover effects).
                    - Tipografía profesional (usar fuentes de Google Fonts si es necesario, e.g., 'Roboto', importadas via @import).
                    - Sombras, bordes redondeados y otros efectos visuales modernos.
                    - Accesibilidad (ARIA roles, contraste adecuado).
                  - JavaScript opcional (en <script>) si el prompt lo requiere.
                - Asegúrate de que el CSS esté bien organizado y comentado dentro de <style>.
                - Estilo: {style_profile}.
                - Considera el contexto de los mensajes anteriores para mantener coherencia.
                """
            else:
                system_prompt = f"""
                Eres GarBotGPT, un asistente que genera páginas web en formato HTML sin CSS.
                - Tipo de documento: página web (solo HTML).
                - Tono: {tone} (formal, informal, técnico).
                - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
                - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
                - Genera código HTML completo con:
                  - Estructura semántica (<header>, <main>, <footer>, etc.).
                  - Metaetiquetas esenciales (<meta charset="UTF-8">, viewport, title).
                  - No incluyas CSS ni estilos inline a menos que se indique explícitamente.
                  - JavaScript opcional (en <script>) si el prompt lo requiere.
                - Asegúrate de que el HTML sea válido y semántico.
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
            model=model,
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )

        document = response.choices[0].message.content
        content_type = 'html' if is_html else 'pdf'
        
        response_cache[cache_key] = {
            'document': document,
            'content_type': content_type,
            'includes_css': requests_css if is_html else False
        }

        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (session_id, content, is_user, content_type) VALUES (?, ?, ?, ?)',
                      (session_id, prompt, True, content_type))
        cursor.execute('INSERT INTO messages (session_id, content, is_user, content_type) VALUES (?, ?, ?, ?)',
                      (session_id, document, False, content_type))
        conn.commit()
        conn.close()

        socketio.emit('newMessage', {
            'sessionId': session_id,
            'content': prompt,
            'isUser': True,
            'contentType': content_type
        }, room=session_id)
        socketio.emit('newMessage', {
            'sessionId': session_id,
            'content': document,
            'isUser': False,
            'contentType': content_type
        }, room=session_id)

        return jsonify({
            'document': document,
            'content_type': content_type,
            'includes_css': requests_css if is_html else False
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 403
    except openai.AuthenticationError:
        return jsonify({'error': 'Error de autenticación con OpenAI. Verifica la clave API.'}), 401
    except openai.RateLimitError:
        return jsonify({'error': 'Límite de solicitudes alcanzado. Intenta de nuevo más tarde.'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_style_profile(profile_id):
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
            'confidence_scores': json.loads(profile['confidence_scores'])
        }
    return get_default_style_profile()

@app.route('/generate_preview', methods=['POST'])
@login_required
def generate_preview():
    try:
        check_usage_limit()
        
        data = request.json
        content = data.get('content', '')
        content_type = data.get('content_type', 'pdf')
        style_profile_id = data.get('style_profile_id', None)
        includes_css = data.get('includes_css', False)
        if not content:
            return jsonify({'error': 'El contenido está vacío'}), 400

        if content_type == 'html':
            preview_id = f"{current_user.id}:{uuid.uuid4()}"
            html_previews[preview_id] = content
            if len(html_previews) > 100:
                oldest_key = next(iter(html_previews))
                del html_previews[oldest_key]
            return jsonify({
                'preview_id': preview_id,
                'content_type': 'html',
                'includes_css': includes_css
            })
        
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
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview_html/<preview_id>', methods=['GET'])
@login_required
def preview_html(preview_id):
    if preview_id in html_previews and preview_id.startswith(f"{current_user.id}:"):
        html_content = html_previews[preview_id]
        if not html_content.strip().startswith('<!DOCTYPE html>'):
            html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body>{html_content}</body>
</html>"""
        return Response(html_content, mimetype='text/html')
    return jsonify({'error': 'Vista previa no encontrada'}), 404

@app.route('/analyze_document', methods=['POST'])
@login_required
def analyze_document_endpoint():
    try:
        check_usage_limit()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file']
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt') or 
                file.filename.endswith('.jpg') or file.filename.endswith('.jpeg') or file.filename.endswith('.png')):
            return jsonify({'error': 'Formato no soportado. Usa PDF, TXT o imagen (JPG, JPEG, PNG).'}), 400
        
        if file.filename.endswith(('.jpg', '.jpeg', '.png')):
            style_profile, content = analyze_image(file)
        else:
            style_profile, content = analyze_document(file)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO style_profiles (
                user_id, font_name, font_size, tone, margins, structure, 
                text_color, background_color, alignment, line_spacing, document_purpose, confidence_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(style_profile['confidence_scores'])
        ))
        profile_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'style_profile_id': str(profile_id),
            'style_profile': style_profile,
            'content_summary': content[:500]
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 403
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
            'confidence_scores': json.loads(profile['confidence_scores'])
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
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM style_profiles WHERE user_id = ?', (current_user.id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Todos los perfiles de estilo han sido eliminados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('joinSession')
def on_join(data):
    session_id = data['sessionId']
    join_room(session_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT content, is_user, content_type FROM messages WHERE session_id = ?', (session_id,))
    messages = cursor.fetchall()
    conn.close()
    for msg in messages:
        emit('newMessage', {
            'sessionId': session_id,
            'content': msg['content'],
            'isUser': msg['is_user'],
            'contentType': msg['content_type']
        }, room=session_id)

@socketio.on('newMessage')
def on_new_message(data):
    session_id = data['sessionId']
    content = data['content']
    is_user = data['isUser']
    content_type = data['contentType']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (session_id, content, is_user, content_type) VALUES (?, ?, ?, ?)',
                  (session_id, content, is_user, content_type))
    conn.commit()
    conn.close()
    
    emit('newMessage', data, room=session_id, broadcast=True)

@socketio.on('editMessage')
def on_edit_message(data):
    session_id = data['sessionId']
    old_content = data['oldContent']
    new_content = data['newContent']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE messages SET content = ? WHERE session_id = ? AND content = ? AND is_user = 1',
                  (new_content, session_id, old_content))
    conn.commit()
    conn.close()
    
    emit('editMessage', data, room=session_id, broadcast=True)

@socketio.on('clearChat')
def on_clear_chat(data):
    session_id = data['sessionId']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
    
    emit('clearChat', data, room=session_id, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))