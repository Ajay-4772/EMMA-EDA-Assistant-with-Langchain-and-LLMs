import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import sqlite3
import os
from datetime import datetime
from typing import Union, Dict, Any, List
import base64
from io import BytesIO, StringIO
import requests
import contextlib
import sys
import traceback

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4  # type: ignore[import]
    from reportlab.platypus import (  # type: ignore[import]
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        PageBreak,
        Table,
        TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore[import]
    from reportlab.lib.units import inch  # type: ignore[import]
    from reportlab.lib import colors  # type: ignore[import]
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # type: ignore[import]
    from PIL import Image as PILImage
    import plotly.io as pio
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Optional cloud import
try:
    import boto3  # type: ignore[import]
    from botocore.exceptions import ClientError  # type: ignore[import]
    _AWS_AVAILABLE = True
except Exception:
    _AWS_AVAILABLE = False
try:
    from google.cloud import storage as gcs_storage  # type: ignore[import]
    from google.api_core.exceptions import NotFound as GCSNotFound  # type: ignore[import]
    _GCS_AVAILABLE = True
except Exception:
    _GCS_AVAILABLE = False
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore[import]
    from azure.core.exceptions import ResourceNotFoundError  # type: ignore[import]
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False

# Database setup for chat history
def init_database():
    """Initialize SQLite database for chat history"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            file_name TEXT,
            data_shape TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_message(session_id: int, user_message: str, bot_response: str):
    """Save a chat message to database"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_messages (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
    ''', (session_id, user_message, str(bot_response)))
    conn.commit()
    conn.close()

def create_chat_session(title: str, file_name: str, data_shape: str) -> int:
    """Create a new chat session and return session ID"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (session_title, file_name, data_shape)
        VALUES (?, ?, ?)
    ''', (title, file_name, data_shape))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_chat_sessions() -> List[Dict]:
    """Get all chat sessions"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, session_title, timestamp, file_name, data_shape
        FROM chat_sessions
        ORDER BY timestamp DESC
    ''')
    sessions = []
    for row in cursor.fetchall():
        sessions.append({
            'id': row[0],
            'title': row[1],
            'timestamp': row[2],
            'file_name': row[3],
            'data_shape': row[4]
        })
    conn.close()
    return sessions

def get_chat_messages(session_id: int) -> List[Dict]:
    """Get all messages for a specific session"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_message, bot_response, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.append({
            'user': row[0],
            'bot': row[1],
            'timestamp': row[2]
        })
    conn.close()
    return messages

# Enhanced data loading functions with large file support
def load_data(uploaded_file) -> tuple[Union[pd.DataFrame, str], str]:
    """Load data from uploaded file with enhanced format support and large file handling"""
    file_name = uploaded_file.name.lower()
    file_size = uploaded_file.size
    
    try:
        # Show progress for large files
        if file_size > 100 * 1024 * 1024:  # 100MB
            st.info(f"📁 Loading large file ({file_size / (1024*1024*1024):.1f} GB)... This may take a moment.")
        
        if file_name.endswith('.csv'):
            # Enhanced CSV loading with chunking for large files
            if file_size > 500 * 1024 * 1024:  # 500MB
                # Use chunking for very large files
                chunk_size = int(st.session_state.get('chunk_size', 100000))
                chunks = []
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                # UI elements for streaming/progress
                progress_text = st.empty()
                preview_placeholder = st.empty()
                progress_bar = st.progress(0)
                processed_rows = 0
                start_time = datetime.now()
                est_total = None  # unknown total rows without a pre-pass
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        chunk_iter = pd.read_csv(uploaded_file, encoding=encoding, chunksize=chunk_size)
                        chunk_index = 0
                        for chunk in chunk_iter:
                            chunks.append(chunk)
                            chunk_index += 1
                            processed_rows += len(chunk)
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0
                            # Soft progress (cap at 0.95, finish at the end)
                            soft_progress = min(0.95, 0.05 + 0.02 * chunk_index)
                            progress_bar.progress(soft_progress)
                            eta = "calculating..." if rows_per_sec == 0 else f"~{int((file_size/ (1024*1024)) / max(1, (processed_rows/ chunk_size)))} chunks left"
                            progress_text.text(f"🔄 Processed ~{processed_rows:,} rows • Chunk {chunk_index} • {eta}")
                            
                            # Update streaming preview if enabled
                            if st.session_state.get('streaming_enabled'):
                                try:
                                    # Only preview up to the first 1000 rows to keep UI responsive
                                    preview_df = pd.concat(chunks, ignore_index=True)
                                    preview_placeholder.dataframe(preview_df.head(1000), use_container_width=True, height=400)
                                except Exception:
                                    preview_placeholder.dataframe(chunk.head(1000), use_container_width=True, height=400)
                        data = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        progress_text.text("✅ Completed loading all chunks.")
                        return data, "csv"
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any supported encoding")
            else:
                # Standard loading for smaller files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        data = pd.read_csv(uploaded_file, encoding=encoding)
                        return data, "csv"
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        elif file_name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep='\t')
            return data, "tsv"
        elif file_name.endswith(('.xlsx', '.xls')):
            # Enhanced Excel loading with memory optimization
            if file_size > 100 * 1024 * 1024:  # 100MB
                data = pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
            else:
                data = pd.read_excel(uploaded_file)
            return data, "excel"
        elif file_name.endswith('.json'):
            data = pd.read_json(uploaded_file)
            return data, "json"
        elif file_name.endswith('.parquet'):
            # Parquet is already optimized; just read directly
            data = pd.read_parquet(uploaded_file)
            return data, "parquet"
        elif file_name.endswith('.txt'):
            data = uploaded_file.read().decode('utf-8')
            return data, "txt"
        elif file_name.endswith('.pdf'):
            # For PDF files, extract text
            try:
                import PyPDF2  # type: ignore
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text, "pdf"
            except ImportError:
                st.error("PyPDF2 is not installed. Please install it using: pip install PyPDF2")
                return None, None
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
                return None, None
        else:
            st.error("Unsupported file format")
            return None, None
    except Exception as e:
        st.error(f"Error loading file: {e}")

        
        return None, None

def save_results(data: pd.DataFrame, filename: str = "analysis_results.csv"):
    """Save analysis results to file"""
    try:
        data.to_csv(filename, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return False

# PDF Report Generation Functions
def convert_plotly_to_image(plotly_fig, width=800, height=600):
    """Convert Plotly figure to PIL Image for PDF inclusion"""
    if not _PDF_AVAILABLE:
        return None
    
    try:
        # Convert plotly figure to PNG bytes
        img_bytes = plotly_fig.to_image(format="png", width=width, height=height)
        
        # Convert bytes to PIL Image
        img = PILImage.open(BytesIO(img_bytes))
        return img
    except Exception as e:
        st.error(f"Error converting plotly figure to image: {e}")
        return None

def create_pdf_report(chat_history: List[Dict], data: pd.DataFrame = None, filename: str = None) -> bytes:
    """Create a comprehensive PDF report with chat history, charts, and data analysis"""
    if not _PDF_AVAILABLE:
        raise ImportError("PDF generation libraries not available. Please install reportlab, Pillow, and kaleido.")
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        leftIndent=20,
        textColor=colors.darkgreen,
        fontName='Helvetica-Bold'
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leftIndent=40,
        textColor=colors.black
    )
    
    # Build PDF content
    story = []
    
    # Title
    story.append(Paragraph("EMMA Enhanced - Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Data overview if available
    if data is not None and isinstance(data, pd.DataFrame):
        story.append(Paragraph("Dataset Overview", heading_style))
        
        # Create data summary table
        data_summary = [
            ['Metric', 'Value'],
            ['Total Rows', f"{len(data):,}"],
            ['Total Columns', f"{len(data.columns):,}"],
            ['Numeric Columns', f"{len(data.select_dtypes(include=[np.number]).columns):,}"],
            ['Categorical Columns', f"{len(data.select_dtypes(include=['object']).columns):,}"],
            ['Memory Usage', f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"],
            ['Missing Values', f"{data.isnull().sum().sum():,}"]
        ]
        
        data_table = Table(data_summary, colWidths=[2*inch, 2*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(data_table)
        story.append(Spacer(1, 20))
    
    # Chat history
    story.append(Paragraph("Analysis Session", heading_style))
    story.append(Spacer(1, 12))
    
    for i, chat in enumerate(chat_history, 1):
        # Question
        story.append(Paragraph(f"Q{i}: {chat['user']}", question_style))
        
        # Answer
        bot_response = chat['bot']
        if isinstance(bot_response, dict):
            if "response" in bot_response:
                bot_response = bot_response["response"]
            else:
                bot_response = str(bot_response)
        
        # Clean up response text for PDF
        clean_response = strip_code_blocks(str(bot_response))
        clean_response = clean_response.replace('**', '').replace('*', '')
        
        story.append(Paragraph(clean_response, answer_style))
        
        # Embed table if present
        if "table" in chat and isinstance(chat["table"], pd.DataFrame) and not chat["table"].empty:
            try:
                df_tbl = chat["table"].copy()
                # Limit extremely wide tables
                max_cols = 6
                if df_tbl.shape[1] > max_cols:
                    df_tbl = df_tbl.iloc[:, :max_cols]
                table_data = [list(df_tbl.columns)] + df_tbl.head(20).values.tolist()
                pdf_table = Table(table_data, repeatRows=1)
                pdf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
                ]))
                story.append(Spacer(1, 8))
                story.append(pdf_table)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"[Table could not be included: {str(e)}]", styles['Normal']))
        
        # Add chart if available
        if "plot" in chat and chat["plot"] is not None:
            try:
                # Convert plotly figure to image
                img = convert_plotly_to_image(chat["plot"], width=600, height=400)
                if img:
                    # Save image to buffer
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    # Add image to PDF
                    story.append(Spacer(1, 12))
                    story.append(Paragraph("📊 Visualization:", styles['Normal']))
                    story.append(Image(img_buffer, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 20))
            except Exception as e:
                story.append(Paragraph(f"[Chart could not be included: {str(e)}]", styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Add page break every 3 Q&A pairs to avoid overcrowding
        if i % 3 == 0 and i < len(chat_history):
            story.append(PageBreak())
    
    # Add summary section
    story.append(PageBreak())
    story.append(Paragraph("Session Summary", heading_style))
    story.append(Spacer(1, 12))
    
    summary_text = f"""
    This report contains {len(chat_history)} questions and answers from your EMMA Enhanced analysis session.
    
    Key Features Demonstrated:
    • Natural language data analysis
    • Dynamic visualization generation
    • Statistical insights and recommendations
    • Interactive chart creation
    
    EMMA Enhanced provides comprehensive exploratory data analysis capabilities
    through conversational AI, making data analysis accessible to users of all skill levels.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def generate_session_pdf(chat_history: List[Dict], data: pd.DataFrame = None) -> bytes:
    """Generate PDF report for the current session"""
    if not chat_history:
        raise ValueError("No chat history available for PDF generation")
    
    return create_pdf_report(chat_history, data)

# Enhanced visualization functions
def _detect_columns_by_keywords(data: pd.DataFrame, keywords: list[str]) -> list[str]:
    found: list[str] = []
    lower_map = {c.lower(): c for c in data.columns}
    for kw in keywords:
        if kw.lower() in lower_map:
            found.append(lower_map[kw.lower()])
        else:
            # fuzzy contains
            for c in data.columns:
                if kw.lower() in c.lower():
                    found.append(c)
                    break
    return list(dict.fromkeys(found))

def _guess_group_and_value_cols(data: pd.DataFrame, prompt: str) -> tuple[str | None, str | None]:
    # Try to guess a categorical group and a numeric value based on the prompt and data
    cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    prompt_l = prompt.lower()

    # Prefer common retail columns
    cat_priority = ["brand", "category", "product", "city", "state", "segment", "pants_description", "name"]
    num_priority = ["price", "mrp", "amount", "revenue", "sales", "quantity", "rating", "ratings"]

    cat_candidates = _detect_columns_by_keywords(data, cat_priority)
    num_candidates = _detect_columns_by_keywords(data, num_priority)

    group_col = next((c for c in cat_candidates if c in cat_cols), cat_cols[0] if cat_cols else None)
    value_col = next((c for c in num_candidates if c in num_cols), num_cols[0] if num_cols else None)

    # High-level intent overrides for common entities
    # If the user mentions product / customer / category explicitly, prefer that as the grouping column
    if "product" in prompt_l:
        prod_cols = _detect_columns_by_keywords(data, ["product", "item", "sku", "name"])
        if prod_cols:
            group_col = prod_cols[0]
    if "customer" in prompt_l or "client" in prompt_l or "buyer" in prompt_l:
        cust_cols = _detect_columns_by_keywords(data, ["customer", "client", "buyer"])
        if cust_cols:
            group_col = cust_cols[0]
    if "category" in prompt_l or "segment" in prompt_l:
        cat_cols_kw = _detect_columns_by_keywords(data, ["category", "segment"])
        if cat_cols_kw:
            group_col = cat_cols_kw[0]

    # Prompt-driven overrides: if any column name appears literally in the prompt, prefer it
    for c in cat_cols:
        if c.lower() in prompt_l:
            group_col = c
            break
    for c in num_cols:
        if c.lower() in prompt_l:
            value_col = c
            break

    # overrides from prompt like "by X" or "vs"
    by_match = re.search(r"by\s+([a-zA-Z0-9_ ]+)", prompt_l)
    if by_match:
        by_name = by_match.group(1).strip()
        for c in data.columns:
            if by_name in c.lower():
                group_col = c
                break
    vs_match = re.search(r"(\w[\w\s]+)\s+vs\s+(\w[\w\s]+)", prompt_l)
    if vs_match:
        left_raw, right_raw = vs_match.group(1).strip(), vs_match.group(2).strip()
        # Use the last word as the core column hint so phrases like "scatter plot of quantity vs amount"
        # still correctly map to "Quantity" and "Amount"
        left = left_raw.split()[-1]
        right = right_raw.split()[-1]
        for c in data.columns:
            if left in c.lower():
                group_col = c
            if right in c.lower():
                value_col = c

    return group_col, value_col

def _format_currency_axis(fig, axis: str = "y", currency_columns: list[str] | None = None):
    if currency_columns is None:
        currency_columns = ["price", "mrp", "amount", "revenue", "sales", "total"]
    fig.update_layout(
        **{f"{axis}axis": dict(tickprefix="₹" if axis == "y" else None, separatethousands=True)}
    )

def _apply_common_layout(fig, title: str, x_label: str | None = None, y_label: str | None = None):
    fig.update_layout(
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=20, t=60, b=40),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    if x_label:
        fig.update_xaxes(title=x_label)
    if y_label:
        fig.update_yaxes(title=y_label)

def generate_viz_from_prompt(data: pd.DataFrame, prompt: str) -> go.Figure | None:
    """Deterministic viz generator. Returns a high-quality Plotly figure or None."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    prompt_l = prompt.lower()

    # Filters mentioned like "MRP less than 2000"
    working = data.copy()
    range_filters = [
        (r"mrp\s*<\s*(\d+)", "mrp", "<"),
        (r"price\s*<\s*(\d+)", "price", "<"),
        (r"mrp\s*>\s*(\d+)", "mrp", ">"),
        (r"price\s*>\s*(\d+)", "price", ">"),
    ]
    for pat, colkw, op in range_filters:
        m = re.search(pat, prompt_l)
        if m:
            val = float(m.group(1))
            # map to actual column
            cols = _detect_columns_by_keywords(working, [colkw])
            if cols:
                col = cols[0]
                if op == "<":
                    working = working[working[col] < val]
                else:
                    working = working[working[col] > val]

    # Determine chart type intent
    is_heatmap = any(k in prompt_l for k in ["heatmap", "correlation matrix", "corr"])
    is_scatter = any(k in prompt_l for k in ["scatter", "relationship", "vs "])
    is_hist = any(k in prompt_l for k in ["histogram", "distribution", "hist "])
    is_pie = any(k in prompt_l for k in ["pie", "proportion", "share"])
    is_line = any(k in prompt_l for k in ["line", "trend", "time series"])
    is_box = any(k in prompt_l for k in ["box plot", "boxplot", "box-plot", "outlier", "outliers"])
    is_bar = any(k in prompt_l for k in ["bar", "top", "bottom", "count", "by "])

    # Guess columns
    group_col, value_col = _guess_group_and_value_cols(working, prompt)

    # Heatmap (correlations)
    if is_heatmap and len(working.select_dtypes(include=[np.number]).columns) >= 2:
        corr = working.select_dtypes(include=[np.number]).corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
        _apply_common_layout(fig, "Correlation Heatmap")
        return fig

    # Scatter
    if is_scatter and len(working.select_dtypes(include=[np.number]).columns) >= 2:
        numeric = working.select_dtypes(include=[np.number]).columns.tolist()
        # If the prompt mentions a date/time column, prefer it on the x‑axis even if it's not numeric
        time_col = None
        if "date" in prompt_l or "time" in prompt_l:
            for c in working.columns:
                if np.issubdtype(working[c].dtype, np.datetime64):
                    time_col = c
                    break
        if time_col is not None:
            x = time_col
            y = value_col if value_col in numeric else (numeric[0] if numeric else None)
        else:
            x = group_col if group_col in numeric else numeric[0]
            y = value_col if value_col in numeric else (numeric[1] if len(numeric) > 1 else numeric[0])
        if y is None:
            return None
        fig = px.scatter(working, x=x, y=y, hover_data=[c for c in working.columns if c not in [x, y]])
        _apply_common_layout(fig, f"{x} vs {y}", x, y)
        return fig

    # Box plot (outliers / distribution)
    if is_box:
        # Prefer an explicitly-mentioned numeric column, otherwise use guessed numeric
        if value_col is None or value_col not in working.columns or not np.issubdtype(working[value_col].dtype, np.number):
            numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
            value_col = numeric_cols[0] if numeric_cols else None
        if value_col is None:
            return None

        # If the prompt says "by <something>", show grouped box plot when that column is categorical
        by_match = re.search(r"\bby\s+([a-zA-Z0-9_ ]+)", prompt_l)
        x_col = None
        if by_match:
            by_name = by_match.group(1).strip()
            for c in working.columns:
                if by_name in c.lower():
                    x_col = c
                    break
        if x_col is None and group_col and group_col in working.columns:
            # Use guessed categorical group if it's non-numeric and not too high-cardinality
            if not np.issubdtype(working[group_col].dtype, np.number) and working[group_col].nunique(dropna=False) <= 50:
                x_col = group_col

        if x_col:
            fig = px.box(working, x=x_col, y=value_col, points="outliers")
            _apply_common_layout(fig, f"Box plot of {value_col} by {x_col}", x_col, value_col)
        else:
            fig = px.box(working, y=value_col, points="outliers")
            _apply_common_layout(fig, f"Box plot of {value_col}", None, value_col)
        return fig

    # Histogram
    if is_hist and value_col is not None and value_col in working.columns and np.issubdtype(working[value_col].dtype, np.number):
        fig = px.histogram(working, x=value_col, nbins=30)
        _apply_common_layout(fig, f"Distribution of {value_col}", value_col, "Count")
        return fig

    # Pie
    if is_pie and group_col and value_col and np.issubdtype(working[value_col].dtype, np.number):
        agg = working.groupby(group_col, dropna=False)[value_col].sum().sort_values(ascending=False).head(12)
        fig = px.pie(values=agg.values, names=agg.index)
        _apply_common_layout(fig, f"{value_col} share by {group_col}")
        return fig

    # Line (try time-like x)
    if is_line:
        # pick a datetime-like column or index
        time_col = None
        for c in working.columns:
            if np.issubdtype(working[c].dtype, np.datetime64):
                time_col = c; break
        if time_col is None:
            for c in working.columns:
                if any(k in c.lower() for k in ["date", "time", "day", "month", "year"]):
                    try:
                        working[c] = pd.to_datetime(working[c], errors='coerce')
                        if working[c].notna().any():
                            time_col = c
                            break
                    except Exception:
                        pass
        if time_col and value_col and np.issubdtype(working[value_col].dtype, np.number):
            ts = working.dropna(subset=[time_col]).sort_values(time_col)
            fig = px.line(ts, x=time_col, y=value_col)
            _apply_common_layout(fig, f"{value_col} over time", time_col, value_col)
            return fig

    # Bar (default categorical aggregation)
    if group_col and value_col and np.issubdtype(working[value_col].dtype, np.number):
        # Refine group/value choice based on explicit words in the prompt
        if "product" in prompt_l:
            prod_cols = _detect_columns_by_keywords(working, ["product", "item", "sku", "name"])
            if prod_cols:
                group_col = prod_cols[0]
        if "customer" in prompt_l:
            cust_cols = _detect_columns_by_keywords(working, ["customer", "client", "buyer"])
            if cust_cols:
                group_col = cust_cols[0]
        if "category" in prompt_l:
            cat_cols = _detect_columns_by_keywords(working, ["category", "segment"])
            if cat_cols:
                group_col = cat_cols[0]
        if "quantity" in prompt_l or "qty" in prompt_l or "units" in prompt_l:
            qty_cols = _detect_columns_by_keywords(working, ["quantity", "qty", "units", "items"])
            if qty_cols:
                value_col = qty_cols[0]

        # Decide on aggregation: sum for totals/counts, mean otherwise
        use_sum = any(
            kw in prompt_l
            for kw in [
                "total",
                "sum",
                "count",
                "number of",
                "qty",
                "quantity",
                "units",
                "sold",
                "revenue",
                "sales",
                "amount",
            ]
        )
        if use_sum:
            agg = working.groupby(group_col, dropna=False)[value_col].sum()
            y_label = f"Total {value_col}"
            title_prefix = "Total"
        else:
            agg = working.groupby(group_col, dropna=False)[value_col].mean()
            y_label = f"Average {value_col}"
            title_prefix = "Average"

        # handle prompts: top N
        m = re.search(r"top\s*(\d+)", prompt_l)
        n = int(m.group(1)) if m else 12
        agg = agg.sort_values(ascending=False).head(n)
        fig = px.bar(
            x=agg.index.astype(str), y=agg.values, labels={"x": group_col, "y": y_label}
        )
        # Attach hover template (note: use literal %{x} / %{y}, so escape braces in f-string)
        hover_tmpl = f"{group_col}: %{{x}}<br>{value_col}: %{{y:,.2f}}<extra></extra>"
        fig.update_traces(hovertemplate=hover_tmpl)
        _apply_common_layout(fig, f"{title_prefix} {value_col} by {group_col}", group_col, value_col)
        if any(k in value_col.lower() for k in ["price", "mrp", "amount", "revenue", "sales", "total"]):
            _format_currency_axis(fig, "y")
        return fig

    return None

def create_visualization(data: pd.DataFrame, viz_type: str, x_col: str = None, y_col: str = None, title: str = "Chart"):
    """Create various types of visualizations"""
    if viz_type == "bar":
        fig = px.bar(data, x=x_col, y=y_col, title=title)
    elif viz_type == "pie":
        fig = px.pie(data, values=y_col, names=x_col, title=title)
    elif viz_type == "line":
        fig = px.line(data, x=x_col, y=y_col, title=title)
    elif viz_type == "scatter":
        fig = px.scatter(data, x=x_col, y=y_col, title=title)
    elif viz_type == "box":
        fig = px.box(data, x=x_col, y=y_col, title=title)
    elif viz_type == "histogram":
        fig = px.histogram(data, x=x_col, title=title)
    else:
        fig = px.bar(data, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    return fig

def get_download_link(fig, filename: str, file_type: str = "png"):
    """Generate download link for plotly figures"""
    if file_type == "png":
        img_bytes = fig.to_image(format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download PNG</a>'
    elif file_type == "pdf":
        img_bytes = fig.to_image(format="pdf")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF</a>'
    else:
        href = ""
    return href

# GROQ / OpenRouter-like API Integration - Enhanced for Dynamic Analysis
def ask_groq(prompt, model="openai/gpt-3.5-turbo", data=None, api_type="groq"):
    """Send a comprehensive prompt to various APIs for dynamic EDA analysis"""
    
    if api_type == "groq":
        return ask_groq_api(prompt, model, data)
    elif api_type == "ollama":
        return ask_ollama_api(prompt, model, data)
    elif api_type == "openai":
        return ask_openai_api(prompt, model, data)
    else:
        return f"❌ Unsupported API type: {api_type}"

def ask_groq_api(prompt, model="openai/gpt-3.5-turbo", data=None):
    """Send a comprehensive prompt to OpenRouter API for dynamic EDA analysis"""
    api_key = "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"  # Replace with your OpenRouter API key
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {data.select_dtypes(include=[np.number]).columns.tolist()}
- Categorical columns: {data.select_dtypes(include=['object']).columns.tolist()}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for conversational EDA like ChatGPT
        system_prompt = f"""You are EMMA, a friendly and intelligent data analysis assistant. Think of yourself as ChatGPT but specialized in data analysis. Be conversational, helpful, and engaging.

CORE PRINCIPLES:
1. **Conversational Style**: Talk like ChatGPT - friendly, clear, and engaging. Use natural language and be helpful.
2. **Smart Analysis**: Analyze the data dynamically and provide insights based on actual values.
3. **Context Awareness**: Remember previous questions and build on the conversation.
4. **Helpful Responses**: Provide actionable insights and explain things clearly.
5. **Visualization Only When Asked**: Only create charts when explicitly requested with words like "pie chart", "bar chart", "heatmap", "visualize", etc.
6. **No Automatic Charts**: Don't generate charts for general questions like "What's the average?" or "Who is the highest?" - just answer conversationally.
7. **No Code Explanations**: When creating visualizations, generate ONLY executable code, no text explanations or insights.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[Groq API error: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[Groq connection error: {str(e)}]"

def ask_ollama_api(prompt, model="mistral", data=None):
    """Send a comprehensive prompt to local Ollama API for dynamic EDA analysis"""
    url = "http://localhost:11434/api/generate"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for comprehensive EDA with code generation
        system_prompt = f"""You are EMMA, an expert Exploratory Data Analysis (EDA) assistant powered by Ollama. You are designed to think dynamically and provide comprehensive, accurate analysis based on the actual data provided.

CORE PRINCIPLES:
1. **Dynamic Thinking**: Never use predefined or hardcoded responses. Analyze the data in real-time and provide unique insights.
2. **Complete Analysis**: Always analyze the ENTIRE dataset, not just samples or subsets.
3. **Data-Driven Responses**: Base all answers on actual values, statistics, and patterns in the data.
4. **ChatGPT-Style Communication**: Be conversational, clear, and helpful. Use natural language with proper formatting.
5. **Professional EDA**: Provide statistical rigor with practical insights.
6. **Code Generation**: When asked for visualizations, ALWAYS provide executable Python code, not text descriptions.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_content}",
        "stream": False,
        "options": {
            "num_predict": 4000,  # Increased for more detailed responses
            "temperature": 0.3,   # Slightly higher for more conversational tone
            "top_k": 20,          # More token variety
            "top_p": 0.9          # Nucleus sampling
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)  # Increased timeout
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "[No response from model]")
        else:
            return f"[Ollama API error: {response.status_code} - {response.text}]"
    except requests.exceptions.Timeout:
        return "⏰ Sorry, I'm taking a bit longer than expected. Please try again!"
    except Exception as e:
        return f"❌ Ollama connection error: {str(e)}"

def ask_openai_api(prompt, model="gpt-3.5-turbo", data=None):
    """Send a comprehensive prompt to OpenAI API for dynamic EDA analysis"""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    url = "https://api.openai.com/v1/chat/completions"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for comprehensive EDA with code generation
        system_prompt = f"""You are EMMA, an expert Exploratory Data Analysis (EDA) assistant powered by OpenAI. You are designed to think dynamically and provide comprehensive, accurate analysis based on the actual data provided.

CORE PRINCIPLES:
1. **Dynamic Thinking**: Never use predefined or hardcoded responses. Analyze the data in real-time and provide unique insights.
2. **Complete Analysis**: Always analyze the ENTIRE dataset, not just samples or subsets.
3. **Data-Driven Responses**: Base all answers on actual values, statistics, and patterns in the data.
4. **ChatGPT-Style Communication**: Be conversational, clear, and helpful. Use natural language with proper formatting.
5. **Professional EDA**: Provide statistical rigor with practical insights.
6. **Code Generation**: When asked for visualizations, ALWAYS provide executable Python code, not text descriptions.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[OpenAI API error: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[OpenAI connection error: {str(e)}]"

def extract_code_from_response(response_text):
    """Extract and clean Python code from LLM response with enhanced robustness"""
    import re
    
    # Look for code blocks with more flexible patterns
    code_blocks = re.findall(r"```(?:python)?\n([\s\S]+?)```", response_text)
    if code_blocks:
        code = code_blocks[0]
    else:
        # Look for code blocks without language specification
        code_blocks = re.findall(r"```\n([\s\S]+?)```", response_text)
        if code_blocks:
            code = code_blocks[0]
        else:
            # Look for lines that contain Python code patterns
            lines = response_text.splitlines()
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                # Check if this looks like Python code
                if (line.startswith(('import', 'from', 'plt.', 'data.', 'fig', 'sns.', 'px.')) or 
                    line.startswith((' ', '\t')) or
                    '=' in line or
                    '(' in line or
                    line.endswith(')') or
                    'plt.' in line or
                    'data.' in line or
                    'fig.' in line or
                    'sns.' in line or
                    'px.' in line or
                    'seaborn.' in line):
                    code_lines.append(line)
                    in_code_block = True
                elif in_code_block and line == '':
                    code_lines.append(line)
                elif in_code_block and not line.startswith((' ', '\t')) and not any(keyword in line for keyword in ['plt.', 'data.', 'fig', 'import', 'sns.', 'px.']):
                    break
            
            if code_lines:
                code = '\n'.join(code_lines)
            else:
                # If no code found, try to generate basic visualization code based on the request
                return generate_fallback_code(response_text)
    
    # Clean the code
    cleaned_lines = []
    for line in code.splitlines():
        line = line.strip()
        # Skip imports, comments, and empty lines
        if (not line.startswith('import') and 
            not line.startswith('from') and 
            not line.startswith('#') and 
            not line.startswith('"""') and
            not line.startswith("'''") and
            line != '' and
            not line.startswith('fig.show()') and
            not line.startswith('plt.show()') and
            not line.startswith('print(') and
            not line.startswith('display(')):
            cleaned_lines.append(line)
    
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Fix common issues
    cleaned_code = cleaned_code.replace('df.', 'data.')
    cleaned_code = cleaned_code.replace('df[', 'data[')
    cleaned_code = cleaned_code.replace('df(', 'data(')
    cleaned_code = cleaned_code.replace('.show()', '')
    
    # Fix plotly references
    cleaned_code = cleaned_code.replace('plotly.express', 'px')
    cleaned_code = cleaned_code.replace('plotly.graph_objects', 'go')
    cleaned_code = cleaned_code.replace('plotly.', 'px.')
    
    # Fix correlation issues
    if 'data.corr()' in cleaned_code and 'select_dtypes' not in cleaned_code:
        cleaned_code = cleaned_code.replace('data.corr()', 'data.select_dtypes(include=[np.number]).corr()')
    
    # Ensure proper variable assignments and figure creation
    lines = cleaned_code.split('\n')
    fixed_lines = []
    has_fig_assignment = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Fix common patterns
        if 'data.nlargest(5, \'salary\')' in line and '=' not in line:
            fixed_lines.append('top_earners = data.nlargest(5, \'salary\')')
        elif 'px.bar(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.bar(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.scatter(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.hist(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.plot(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        else:
            fixed_lines.append(line)
    
    cleaned_code = '\n'.join(fixed_lines)
    
    # Ensure we have a figure assignment at the end
    if not has_fig_assignment and not cleaned_code.endswith('fig = plt.gcf()'):
        cleaned_code += '\nfig = plt.gcf()'
    
    return cleaned_code

def generate_fallback_code(response_text):
    """Generate basic visualization code when LLM doesn't provide code"""
    import re
    
    # Extract keywords from the response to determine chart type
    response_lower = response_text.lower()
    
    # Determine chart type based on keywords
    if any(word in response_lower for word in ['heatmap', 'heat map', 'correlation matrix']):
        return """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 8))
if 'City' in data.columns and 'Product' in data.columns and 'Total_Amount' in data.columns:
    # Group data by City and Product, calculate average sales
    city_product_sales = data.groupby(['City', 'Product'])['Total_Amount'].mean().unstack()
    sns.heatmap(city_product_sales, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title("Average Sales by City and Product")
    plt.xlabel("Product")
    plt.ylabel("City")
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and len(data.select_dtypes(include=[np.number]).columns) > 1:
    # Create correlation heatmap for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
else:
    # Default heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
    else:
        plt.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Data Visualization')
plt.tight_layout()
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['pie', 'pie chart', 'proportion', 'percentage']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))
if 'Total_Amount' in data.columns and 'Category' in data.columns:
    # Group by category and sum amounts
    category_sales = data.groupby('Category')['Total_Amount'].sum()
    plt.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
    plt.title('Sales Proportion by Category')
elif 'Total_Amount' in data.columns and 'Product' in data.columns:
    # Group by product and sum amounts
    product_sales = data.groupby('Product')['Total_Amount'].sum()
    plt.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%')
    plt.title('Sales Proportion by Product')
elif 'salary' in data.columns and 'name' in data.columns:
    plt.pie(data['salary'], labels=data['name'], autopct='%1.1f%%')
    plt.title('Salary Distribution by Name')
elif 'age' in data.columns and 'name' in data.columns:
    plt.pie(data['age'], labels=data['name'], autopct='%1.1f%%')
    plt.title('Age Distribution by Name')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.pie(data[numeric_cols[0]], labels=data.index, autopct='%1.1f%%')
        plt.title(f'{numeric_cols[0]} Distribution')
plt.axis('equal')
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['bar', 'bar chart', 'count', 'highest', 'top', 'revenue', 'sales']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))
if 'Total_Amount' in data.columns and 'Category' in data.columns:
    # Group by category and sum amounts
    category_sales = data.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(category_sales.index, category_sales.values)
    plt.title('Total Sales by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and 'Product' in data.columns:
    # Group by product and sum amounts
    product_sales = data.groupby('Product')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(product_sales.index, product_sales.values)
    plt.title('Total Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and 'Customer_Name' in data.columns:
    # Group by customer and sum amounts
    customer_sales = data.groupby('Customer_Name')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(customer_sales.index, customer_sales.values)
    plt.title('Total Sales by Customer')
    plt.xlabel('Customer')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'salary' in data.columns:
    plt.bar(range(len(data)), data['salary'])
    plt.title('Salary Distribution')
    plt.xlabel('Employee Index')
    plt.ylabel('Salary')
elif 'age' in data.columns:
    plt.bar(range(len(data)), data['age'])
    plt.title('Age Distribution')
    plt.xlabel('Employee Index')
    plt.ylabel('Age')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.bar(range(len(data)), data[numeric_cols[0]])
        plt.title(f'{numeric_cols[0]} Distribution')
        plt.xlabel('Index')
        plt.ylabel(numeric_cols[0])
plt.tight_layout()
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['histogram', 'distribution', 'hist', 'average', 'quantity', 'items']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
if 'Quantity' in data.columns:
    plt.hist(data['Quantity'], bins=range(1, data['Quantity'].max()+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Quantity Distribution per Transaction')
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.xticks(range(1, data['Quantity'].max()+1))
elif 'Total_Amount' in data.columns:
    plt.hist(data['Total_Amount'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Total Amount Distribution')
    plt.xlabel('Total Amount')
    plt.ylabel('Frequency')
elif 'Unit_Price' in data.columns:
    plt.hist(data['Unit_Price'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Unit Price Distribution')
    plt.xlabel('Unit Price')
    plt.ylabel('Frequency')
elif 'salary' in data.columns:
    plt.hist(data['salary'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Salary Distribution Histogram')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
elif 'age' in data.columns:
    plt.hist(data['age'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Age Distribution Histogram')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.hist(data[numeric_cols[0]], bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'{numeric_cols[0]} Distribution Histogram')
        plt.xlabel(numeric_cols[0])
        plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['scatter', 'correlation', 'relationship']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.7)
    plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
elif 'salary' in data.columns and 'age' in data.columns:
    plt.scatter(data['age'], data['salary'], alpha=0.7)
    plt.title('Age vs Salary')
    plt.xlabel('Age')
    plt.ylabel('Salary')
else:
    plt.text(0.5, 0.5, 'Insufficient numeric data for scatter plot', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Data Visualization')
plt.grid(True, alpha=0.3)
fig = plt.gcf()"""
    
    else:
        # Default to a simple bar chart
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    plt.bar(range(len(data)), data[numeric_cols[0]])
    plt.title(f'{numeric_cols[0]} Distribution')
    plt.xlabel('Index')
    plt.ylabel(numeric_cols[0])
else:
    plt.text(0.5, 0.5, 'No numeric data available for visualization', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Data Overview')
plt.tight_layout()
fig = plt.gcf()"""

def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks and inline backticked code from text."""
    try:
        no_fenced = re.sub(r"```[\s\S]*?```", "[Code Block]", str(text))
        no_inline = re.sub(r"`[^`]+`", "", no_fenced)
        return no_inline
    except Exception:
        return str(text)

def _find_markdown_table_blocks(text: str) -> list[str]:
    """Find markdown table blocks in text."""
    try:
        pattern = re.compile(r"(?:^|\n)(\|.+\|[\s\S]*?)(?:\n\n|$)")
        return [m.group(1).strip() for m in pattern.finditer(text)]
    except Exception:
        return []

def parse_markdown_table(md_table: str) -> pd.DataFrame | None:
    """Parse a simple GitHub-style markdown table to DataFrame."""
    try:
        # Normalize lines and drop alignment row
        lines = [ln.strip() for ln in md_table.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        # Drop the separator/alignment row (second line)
        if re.match(r"^\|?\s*:?[-\s|:]+:?\s*\|?$", lines[1]):
            header = lines[0]
            rows = lines[2:]
        else:
            header = lines[0]
            rows = lines[1:]
        # Build CSV-like content
        def split_row(row: str) -> list[str]:
            parts = [p.strip() for p in row.strip('|').split('|')]
            return parts
        headers = split_row(header)
        data_rows = [split_row(r) for r in rows]
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    except Exception:
        return None

# Theme-aware CSS
def get_theme_css():
    """Get CSS for clean ChatGPT-like styling"""
    if st.session_state.dark_mode:
        return '''
        <style>
        /* Dark Theme - Clean like ChatGPT */
        .stApp {
            background-color: #343541 !important;
            color: #ffffff !important;
        }
        
        .main .block-container {
            background-color: #343541 !important;
            color: #ffffff !important;
            padding-top: 1rem;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #ffffff !important;
            font-weight: 400;
        }
        
        .stMarkdown label, .stTextInput label, .stFileUploader label {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1em !important;
        }
        
        .stButton>button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background: #0d8a6f;
        }
        
        .stTextInput>div>div>input {
            background: #40414f !important;
            color: #ffffff !important;
            border: 1px solid #565869 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-size: 0.9em !important;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #10a37f !important;
            outline: none !important;
        }
        
        .stFileUploader {
            background: #40414f !important;
            border: 2px dashed #565869 !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        
        .chat-bubble {
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 100%;
            font-size: 0.9em;
            background: #444654;
            color: #ffffff;
            border: none;
        }
        .user-bubble {
            background: #343541;
            color: #ffffff;
            margin-left: 0;
            font-weight: 400;
            border: 1px solid #565869;
        }
        .bot-bubble {
            background: #444654;
            color: #ffffff;
            margin-right: 0;
            border: none;
            font-weight: 400;
        }
        
        .stDataFrame {
            background: #40414f !important;
            color: #ffffff !important;
            border-radius: 6px;
            border: 1px solid #565869;
        }
        
        /* Form submit button styling */
        .stFormSubmitButton > button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
            width: 100%;
        }
        .stFormSubmitButton > button:hover {
            background: #0d8a6f;
        }
        
        /* Success and error messages */
        .stSuccess {
            background: #0c4a6e !important;
            border: 1px solid #0ea5e9 !important;
            color: #7dd3fc !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        
        .stError {
            background: #7f1d1d !important;
            border: 1px solid #f87171 !important;
            color: #fca5a5 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        </style>
        '''
    else:
        return '''
        <style>
        /* Light Theme - Clean like ChatGPT */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        .main .block-container {
            background-color: #ffffff !important;
            color: #000000 !important;
            padding-top: 1rem;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #000000 !important;
            font-weight: 600;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #000000 !important;
            font-weight: 400;
        }
        
        .stMarkdown label, .stTextInput label, .stFileUploader label {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 1em !important;
        }
        
        .stButton>button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background: #0d8a6f;
        }
        
        .stTextInput>div>div>input {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e5e5e5 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-size: 0.9em !important;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #10a37f !important;
            outline: none !important;
        }
        
        .stFileUploader {
            background: #f7f7f8 !important;
            border: 2px dashed #e5e5e5 !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        
        .chat-bubble {
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 100%;
            font-size: 0.9em;
            background: #ffffff;
            color: #000000;
            border: 1px solid #e5e5e5;
        }
        .user-bubble {
            background: #f7f7f8;
            color: #000000;
            margin-left: 0;
            font-weight: 400;
            border: 1px solid #e5e5e5;
        }
        .bot-bubble {
            background: #ffffff;
            color: #000000;
            margin-right: 0;
            border: 1px solid #e5e5e5;
            font-weight: 400;
        }
        
        .stDataFrame {
            background: #ffffff !important;
            color: #000000 !important;
            border-radius: 6px;
            border: 1px solid #e5e5e5;
        }
        
        /* Form submit button styling */
        .stFormSubmitButton > button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
            width: 100%;
        }
        .stFormSubmitButton > button:hover {
            background: #0d8a6f;
        }
        
        /* Success and error messages */
        .stSuccess {
            background: #f0f9ff !important;
            border: 1px solid #bae6fd !important;
            color: #0369a1 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        
        .stError {
            background: #fef2f2 !important;
            border: 1px solid #fecaca !important;
            color: #dc2626 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        </style>
        '''

def _load_from_s3(bucket: str, key: str) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from S3 into a DataFrame or text depending on extension."""
    if not _AWS_AVAILABLE:
        st.error("boto3 not installed. Install boto3 or use Local File.")
        return None, None
    try:
        s3 = boto3.client("s3")
        with st.spinner("Downloading from S3..."):
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
        name = key.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported S3 object type")
        return None, None
    except ClientError as e:
        st.error(f"S3 error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from S3: {e}")
        return None, None

def _load_from_gcs(bucket: str, blob_name: str) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from Google Cloud Storage into a DataFrame or text."""
    if not _GCS_AVAILABLE:
        st.error("google-cloud-storage not installed. Install it or use Local File.")
        return None, None
    try:
        client = gcs_storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        with st.spinner("Downloading from GCS..."):
            body = blob.download_as_bytes()
        name = blob_name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported GCS object type")
        return None, None
    except GCSNotFound as e:
        st.error(f"GCS error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from GCS: {e}")
        return None, None

def _load_from_azure(container: str, blob_name: str, connection_string: str | None = None) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from Azure Blob Storage into a DataFrame or text."""
    if not _AZURE_AVAILABLE:
        st.error("azure-storage-blob not installed. Install it or use Local File.")
        return None, None
    try:
        if not connection_string:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if not connection_string:
            st.error("Set AZURE_STORAGE_CONNECTION_STRING env var or provide it here.")
            return None, None
        service = BlobServiceClient.from_connection_string(connection_string)
        container_client = service.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        with st.spinner("Downloading from Azure Blob..."):
            body = blob_client.download_blob().readall()
        name = blob_name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported Azure blob type")
        return None, None
    except ResourceNotFoundError as e:
        st.error(f"Azure error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from Azure: {e}")
        return None, None

# Ensure fixed model silently (must be a valid model on your provider, e.g. OpenRouter)
if 'model_choice' not in st.session_state:
    # Default to a widely available chat model; you can change this to any valid slug
    # supported by your OpenRouter / Groq provider configuration.
    st.session_state.model_choice = "openai/gpt-3.5-turbo"

# Utility: clear all chat history
def _clear_all_history():
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_messages')
        cursor.execute('DELETE FROM chat_sessions')
        conn.commit()
        conn.close()
    except Exception:
        pass
    # Clear session
    st.session_state.pop('chat_history', None)
    st.session_state.pop('current_session_id', None)

def main():
    # Initialize database
    init_database()
    
    # Theme toggle in sidebar
    with st.sidebar:
        st.title("🎛️ EMMA Settings")
        
        # Theme toggle
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Chat history section
        st.subheader("📚 Chat History")
        
        # Clear all history
        if st.button("🧹 Clear All History"):
            _clear_all_history()
            st.success("History cleared.")
            st.rerun()
        
        # Search functionality
        search_term = st.text_input("🔍 Search chats:", placeholder="Enter keywords...")
        
        # Get chat sessions
        sessions = get_chat_sessions()
        
        if search_term:
            # Filter sessions based on search term
            filtered_sessions = []
            for session in sessions:
                if (search_term.lower() in session['title'].lower() or 
                    search_term.lower() in session['file_name'].lower()):
                    filtered_sessions.append(session)
            sessions = filtered_sessions
        
        # Display chat sessions
        for session in sessions:
            with st.expander(f"📄 {session['title']} ({session['timestamp'][:10]})"):
                st.write(f"**File:** {session['file_name']}")
                st.write(f"**Data Shape:** {session['data_shape']}")
                if st.button(f"Load Session {session['id']}", key=f"load_{session['id']}"):
                    st.session_state.current_session_id = session['id']
                    st.session_state.chat_history = get_chat_messages(session['id'])
                    st.rerun()
    
    # Apply theme CSS
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    st.title("🤖 Meet EMMA: Your EDA Assistant")
    st.write("Hi, I'm EMMA! Upload your data and ask questions in natural language. I'll help you explore, visualize, and understand your data with smart suggestions and insights! ✨")

    # Session controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Chat & Data", key="clear_chat"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("Download Session", key="download_session"):
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                if _PDF_AVAILABLE:
                    try:
                        # Generate PDF report with charts and plots
                        data = st.session_state.get('data') if isinstance(st.session_state.get('data'), pd.DataFrame) else None
                        pdf_bytes = generate_session_pdf(st.session_state.chat_history, data)
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"emma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        # Fallback to text download
                        summary = "EMMA Chat Session Summary\n" + "="*30 + "\n\n"
                        for i, chat in enumerate(st.session_state.chat_history, 1):
                            summary += f"Q{i}: {chat['user']}\n"
                            summary += f"A{i}: {chat['bot']}\n\n"
                        
                        st.download_button(
                            label="📥 Download Text Summary (Fallback)",
                            data=summary,
                            file_name=f"emma_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                else:
                    # PDF libraries not available, show text download
                    summary = "EMMA Chat Session Summary\n" + "="*30 + "\n\n"
                    for i, chat in enumerate(st.session_state.chat_history, 1):
                        summary += f"Q{i}: {chat['user']}\n"
                        summary += f"A{i}: {chat['bot']}\n\n"
                    
                    st.download_button(
                        label="📥 Download Text Summary",
                        data=summary,
                        file_name=f"emma_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    st.info("💡 Install reportlab, Pillow, and kaleido for PDF reports with charts!")
    with col3:
        if st.button("Export Data", key="export_data"):
            if 'data' in st.session_state and isinstance(st.session_state['data'], pd.DataFrame):
                csv = st.session_state['data'].to_csv(index=False)
                st.download_button(
                    label="📊 Download Dataset",
                    data=csv,
                    file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # File upload method controls
    st.markdown("---")
    st.subheader("📁 Data Ingestion")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        upload_method = st.radio(
            "Choose upload method:",
            ["Local File", "Cloud Storage", "Direct URL"],
            horizontal=True,
            help="Select how you want to bring data into EMMA"
        )
    with col_u2:
        streaming_enabled = st.checkbox("🔄 Streaming mode", help="Process data incrementally (for very large files)")
    if 'streaming_enabled' not in st.session_state:
        st.session_state.streaming_enabled = False
    st.session_state.streaming_enabled = streaming_enabled

    # Advanced ingestion controls
    adv_col1, adv_col2 = st.columns([1, 1])
    with adv_col1:
        chunk_size = st.slider(
            "Chunk size (rows per chunk for very large CSVs)",
            min_value=10000, max_value=500000, step=10000, value=int(st.session_state.get('chunk_size', 100000)),
            help="Applied when file > 500MB"
        )
    with adv_col2:
        st.caption("Tip: Larger chunks are faster but use more memory.")
    st.session_state['chunk_size'] = chunk_size

    # File upload with large file support
    if upload_method == "Local File":
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, JSON, TSV, Parquet, PDF, or TXT (Supports up to 1GB files)",
            type=["csv", "xlsx", "xls", "json", "tsv", "parquet", "pdf", "txt"],
            help="Large files are processed with chunking automatically"
        )
    elif upload_method == "Cloud Storage":
        platform = st.selectbox("Cloud Platform", ["AWS S3", "Google Cloud Storage", "Azure Blob"]) 
        if platform == "AWS S3":
            c1, c2 = st.columns(2)
            with c1:
                s3_bucket = st.text_input("S3 Bucket")
            with c2:
                s3_key = st.text_input("S3 Key (path/to/file.csv)")
            if st.button("Load from S3"):
                if s3_bucket and s3_key:
                    data, ftype = _load_from_s3(s3_bucket, s3_key)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from S3: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter both bucket and key.")
        elif platform == "Google Cloud Storage":
            c1, c2 = st.columns(2)
            with c1:
                gcs_bucket = st.text_input("GCS Bucket")
            with c2:
                gcs_blob = st.text_input("Blob (path/to/file.csv)")
            if st.button("Load from GCS"):
                if gcs_bucket and gcs_blob:
                    data, ftype = _load_from_gcs(gcs_bucket, gcs_blob)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from GCS: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter both bucket and blob name.")
        else:  # Azure
            c1, c2 = st.columns(2)
            with c1:
                az_container = st.text_input("Azure Container")
            with c2:
                az_blob = st.text_input("Blob (path/to/file.csv)")
            az_conn = st.text_input("Connection String (optional)", type="password")
            if st.button("Load from Azure"):
                if az_container and az_blob:
                    data, ftype = _load_from_azure(az_container, az_blob, az_conn if az_conn else None)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from Azure: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter container and blob path.")
        uploaded_file = None
    else:
        data_url = st.text_input("Enter direct file URL (CSV/JSON/Parquet)")
        if st.button("Load from URL"):
            st.info("Direct URL loading is coming soon. Download the file and use Local File for now.")
        uploaded_file = None

    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['file_type'] = None

    if uploaded_file is not None:
        with st.spinner("Loading your data..."):
            data, ftype = load_data(uploaded_file)
            if isinstance(data, pd.DataFrame):
                # Auto-convert obvious date-like columns for downstream analysis
                for col in data.columns:
                    if "date" in col.lower() and not np.issubdtype(data[col].dtype, np.datetime64):
                        try:
                            data[col] = pd.to_datetime(data[col], errors="coerce")
                        except Exception:
                            pass

                st.session_state['data'] = data
                st.session_state['file_type'] = ftype
                st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns):,} columns ({ftype})")

                # Enhanced info
                mem_kb = data.memory_usage(deep=True).sum() / 1024
                cols_num = len(data.select_dtypes(include=[np.number]).columns)
                cols_cat = len(data.select_dtypes(include=['object']).columns)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Rows", f"{len(data):,}")
                with c2: st.metric("Columns", f"{len(data.columns):,}")
                with c3: st.metric("Numeric Cols", f"{cols_num:,}")
                with c4: st.metric("Memory (KB)", f"{mem_kb:,.1f}")

                # Full data preview controls
                st.subheader("👀 Data Preview")
                show_full = st.checkbox("Show full dataset in preview (all rows & columns)", value=True)
                if show_full:
                    preview_df = data
                else:
                    r = st.slider("Rows", 5, min(1000, len(data)), min(100, len(data)))
                    c = st.slider("Columns", 5, len(data.columns), min(15, len(data.columns)))
                    preview_df = data.head(r).iloc[:, :c]
                st.dataframe(preview_df, use_container_width=True, height=min(600, max(300, len(preview_df) * 25)))

                # Note about streaming
                if streaming_enabled:
                    st.info("🔄 Streaming mode is enabled. For very large files, EMMA will process data incrementally.")
            else:
                # text/pdf fallback already handled in load_data
                st.session_state['data'] = data
                st.session_state['file_type'] = ftype

    # --- Core analytical reasoning helpers for EMMA ---

    def _find_column_by_keywords(df: pd.DataFrame, keywords: list[str], numeric_required: bool = False) -> str | None:
        """Best-effort column finder based on keyword matches."""
        if df is None or df.empty:
            return None
        lower_map = {c.lower(): c for c in df.columns}
        for kw in keywords:
            kw_l = kw.lower()
            # exact name first
            if kw_l in lower_map:
                col = lower_map[kw_l]
                if not numeric_required or np.issubdtype(df[col].dtype, np.number):
                    return col
            # then substring match
            for name_l, name in lower_map.items():
                if kw_l in name_l:
                    if not numeric_required or np.issubdtype(df[name].dtype, np.number):
                        return name
        if numeric_required:
            num_cols = df.select_dtypes(include=[np.number]).columns
            return num_cols[0] if len(num_cols) else None
        return None

    def _find_amount_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["amount", "total_amount", "sales", "revenue"], numeric_required=True)

    def _find_quantity_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["quantity", "qty", "units", "items"], numeric_required=True)

    def _find_category_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["category", "segment"], numeric_required=False)

    def _find_customer_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["customer", "client", "buyer"], numeric_required=False)

    def _find_product_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["product", "item", "sku", "name"], numeric_required=False)

    def _find_orderid_column(df: pd.DataFrame) -> str | None:
        return _find_column_by_keywords(df, ["orderid", "order_id", "order id"], numeric_required=False)

    def _find_date_column(df: pd.DataFrame) -> str | None:
        col = _find_column_by_keywords(df, ["date", "order_date"], numeric_required=False)
        if col is None:
            return None
        # Auto-convert to datetime if needed
        if not np.issubdtype(df[col].dtype, np.datetime64):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
        return col

    def analyze_query_structured(df: pd.DataFrame | None, prompt: str) -> Dict[str, Any]:
        """
        Deterministic reasoning pipeline for EMMA.
        Returns a structured dict:
        {
          "answer": ...,
          "explanation": ...,
          "table": optional DataFrame,
          "plot": optional Plotly figure
        }
        """
        result: Dict[str, Any] = {"answer": None, "explanation": "", "table": None, "plot": None}

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            result["explanation"] = "Please upload a dataset first so I can analyze it."
            return result

        q = prompt.lower()
        rows, cols = df.shape

        # Does the user clearly want a visualization?
        viz_keywords = [
            "plot", "chart", "graph", "visualize", "histogram", "bar chart", "scatter", "box plot",
            "heatmap", "line chart", "pie chart"
        ]
        wants_viz = any(k in q for k in viz_keywords)

        # 0) Missing values / data quality
        if "missing value" in q or "missing values" in q or "null" in q or "na " in q or "nan" in q:
            miss_counts = df.isnull().sum()
            total_missing = int(miss_counts.sum())
            if total_missing == 0:
                result["answer"] = False
                result["explanation"] = "There are **no missing values** in the dataset."
            else:
                miss_pct = (miss_counts / len(df) * 100.0).round(2)
                info_df = pd.DataFrame({
                    "column": df.columns,
                    "missing_count": miss_counts.values,
                    "missing_percent": miss_pct.values,
                })
                info_df = info_df[info_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
                result["answer"] = True
                result["table"] = info_df
                result["explanation"] = (
                    f"There are **{total_missing}** missing values across **"
                    f"{len(info_df)}** column(s). Details are shown in the table."
                )
            return result

        # 1) Dataset overview / structure
        if ("row" in q and "column" in q) or "rows and columns" in q or "shape" in q or "dataset information" in q:
            info_df = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "missing": df.isnull().sum().values,
                "unique_values": df.nunique().values,
            })
            result["answer"] = {"rows": rows, "columns": cols}
            msg_parts = [
                f"The dataset has **{rows}** rows and **{cols}** columns.",
                "A summary of column types, missing values, and cardinality is shown in the table."
            ]
            # Duplicates in key-like columns
            if "duplicate" in q or "duplicates" in q:
                key_col = _find_orderid_column(df)
                if key_col:
                    dup_count = df.duplicated(subset=[key_col]).sum()
                    msg_parts.append(
                        f"Column **{key_col}** has **{dup_count}** duplicate rows."
                        if dup_count
                        else f"Column **{key_col}** has no duplicates."
                    )
            # Date column check
            if "date" in q:
                date_col = _find_date_column(df)
                if date_col:
                    is_dt = np.issubdtype(df[date_col].dtype, np.datetime64)
                    msg_parts.append(
                        f"Column **{date_col}** is{' ' if is_dt else ' not '}in datetime format."
                    )
            result["table"] = info_df
            result["explanation"] = "\n\n".join(msg_parts)
            return result

        # 2) Total sales amount / total revenue (only when not asking per-category/customer/product)
        if (
            "total" in q
            and any(w in q for w in ["sale", "sales", "amount", "revenue"])
            and "category" not in q
            and "customer" not in q
            and "product" not in q
        ):
            amt_col = _find_amount_column(df)
            if amt_col:
                total_val = float(df[amt_col].sum())
                result["answer"] = total_val
                result["explanation"] = f"The total {amt_col} across all rows is **{total_val:,.2f}**."
            else:
                result["explanation"] = "I couldn't find a numeric amount/sales column to sum."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 3) Min / max sales amount
        if any(w in q for w in ["minimum", "maximum", "min ", "max "]) and any(
            w in q for w in ["sale", "sales", "amount", "revenue"]
        ):
            amt_col = _find_amount_column(df)
            if amt_col:
                min_val = float(df[amt_col].min())
                max_val = float(df[amt_col].max())
                result["answer"] = {"min": min_val, "max": max_val}
                result["explanation"] = (
                    f"The minimum {amt_col} is **{min_val:,.2f}**, and the maximum is **{max_val:,.2f}**."
                )
            else:
                result["explanation"] = "I couldn't find a numeric amount/sales column to compute min/max."
            return result

        # 4) Total quantity sold
        if "total" in q and any(w in q for w in ["quantity", "qty", "units", "items"]):
            qty_col = _find_quantity_column(df)
            if qty_col:
                total_qty = int(df[qty_col].sum())
                result["answer"] = total_qty
                result["explanation"] = f"The total {qty_col} sold is **{total_qty}**."
            else:
                result["explanation"] = "I couldn't find a quantity-like column to sum."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 5) Average order amount / order value (AOV)
        if (
            "average order amount" in q
            or "avg order amount" in q
            or "average order value" in q
            or "avg order value" in q
            or "aov" in q
        ):
            amt_col = _find_amount_column(df)
            order_col = _find_orderid_column(df)
            if amt_col and order_col:
                order_totals = df.groupby(order_col)[amt_col].sum()
                avg_order_amt = float(order_totals.mean())
                result["answer"] = avg_order_amt
                result["explanation"] = (
                    f"The average order amount (mean {amt_col} per order) is **{avg_order_amt:,.2f}**."
                )
            elif amt_col:
                avg_amt = float(df[amt_col].mean())
                result["answer"] = avg_amt
                result["explanation"] = (
                    f"I couldn't clearly identify an order ID column, so using all rows, "
                    f"the average {amt_col} is **{avg_amt:,.2f}**."
                )
            else:
                result["explanation"] = "I couldn't find an amount/sales column to compute the average order amount."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 6) Generic average of amount / sales / revenue
        if "average" in q and any(w in q for w in ["amount", "sales", "revenue", "price", "mrp", "value"]):
            amt_col = _find_amount_column(df)
            if amt_col:
                avg_amt = float(df[amt_col].mean())
                result["answer"] = avg_amt
                result["explanation"] = f"The average {amt_col} is **{avg_amt:,.2f}**."
            else:
                result["explanation"] = "I couldn't find an amount/sales column to compute the average."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 7) Average quantity per order
        if "average" in q and any(w in q for w in ["quantity", "qty", "units"]) and "per order" in q:
            qty_col = _find_quantity_column(df)
            if qty_col:
                avg_qty = float(df[qty_col].mean())
                result["answer"] = avg_qty
                result["explanation"] = f"The average {qty_col} per order is **{avg_qty:,.2f}**."
            else:
                result["explanation"] = "I couldn't find a quantity-like column to average."
            return result

        # 8) Category-level analysis (totals, averages, quantities, unique products)
        if "category" in q:
            cat_col = _find_category_column(df)
            if cat_col:
                amt_col = _find_amount_column(df)
                qty_col = _find_quantity_column(df)
                prod_col = _find_product_column(df)
                agg_map: Dict[str, Any] = {}
                if amt_col:
                    agg_map["total_amount"] = (amt_col, "sum")
                    agg_map["avg_amount"] = (amt_col, "mean")
                if qty_col:
                    agg_map["total_quantity"] = (qty_col, "sum")
                if prod_col:
                    agg_map["unique_products"] = (prod_col, "nunique")
                if agg_map:
                    cat_df = df.groupby(cat_col).agg(**agg_map).reset_index()
                    result["table"] = cat_df
                    # Highest metrics
                    msg_lines: list[str] = []
                    if "total sales" in q or "highest total sales" in q or "revenue" in q:
                        if "total_amount" in cat_df.columns:
                            top_row = cat_df.sort_values("total_amount", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The category with the highest total sales is **{top_row[cat_col]}** "
                                f"with **{top_row['total_amount']:,.2f}**."
                            )
                    if "average order value" in q or "average amount" in q:
                        if "avg_amount" in cat_df.columns:
                            top_row = cat_df.sort_values("avg_amount", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The category with the highest average order value is **{top_row[cat_col]}** "
                                f"with **{top_row['avg_amount']:,.2f}**."
                            )
                    if "sells the most quantity" in q or "most quantity" in q:
                        if "total_quantity" in cat_df.columns:
                            top_row = cat_df.sort_values("total_quantity", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The category selling the most quantity is **{top_row[cat_col]}** "
                                f"with **{int(top_row['total_quantity'])}** units."
                            )
                    if "unique products" in q:
                        if "unique_products" in cat_df.columns:
                            msg_lines.append("Unique products per category are shown in the table.")
                    if not msg_lines:
                        msg_lines.append("Category-level aggregates are shown in the table.")
                    result["explanation"] = "\n\n".join(msg_lines)
                else:
                    result["explanation"] = "I couldn't compute category-level metrics because key numeric columns are missing."
            else:
                result["explanation"] = "I couldn't find a category column in the dataset."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 7) Customer-level analysis (revenue, orders, average spending, top N)
        if "customer" in q:
            cust_col = _find_customer_column(df)
            if cust_col:
                amt_col = _find_amount_column(df)
                order_col = _find_orderid_column(df)
                # Revenue per customer
                revenue_df = None
                if amt_col:
                    revenue_df = df.groupby(cust_col)[amt_col].sum().rename("total_revenue").reset_index()
                # Orders per customer
                orders_df = None
                if order_col:
                    orders_df = df.groupby(cust_col)[order_col].nunique().rename("num_orders").reset_index()
                if revenue_df is not None and orders_df is not None:
                    merged = pd.merge(revenue_df, orders_df, on=cust_col, how="outer")
                elif revenue_df is not None:
                    merged = revenue_df
                elif orders_df is not None:
                    merged = orders_df
                else:
                    merged = None
                if merged is not None:
                    result["table"] = merged.sort_values(
                        "total_revenue" if "total_revenue" in merged.columns else merged.columns[1],
                        ascending=False
                    )
                msg_lines: list[str] = []
                # Highest revenue customer
                if "highest revenue" in q or "generated the highest revenue" in q:
                    if revenue_df is not None and not revenue_df.empty:
                        top_row = revenue_df.sort_values("total_revenue", ascending=False).iloc[0]
                        msg_lines.append(
                            f"The customer with the highest revenue is **{top_row[cust_col]}** "
                            f"with **{top_row['total_revenue']:,.2f}**."
                        )
                # Most orders
                if "most orders" in q:
                    if orders_df is not None and not orders_df.empty:
                        top_row = orders_df.sort_values("num_orders", ascending=False).iloc[0]
                        msg_lines.append(
                            f"The customer who placed the most orders is **{top_row[cust_col]}** "
                            f"with **{int(top_row['num_orders'])}** orders."
                        )
                # Average spending per customer
                if "average spending" in q or "average spend" in q:
                    if revenue_df is not None and not revenue_df.empty:
                        avg_spend = float(revenue_df["total_revenue"].mean())
                        msg_lines.append(
                            f"The average spending per customer is **{avg_spend:,.2f}**."
                        )
                # Top N customers by revenue
                m = re.search(r"top\s*(\d+)", q)
                if ("top" in q and "revenue" in q) or ("top" in q and "customers" in q):
                    n = int(m.group(1)) if m else 5
                    if revenue_df is not None and not revenue_df.empty:
                        top_n = revenue_df.sort_values("total_revenue", ascending=False).head(n)
                        result["table"] = top_n
                        msg_lines.append(
                            f"The top {len(top_n)} customers by revenue are shown in the table."
                        )
                if not msg_lines:
                    msg_lines.append("Customer-level aggregates are shown in the table.")
                result["explanation"] = "\n\n".join(msg_lines)
            else:
                result["explanation"] = "I couldn't find a customer column in the dataset."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 9) Product-level questions (most units, highest sales, highest average price)
        if "product" in q:
            prod_col = _find_product_column(df)
            amt_col = _find_amount_column(df)
            qty_col = _find_quantity_column(df)
            if prod_col:
                agg_map: Dict[str, Any] = {}
                if amt_col:
                    agg_map["total_amount"] = (amt_col, "sum")
                if qty_col:
                    agg_map["total_quantity"] = (qty_col, "sum")
                if amt_col and qty_col:
                    # average price = total_amount / total_quantity
                    tmp = df.copy()
                    tmp["__unit_price__"] = tmp[amt_col] / tmp[qty_col].replace(0, np.nan)
                    price_df = tmp.groupby(prod_col)["__unit_price__"].mean().rename("avg_price").reset_index()
                else:
                    price_df = None
                if agg_map:
                    prod_df = df.groupby(prod_col).agg(**agg_map).reset_index()
                    if price_df is not None:
                        prod_df = pd.merge(prod_df, price_df, on=prod_col, how="left")
                    result["table"] = prod_df
                    msg_lines: list[str] = []
                    if "sold the most" in q or "most units" in q or "most quantity" in q:
                        if "total_quantity" in prod_df.columns:
                            top_row = prod_df.sort_values("total_quantity", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The product that sold the most units is **{top_row[prod_col]}** "
                                f"with **{int(top_row['total_quantity'])}** units."
                            )
                    if "highest sales" in q or "generated the highest sales" in q:
                        if "total_amount" in prod_df.columns:
                            top_row = prod_df.sort_values("total_amount", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The product with the highest total sales is **{top_row[prod_col]}** "
                                f"with **{top_row['total_amount']:,.2f}**."
                            )
                    if "highest average price" in q or "highest price" in q or "average price" in q:
                        if "avg_price" in prod_df.columns:
                            top_row = prod_df.sort_values("avg_price", ascending=False).iloc[0]
                            msg_lines.append(
                                f"The product with the highest average price is **{top_row[prod_col]}** "
                                f"with an average price of **{top_row['avg_price']:,.2f}**."
                            )
                    if not msg_lines:
                        msg_lines.append("Product-level aggregates are shown in the table.")
                    result["explanation"] = "\n\n".join(msg_lines)
                else:
                    result["explanation"] = "I couldn't compute product-level metrics because key numeric columns are missing."
            else:
                result["explanation"] = "I couldn't find a product column in the dataset."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 10) Correlation between Quantity and Amount
        if "correlation" in q and "quantity" in q and any(w in q for w in ["amount", "sales", "revenue"]):
            amt_col = _find_amount_column(df)
            qty_col = _find_quantity_column(df)
            if amt_col and qty_col:
                corr_val = float(df[qty_col].corr(df[amt_col]))
                result["answer"] = corr_val
                result["explanation"] = (
                    f"The correlation between **{qty_col}** and **{amt_col}** is **{corr_val:.3f}**."
                )
            else:
                result["explanation"] = "I couldn't find both a quantity and an amount column to compute correlation."
            if wants_viz:
                result["plot"] = generate_viz_from_prompt(df, prompt)
            return result

        # 11) Percentage of revenue from top X% customers
        if "top" in q and "%" in q and any(w in q for w in ["revenue", "sales", "amount"]):
            cust_col = _find_customer_column(df)
            amt_col = _find_amount_column(df)
            if cust_col and amt_col:
                m = re.search(r"top\s*(\d+)\s*%", q)
                pct = int(m.group(1)) if m else 20
                revenue_per_cust = df.groupby(cust_col)[amt_col].sum().sort_values(ascending=False)
                total_rev = float(revenue_per_cust.sum())
                if total_rev == 0:
                    result["explanation"] = "Total revenue is zero; percentage contribution cannot be computed."
                else:
                    n_top = max(1, int(len(revenue_per_cust) * pct / 100.0))
                    top_rev = float(revenue_per_cust.head(n_top).sum())
                    pct_rev = (top_rev / total_rev) * 100.0
                    result["answer"] = pct_rev
                    result["explanation"] = (
                        f"The top **{pct}%** customers (by revenue) contribute **{pct_rev:.2f}%** "
                        f"of the total revenue."
                    )
                    result["table"] = revenue_per_cust.reset_index().head(n_top)
            else:
                result["explanation"] = "I couldn't find both customer and amount columns to compute the percentage."
            return result

        # 12) Customers with irregular buying patterns (based on order count z-score)
        if "irregular buying" in q or "irregular buying patterns" in q:
            cust_col = _find_customer_column(df)
            order_col = _find_orderid_column(df)
            if cust_col and order_col:
                counts = df.groupby(cust_col)[order_col].nunique()
                mean_c = counts.mean()
                std_c = counts.std(ddof=0)
                if std_c == 0 or np.isnan(std_c):
                    result["explanation"] = (
                        "Order counts per customer show almost no variation, so no irregular patterns are detected."
                    )
                else:
                    z_scores = (counts - mean_c) / std_c
                    irregular = z_scores[abs(z_scores) >= 1.0]  # 1 std dev away
                    if irregular.empty:
                        result["explanation"] = (
                            "No customers stand out as having highly irregular buying patterns based on order counts."
                        )
                    else:
                        irr_df = irregular.rename("z_score").reset_index()
                        result["table"] = irr_df
                        result["explanation"] = (
                            "Customers with notably irregular buying patterns (based on order-count z-scores) "
                            "are listed in the table."
                        )
            else:
                result["explanation"] = "I couldn't find both customer and order ID columns to analyze buying patterns."
            return result

        # 13) Month with highest sales (extract month from Date)
        if "month" in q and any(w in q for w in ["highest sales", "most sales", "highest revenue"]):
            date_col = _find_date_column(df)
            amt_col = _find_amount_column(df)
            if date_col and amt_col:
                tmp = df.dropna(subset=[date_col]).copy()
                if tmp.empty:
                    result["explanation"] = "All dates are missing or invalid; month-level sales cannot be computed."
                else:
                    tmp["__month__"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                    month_sales = tmp.groupby("__month__")[amt_col].sum().sort_values(ascending=False)
                    top_month, top_val = month_sales.index[0], float(month_sales.iloc[0])
                    result["answer"] = {"month": str(top_month.date()), "total_amount": top_val}
                    result["table"] = month_sales.reset_index().rename(columns={amt_col: "total_amount"})
                    result["explanation"] = (
                        f"The month with the highest sales is **{top_month.strftime('%Y-%m')}** "
                        f"with total {amt_col} of **{top_val:,.2f}**."
                    )
                    if wants_viz:
                        # Time-series style chart
                        fig = px.line(month_sales.reset_index(), x="__month__", y=amt_col, markers=True)
                        _apply_common_layout(fig, f"{amt_col} by month", "Month", amt_col)
                        result["plot"] = fig
            else:
                result["explanation"] = "I couldn't find both a date column and an amount column to compute monthly sales."
            return result

        # 14) Generic visualization-only requests – delegate to deterministic viz engine
        if wants_viz:
            fig = generate_viz_from_prompt(df, prompt)
            if fig is not None:
                result["plot"] = fig
                result["explanation"] = "Here is the visualization based on your request."
            else:
                result["explanation"] = (
                    "I couldn't confidently create a chart for this request. "
                    "Try being more specific about the columns you want to see."
                )
            return result

        # Fallback: unrecognized analytical intent
        result["explanation"] = (
            "I couldn't automatically recognize this request as a data operation. "
            "Try asking about totals, averages, correlations, top customers/products, categories, "
            "or distributions."
        )
        return result

    # Chat interface
    st.markdown("### 💬 Chat with EMMA")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f'<div class="chat-bubble user-bubble">{chat["user"]}</div>', unsafe_allow_html=True)
        
        # Bot response
        bot_response = chat["bot"]
        
        # Handle inline plots first
        if "plot" in chat:
            # Generate a unique key for each plot based on chat index and timestamp
            plot_key = f"plot_{i}_{hash(str(chat.get('timestamp', datetime.now())))}"
            st.plotly_chart(chat["plot"], use_container_width=True, key=plot_key)
        
        # Display extracted table if available
        if "table" in chat and isinstance(chat["table"], pd.DataFrame):
            st.dataframe(chat["table"], use_container_width=True)
        
        # Then display text response
        if isinstance(bot_response, dict):
            # Display the text response if it exists
            if "response" in bot_response:
                st.markdown(f'<div class="chat-bubble bot-bubble">{bot_response["response"]}</div>', unsafe_allow_html=True)
            
            # Handle tabular data
            elif "tabular_data" in bot_response:
                st.dataframe(bot_response["tabular_data"], use_container_width=True)
        else:
            # Display string response
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_response}</div>', unsafe_allow_html=True)
    
    # Chat input
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    # Create a unique key for the input field
    input_key = f"user_input_{st.session_state.input_key}"
    
    # Add JavaScript for Enter key support
    st.markdown("""
    <script>
    // Function to handle Enter key press
    function handleEnterKey() {
        const textInputs = document.querySelectorAll('input[data-testid="stTextInput"]');
        textInputs.forEach(input => {
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    // Find the submit button and click it
                    const submitButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                    if (submitButton) {
                        submitButton.click();
                    }
                }
            });
        });
    }
    
    // Run the function when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', handleEnterKey);
    } else {
        handleEnterKey();
    }
    
    // Also run on Streamlit rerun
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                handleEnterKey();
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Use text_area for better multi-line support and Enter key handling
    with st.form(key=f"chat_form_{st.session_state.input_key}"):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_area(
                "Ask EMMA anything about your data:", 
                key=input_key, 
                placeholder="e.g., 'Show me people under 30' or 'What's the average salary?' (Press Shift+Enter for new line, Enter to send)",
                height=100,
                max_chars=2000
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("➤", use_container_width=True)
    
    # Handle input submission
    if submit_button and user_input:
        # Check if this is a duplicate of the last question
        if (st.session_state.chat_history and 
            st.session_state.chat_history[-1]["user"] == user_input):
            st.warning("You just asked this question. Please ask something different or wait a moment.")
        else:
            with st.spinner("🤔 EMMA is thinking..."):
                df = st.session_state.get('data') if isinstance(st.session_state.get('data'), pd.DataFrame) else None

                # Use deterministic reasoning pipeline
                result = analyze_query_structured(df, user_input)

                # Build a clean textual response
                text_parts: list[str] = []
                if result.get("answer") is not None:
                    text_parts.append(f"**Answer:** {result['answer']}")
                if result.get("explanation"):
                    text_parts.append(result["explanation"])
                bot_text = "\n\n".join(text_parts) if text_parts else "I couldn't compute anything for this question."

                chat_item: Dict[str, Any] = {
                    "user": user_input,
                    "bot": bot_text,
                    "timestamp": datetime.now()
                }
                if isinstance(result.get("table"), pd.DataFrame) and not result["table"].empty:
                    chat_item["table"] = result["table"]
                if result.get("plot") is not None:
                    chat_item["plot"] = result["plot"]

                st.session_state.chat_history.append(chat_item)

                # Save to database (store just the textual part)
                if 'current_session_id' in st.session_state:
                    save_chat_message(
                        st.session_state.current_session_id,
                        user_input,
                        bot_text
                    )

                st.session_state.input_key += 1
                st.rerun()

if __name__ == "__main__":
    main()