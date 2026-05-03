# Main Flask application
# Handles login, file upload, MRI analysis, and report generation
from __future__ import annotations

"""App module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


import hashlib
import hmac
import io
import json
import os
import secrets
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset import Dcm, Series, gap_mm, group_series, load_px, plane, read_one
from algorithms.slice_position import need_slice_pos, slice_pos_measure
from algorithms.slice_thickness import need_slice_thickness, slice_thickness_measure
from algorithms.geometric_accuracy import need_geometric, evaluate_geometric
from algorithms.ghosting import need_ghosting, ghosting_measure
from algorithms.piu import need_piu, piu_measure
from algorithms.high_contrast_resolution import need_resolution, resolution_measure

# local helper because dataset.safe_float does not exist

def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static'),
)
app.config['SECRET_KEY'] = os.getenv('MRI_APP_SECRET', 'mri-qa-secret-key')
app.config['APP_START_TOKEN'] = secrets.token_hex(16)

UPLOAD_DIR = BASE_DIR / 'uploaded_sets'
CACHE_DIR = BASE_DIR / '.cache'
# Registered users are stored at runtime. The JSON file is generated automatically and is not packaged with the source code.
DATA_DIR = Path(os.getenv('MRI_APP_DATA_DIR', BASE_DIR / 'data'))
USER_STORE = DATA_DIR / 'users.json'
UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
DATA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

APP_USER = os.getenv('MRI_APP_USER', 'user')
APP_PASSWORD = os.getenv('MRI_APP_PASSWORD', 'User@123')


def _hash_password(password: str, salt: str | None = None) -> dict[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 120_000).hex()
    return {'salt': salt, 'hash': digest}


def _verify_password(password: str, record: dict[str, str]) -> bool:
    if not record or 'salt' not in record or 'hash' not in record:
        return False
    candidate = _hash_password(password, record['salt'])['hash']
    return hmac.compare_digest(candidate, record['hash'])


def load_users() -> dict[str, dict[str, str]]:
    if not USER_STORE.exists():
        users = {APP_USER: _hash_password(APP_PASSWORD)}
        USER_STORE.write_text(json.dumps(users, indent=2), encoding='utf-8')
        return users
    try:
        data = json.loads(USER_STORE.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_users(users: dict[str, dict[str, str]]) -> None:
    USER_STORE.write_text(json.dumps(users, indent=2), encoding='utf-8')


def validate_credentials(username: str, password: str) -> bool:
    # Keep environment credentials working, then support registered users.
    if username == APP_USER and password == APP_PASSWORD:
        return True
    return _verify_password(password, load_users().get(username, {}))


def status_pass_fail(status: str | None) -> str:
    return 'PASS' if status == 'PASS' else 'FAIL'


@dataclass
class SetLocation:
    path: Path
    source: str


ROLE_LABELS = {
    'localizer': 'Localizer',
    'acr_t1': 'ACR T1',
    'acr_t2': 'ACR T2',
}


def sanitize_set_name(name: str) -> str:
    clean = re.sub(r'[^A-Za-z0-9._-]+', '_', (name or '').strip())
    clean = clean.strip('._-')
    return clean[:80]


def auth_required() -> bool:
    if session.get('app_start_token') != app.config['APP_START_TOKEN']:
        session.clear()
        return False
    return session.get('logged_in') is True


@app.before_request
def protect_routes():
    if request.path.startswith('/static/'):
        return None
    if request.path in {'/api/login', '/api/register', '/api/session', '/health'}:
        return None
    if request.path == '/':
        return None
    if not auth_required() and request.path.startswith('/api/'):
        return jsonify({'error': 'Authentication required'}), 401
    return None


@app.get('/')
def index():
    if auth_required():
        return render_template('app.html')
    return render_template('login.html')


@app.get('/health')
def health():
    return jsonify({'status': 'ok'})


@app.post('/api/login')
def api_login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    if validate_credentials(username, password):
        session['logged_in'] = True
        session['username'] = username
        session['app_start_token'] = app.config['APP_START_TOKEN']
        return jsonify({'ok': True, 'username': username})
    return jsonify({'error': 'Invalid username or password'}), 401


@app.post('/api/register')
def api_register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    if not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}', username):
        return jsonify({'error': 'Username must be 3-40 letters, numbers, dots, dashes, or underscores'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    users = load_users()
    if username in users or username == APP_USER:
        return jsonify({'error': 'Username already exists'}), 400
    users[username] = _hash_password(password)
    save_users(users)
    session['logged_in'] = True
    session['username'] = username
    session['app_start_token'] = app.config['APP_START_TOKEN']
    return jsonify({'ok': True, 'username': username})


@app.post('/api/logout')
def api_logout():
    session.clear()
    return jsonify({'ok': True})


@app.get('/api/session')
def api_session():
    return jsonify({'authenticated': auth_required(), 'username': session.get('username')})


def find_set_folder(set_name: str) -> SetLocation | None:
    path = UPLOAD_DIR / set_name
    if path.exists() and path.is_dir():
        return SetLocation(path=path, source='uploaded_sets')
    return None


def is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def extract_zip_safely(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            member_path = dest / info.filename
            if not is_within_directory(dest, member_path):
                raise ValueError('Unsafe zip content detected')
            zf.extract(info, dest)


def detect_real_mr_dicoms(root: Path) -> list[Dcm]:
    found: list[Dcm] = []
    for file_path in root.rglob('*'):
        if not file_path.is_file():
            continue
        dcm = read_one(file_path)
        if dcm is None:
            continue
        if dcm.mod != 'MR':
            continue
        found.append(dcm)
    return found


def classify_series(series: list[Series]) -> dict[str, Series | None]:
    ranked = sorted(series, key=lambda s: (s.n, s.first.inst or -1), reverse=True)
    result: dict[str, Series | None] = {'localizer': None, 'acr_t1': None, 'acr_t2': None}

    def score_localizer(s: Series) -> tuple:
        d = s.first
        text = f"{d.desc} {d.proto} {d.seq}".lower()
        return (
            1 if 'localizer' in text or 'localiser' in text else 0,
            1 if s.n in (1, 3) else 0,
            s.n,
        )

    def score_t1(s: Series) -> tuple:
        d = s.first
        text = f"{d.desc} {d.proto} {d.seq}".lower()
        return (
            1 if 't1' in text else 0,
            1 if d.tr is not None and abs(float(d.tr) - 500) <= 180 else 0,
            1 if d.te is not None and abs(float(d.te) - 10) <= 15 else 0,
            1 if s.n >= 11 else 0,
            s.n,
        )

    def score_t2(s: Series) -> tuple:
        d = s.first
        text = f"{d.desc} {d.proto} {d.seq}".lower()
        return (
            1 if 't2' in text else 0,
            1 if d.tr is not None and abs(float(d.tr) - 2000) <= 500 else 0,
            1 if d.te is not None and abs(float(d.te) - 80) <= 20 else 0,
            1 if s.n >= 11 else 0,
            s.n,
        )

    if ranked:
        result['localizer'] = max(ranked, key=score_localizer)
        result['acr_t1'] = max(ranked, key=score_t1)
        result['acr_t2'] = max(ranked, key=score_t2)

    for role, series_obj in result.items():
        if series_obj is not None:
            series_obj.label = role
            series_obj.score = float(series_obj.n)
    return result


def build_dataset_from_any_set(set_path: Path) -> dict[str, Any]:
    files = detect_real_mr_dicoms(set_path)
    grouped = sorted(group_series(files), key=lambda s: (s.first.desc.lower(), s.n, s.first.inst or -1))
    selected = classify_series(grouped)
    return {
        'localizer': selected['localizer'],
        'acr_t1': selected['acr_t1'],
        'acr_t2': selected['acr_t2'],
        'site_t1': None,
        'site_t2': None,
        'unknown': [],
        # Keep every valid MR series for the Dataset Overview. Modules still use the
        # automatically selected ACR Localizer/T1/T2 series above.
        'all': grouped,
        'dicom_count': len(files),
    }



def folder_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for item in path.rglob('*'):
        try:
            latest = max(latest, item.stat().st_mtime)
        except OSError:
            pass
    return latest


def build_dataset_cached(set_name: str, set_path: Path) -> dict[str, Any]:
    mtime = folder_mtime(set_path)
    cached = DATA_CACHE.get(set_name)
    if cached and cached[0] == mtime:
        return cached[1]
    data = build_dataset_from_any_set(set_path)
    DATA_CACHE[set_name] = (mtime, data)
    return data

def slice_payload(series: Series | None, set_name: str, role: str, title: str | None = None) -> dict[str, Any]:
    display_title = title or ROLE_LABELS.get(role, role)
    if series is None:
        return {'available': False, 'title': display_title, 'slices': []}
    d = series.first
    return {
        'available': True,
        'title': display_title,
        'series_uid': series.suid,
        'count': series.n,
        'preview_url': url_for('get_slice_image', set_name=set_name, role=role, slice_no=1),
        'slices': [
            {
                'number': i,
                'instance': item.inst,
                'filename': item.path.name,
                'image_url': url_for('get_slice_image', set_name=set_name, role=role, slice_no=i),
            }
            for i, item in enumerate(series.files, start=1)
        ],
        'metadata': {
            'tr': as_float(d.tr),
            'te': as_float(d.te),
            'thickness': as_float(d.thick),
            'gap': as_float(gap_mm(d)),
            'description': d.desc or None,
            'protocol': d.proto or None,
            'plane': plane(d.iop),
            'rows': d.rows,
            'cols': d.cols,
        },
    }


def format_measurement(value: Any, suffix: str = '') -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.2f}{suffix}'
    return f'{value}{suffix}'


def build_modules(data: dict[str, Any]) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []

    if need_geometric(data.get('localizer'), data.get('acr_t1')):
        g = evaluate_geometric(data['localizer'], data['acr_t1'])
        checks = [
            {
                'label': item['name'],
                'value': f"{item['measured_mm']:.2f} mm",
                'target': f"{item['expected_mm']:.2f} mm",
                'result': status_pass_fail(item['status']),
                'why': f"Error {item['error_mm']:+.2f} mm",
            }
            for item in g['checks']
        ]
        modules.append({
            'name': 'Geometric Accuracy',
            'status': status_pass_fail(g['status']),
            'why': g.get('reason') or f"{len(checks)} measurements checked against ACR tolerance.",
            'measurements': checks,
        })
    else:
        modules.append({'name': 'Geometric Accuracy', 'status': 'FAIL', 'why': 'Required series not available.', 'measurements': []})

    thickness_measurements: list[dict[str, Any]] = []
    thickness_status = 'FAIL'
    seen = False
    any_fail = False
    for role in ('acr_t1', 'acr_t2'):
        s = data.get(role)
        if need_slice_thickness(s):
            m = slice_thickness_measure(s, 1)
            seen = True
            any_fail = any_fail or status_pass_fail(m['status']) == 'FAIL'
            thickness_measurements.append({
                'label': ROLE_LABELS[role],
                'value': f"{m['thickness_mm']:.2f} mm",
                'target': '5.00 mm',
                'result': status_pass_fail(m['status']),
                'why': f"Top {m['top_len_mm']:.2f} mm / Bottom {m['bottom_len_mm']:.2f} mm",
            })
    if seen:
        thickness_status = 'FAIL' if any_fail else 'PASS'
    modules.append({'name': 'Slice Thickness', 'status': thickness_status, 'why': 'Uses slice 1 ramp lengths.', 'measurements': thickness_measurements})

    pos_measurements: list[dict[str, Any]] = []
    pos_status = 'FAIL'
    seen = False
    any_fail = False
    for role in ('acr_t1', 'acr_t2'):
        s = data.get(role)
        if need_slice_pos(s):
            for idx in (1, 11):
                m = slice_pos_measure(s, idx)
                seen = True
                any_fail = any_fail or status_pass_fail(m['status']) == 'FAIL'
                pos_measurements.append({
                    'label': f"{ROLE_LABELS[role]} slice {idx}",
                    'value': f"{m['slice_disp_mm']:+.2f} mm",
                    'target': 'Within tolerance',
                    'result': status_pass_fail(m['status']),
                    'why': f"Bar diff {m['bar_diff_mm']:+.2f} mm",
                })
    if seen:
        pos_status = 'FAIL' if any_fail else 'PASS'
    modules.append({'name': 'Slice Position', 'status': pos_status, 'why': 'Checks slices 1 and 11.', 'measurements': pos_measurements})

    ghost_measurements: list[dict[str, Any]] = []
    ghost_status = 'FAIL'
    if need_ghosting(data.get('acr_t1')):
        m = ghosting_measure(data['acr_t1'], 7)
        ghost_status = status_pass_fail(m['status'])
        ghost_measurements = [{
            'label': 'ACR T1 slice 7',
            'value': f"{m['psg_percent']:.2f}%",
            'target': 'Lower is better',
            'result': status_pass_fail(m['status']),
            'why': f"Ratio {m['ghosting_ratio']:.5f}",
        }]
    modules.append({'name': 'Percent-Signal Ghosting', 'status': ghost_status, 'why': 'Uses central phantom slice.', 'measurements': ghost_measurements})

    piu_measurements: list[dict[str, Any]] = []
    piu_status = 'FAIL'
    seen = False
    any_fail = False
    for role in ('acr_t1', 'acr_t2'):
        s = data.get(role)
        if need_piu(s):
            m = piu_measure(s, 7)
            seen = True
            any_fail = any_fail or status_pass_fail(m['status']) == 'FAIL'
            piu_measurements.append({
                'label': ROLE_LABELS[role],
                'value': f"{m['piu']:.2f}%",
                'target': f">= {m['limits']['target']:.1f}%",
                'result': status_pass_fail(m['status']),
                'why': f"High {m['high']:.2f} / Low {m['low']:.2f}",
            })
    if seen:
        piu_status = 'FAIL' if any_fail else 'PASS'
    modules.append({'name': 'Image Intensity Uniformity', 'status': piu_status, 'why': 'Central ROI uniformity check.', 'measurements': piu_measurements})

    res_measurements: list[dict[str, Any]] = []
    res_status = 'FAIL'
    seen = False
    any_fail = False
    for role in ('acr_t1', 'acr_t2'):
        s = data.get(role)
        if need_resolution(s):
            m = resolution_measure(s, 1)
            seen = True
            any_fail = any_fail or status_pass_fail(m['status']) == 'FAIL'
            res_measurements.append({
                'label': ROLE_LABELS[role],
                'value': f"RL {m['right_left_mm']} mm / TB {m['top_bottom_mm']} mm",
                'target': 'Visual pattern resolved',
                'result': status_pass_fail(m['status']),
                'why': 'Based on high-contrast insert visibility.',
            })
    if seen:
        res_status = 'FAIL' if any_fail else 'PASS'
    modules.append({'name': 'High-Contrast Spatial Resolution', 'status': res_status, 'why': 'Resolution insert check.', 'measurements': res_measurements})

    return modules


def count_pass_fail(modules: list[dict[str, Any]]) -> tuple[int, int]:
    passed = sum(1 for module in modules if status_pass_fail(module.get('status')) == 'PASS')
    failed = sum(1 for module in modules if status_pass_fail(module.get('status')) == 'FAIL')
    return passed, failed


def build_analysis(set_name: str, data: dict[str, Any], source: str) -> dict[str, Any]:
    modules = build_modules(data)
    passed, failed = count_pass_fail(modules)
    return {
        'name': set_name,
        'source': source,
        'validation': f"{data.get('dicom_count', 0)} valid MR DICOM files detected",
        'passed': passed,
        'failed': failed,
        'overview': {
            role: slice_payload(data.get(role), set_name, role)
            for role in ('localizer', 'acr_t1', 'acr_t2')
        },
        'all_series': [
            slice_payload(series, set_name, f'series_{idx}', title=(series.first.desc or f'Series {idx + 1}'))
            for idx, series in enumerate(data.get('all', []))
        ],
        'modules': modules,
        'report_url': url_for('download_report', set_name=set_name),
    }


@app.get('/api/sets')
def get_sets():
    sets: list[dict[str, str]] = []
    for folder in sorted(UPLOAD_DIR.iterdir()) if UPLOAD_DIR.exists() else []:
        if folder.is_dir():
            sets.append({'name': folder.name, 'source': 'uploaded_sets'})
    return jsonify(sets)


@app.delete('/api/sets/<set_name>')
def delete_set(set_name: str):
    location = find_set_folder(set_name)
    if location is None:
        return jsonify({'error': 'Set not found'}), 404
    shutil.rmtree(location.path, ignore_errors=True)
    DATA_CACHE.pop(set_name, None)
    for item in CACHE_DIR.glob(f'{set_name}_*.png'):
        item.unlink(missing_ok=True)
    return jsonify({'ok': True})


@app.get('/api/analyze/<set_name>')
def analyze_set(set_name: str):
    location = find_set_folder(set_name)
    if location is None:
        return jsonify({'error': 'Set not found'}), 404
    try:
        data = build_dataset_cached(set_name, location.path)
        if not data['all']:
            return jsonify({'error': 'No valid MR DICOM files found in this set'}), 400
        return jsonify(build_analysis(set_name, data, 'uploaded_sets'))
    except Exception as exc:
        return jsonify({'error': f'Analysis failed: {exc}'}), 500


@app.post('/api/upload')
def upload_set():
    raw_name = request.form.get('set_name', '')
    zip_file = request.files.get('zip_file')
    set_name = sanitize_set_name(raw_name)

    if not set_name:
        return jsonify({'error': 'Missing or invalid set name'}), 400
    if zip_file is None or not zip_file.filename:
        return jsonify({'error': 'Missing zip file'}), 400
    if not zip_file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Only zip files are allowed'}), 400

    set_dir = UPLOAD_DIR / set_name
    if set_dir.exists():
        return jsonify({'error': 'Set name already exists'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        zip_path = temp_root / zip_file.filename
        zip_file.save(zip_path)
        extracted = temp_root / 'extracted'
        extracted.mkdir()
        try:
            extract_zip_safely(zip_path, extracted)
        except Exception as exc:
            return jsonify({'error': f'Failed to unzip file: {exc}'}), 400

        dicoms = detect_real_mr_dicoms(extracted)
        if not dicoms:
            return jsonify({'error': 'No real MR DICOM files were found. Broken, fake, or non-MR files were ignored.'}), 400

        set_dir.mkdir(parents=True, exist_ok=True)
        for item in extracted.iterdir():
            target = set_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    DATA_CACHE.pop(set_name, None)
    return jsonify({'message': 'Upload successful', 'set_name': set_name, 'source': 'uploaded_sets'})


@app.get('/api/sets/<set_name>/slice-image/<role>/<int:slice_no>.png')
def get_slice_image(set_name: str, role: str, slice_no: int):
    location = find_set_folder(set_name)
    if location is None:
        abort(404)
    data = build_dataset_cached(set_name, location.path)
    if role in ROLE_LABELS:
        series = data.get(role)
    elif role.startswith('series_'):
        try:
            series = data.get('all', [])[int(role.split('_', 1)[1])]
        except Exception:
            series = None
    else:
        series = None
    if series is None or slice_no < 1 or slice_no > series.n:
        abort(404)

    safe_cache_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"{set_name}_{role}_{slice_no}")
    cache_file = CACHE_DIR / f"{safe_cache_name}.png"
    if not cache_file.exists():
        arr = load_px(series.files[slice_no - 1]).astype('float32')
        lo = float(arr.min())
        hi = float(arr.max())
        if hi <= lo:
            scaled = (arr * 0).astype('uint8')
        else:
            scaled = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype('uint8')
        Image.fromarray(scaled, mode='L').save(cache_file, format='PNG', optimize=True)
    return send_file(cache_file, mimetype='image/png')


@app.get('/api/report/<set_name>.pdf')
def download_report(set_name: str):
    location = find_set_folder(set_name)
    if location is None:
        abort(404)
    data = build_dataset_cached(set_name, location.path)
    analysis = build_analysis(set_name, data, 'uploaded_sets')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleModern', parent=styles['Title'], textColor=colors.HexColor('#173b70'), fontSize=20, leading=24)
    section_style = ParagraphStyle('SectionModern', parent=styles['Heading2'], textColor=colors.HexColor('#214f9a'), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyModern', parent=styles['BodyText'], fontSize=9.5, leading=13)
    header_cell_style = ParagraphStyle('HeaderCell', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold')

    story: list[Any] = []
    story.append(Paragraph('MRI Phantom QA Report', title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Set: <b>{set_name}</b>", body_style))
    story.append(Paragraph(f"Validation: {analysis['validation']}", body_style))
    story.append(Paragraph(f"Passed modules: {analysis['passed']} | Failed modules: {analysis['failed']}", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph('Dataset Overview', section_style))
    story.append(Paragraph('All valid MR series detected in the uploaded ZIP are listed below.', body_style))
    overview_rows = [[
        Paragraph('Series', header_cell_style), Paragraph('Slices', header_cell_style), Paragraph('TR', header_cell_style),
        Paragraph('TE', header_cell_style), Paragraph('Thick', header_cell_style), Paragraph('Gap', header_cell_style),
        Paragraph('Description', header_cell_style)
    ]]
    for item in analysis.get('all_series', []):
        meta = item.get('metadata', {})
        overview_rows.append([
            Paragraph(item.get('title') or '-', body_style),
            Paragraph(str(item.get('count', '-')), body_style),
            Paragraph(format_measurement(meta.get('tr')), body_style),
            Paragraph(format_measurement(meta.get('te')), body_style),
            Paragraph(format_measurement(meta.get('thickness')), body_style),
            Paragraph(format_measurement(meta.get('gap')), body_style),
            Paragraph(meta.get('description') or '-', body_style),
        ])
    overview_table = Table(overview_rows, colWidths=[36 * mm, 15 * mm, 17 * mm, 17 * mm, 16 * mm, 14 * mm, 63 * mm], repeatRows=1)
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#173b70')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c9d6ea')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f9ff')]),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph('Module Summary', section_style))
    module_rows = [[Paragraph('Module', header_cell_style), Paragraph('Status', header_cell_style), Paragraph('Reason', header_cell_style), Paragraph('Key Measurement', header_cell_style)]]
    for module in analysis['modules']:
        key = module['measurements'][0]['value'] if module['measurements'] else '-'
        module_rows.append([
            Paragraph(module['name'], body_style),
            Paragraph(module['status'], body_style),
            Paragraph(module['why'], body_style),
            Paragraph(key, body_style),
        ])
    module_table = Table(module_rows, colWidths=[46 * mm, 20 * mm, 70 * mm, 42 * mm], repeatRows=1)
    module_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#214f9a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d7e1f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fbff')]),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(module_table)

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f'{set_name}_report.pdf', mimetype='application/pdf')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
