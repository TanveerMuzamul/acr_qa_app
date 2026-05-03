# Contains unit tests for backend routes and MRI analysis logic
from __future__ import annotations

"""Test App module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""


import io
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from dataset import Dcm, Series
from backend.app import (
    APP_PASSWORD,
    APP_USER,
    app,
    as_float,
    build_analysis,
    build_modules,
    classify_series,
    count_pass_fail,
    sanitize_set_name,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    uploaded = tmp_path / 'uploaded_sets'
    cache = tmp_path / '.cache'
    for path in (uploaded, cache):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('backend.app.UPLOAD_DIR', uploaded)
    monkeypatch.setattr('backend.app.CACHE_DIR', cache)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def login(client):
    return client.post('/api/login', json={'username': APP_USER, 'password': APP_PASSWORD})


def make_dcm(name: str, tr=None, te=None, desc=''):
    return Dcm(
        path=Path(name),
        suid=f'suid-{name}',
        iuid=f'iuid-{name}',
        desc=desc,
        proto=desc,
        seq='',
        scan_seq='',
        inst=1,
        echo_no=1,
        te=te,
        tr=tr,
        etl=None,
        rows=256,
        cols=256,
        ps=[1.0, 1.0],
        thick=5.0,
        space=10.0,
        iop=[1, 0, 0, 0, 1, 0],
        ipp=[0, 0, 0],
        maker='x',
        mod='MR',
        b0=1.5,
    )


def make_series(label: str, count: int, tr=None, te=None, desc=''):
    files = [make_dcm(f'{label}_{i}.dcm', tr=tr, te=te, desc=desc) for i in range(1, count + 1)]
    return Series(uid=label, suid=label, echo_no=1, te_key=int(te or 0), files=files, label=label)


def test_sanitize_set_name_basic():
    assert sanitize_set_name(' demo set ') == 'demo_set'


def test_sanitize_set_name_removes_specials():
    assert sanitize_set_name('a/b\\c:*?') == 'a_b_c'


def test_sanitize_set_name_limits_length():
    assert len(sanitize_set_name('a' * 120)) == 80


def test_sanitize_set_name_empty():
    assert sanitize_set_name('***') == ''


def test_as_float_number():
    assert as_float('12.5') == 12.5


def test_as_float_none():
    assert as_float(None) is None


def test_as_float_bad_value():
    assert as_float('bad') is None


def test_login_success(client):
    response = login(client)
    assert response.status_code == 200
    assert response.get_json()['ok'] is True


def test_login_failure(client):
    response = client.post('/api/login', json={'username': 'x', 'password': 'y'})
    assert response.status_code == 401


def test_session_requires_auth(client):
    response = client.get('/api/sets')
    assert response.status_code == 401


def test_health_open(client):
    response = client.get('/health')
    assert response.status_code == 200


def test_root_shows_login_when_logged_out(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Sign in' in response.data


def test_session_after_login(client):
    login(client)
    response = client.get('/api/session')
    assert response.get_json()['authenticated'] is True


def test_logout(client):
    login(client)
    response = client.post('/api/logout')
    assert response.status_code == 200
    response = client.get('/api/session')
    assert response.get_json()['authenticated'] is False


def test_empty_set_list(client):
    login(client)
    response = client.get('/api/sets')
    assert response.status_code == 200
    assert response.get_json() == []


def test_upload_requires_name(client):
    login(client)
    response = client.post('/api/upload', data={})
    assert response.status_code == 400


def test_upload_requires_zip(client):
    login(client)
    response = client.post('/api/upload', data={'set_name': 'demo'})
    assert response.status_code == 400


def test_upload_rejects_non_zip(client):
    login(client)
    response = client.post('/api/upload', data={'set_name': 'demo', 'zip_file': (io.BytesIO(b'abc'), 'a.txt')})
    assert response.status_code == 400


def test_upload_rejects_fake_dicom(client):
    login(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('x.txt', 'hello')
    buf.seek(0)
    response = client.post('/api/upload', data={'set_name': 'demo', 'zip_file': (buf, 'demo.zip')}, content_type='multipart/form-data')
    assert response.status_code == 400


def test_get_sets_after_folder_exists(client):
    login(client)
    from backend import app as module
    (module.UPLOAD_DIR / 'set_001').mkdir()
    response = client.get('/api/sets')
    assert response.status_code == 200
    assert response.get_json()[0]['name'] == 'set_001'


def test_delete_missing_set(client):
    login(client)
    response = client.delete('/api/sets/none')
    assert response.status_code == 404


def test_delete_uploaded_set(client):
    login(client)
    from backend import app as module
    (module.UPLOAD_DIR / 'demo').mkdir()
    response = client.delete('/api/sets/demo')
    assert response.status_code == 200
    assert not (module.UPLOAD_DIR / 'demo').exists()



def test_analyze_missing_set(client):
    login(client)
    response = client.get('/api/analyze/none')
    assert response.status_code == 404


def test_classify_series_prefers_t1_t2_and_localizer():
    localizer = make_series('loc', 3, tr=700, te=60, desc='localizer')
    t1 = make_series('t1', 11, tr=500, te=10, desc='acr t1')
    t2 = make_series('t2', 11, tr=2000, te=80, desc='acr t2')
    result = classify_series([localizer, t1, t2])
    assert result['localizer'] is localizer
    assert result['acr_t1'] is t1
    assert result['acr_t2'] is t2


def test_count_pass_fail():
    passed, failed = count_pass_fail([
        {'status': 'PASS'}, {'status': 'FAIL'}, {'status': 'FAIL'}, {'status': 'PASS'}
    ])
    assert passed == 2
    assert failed == 2


def test_build_analysis_contains_report_url():
    data = {'localizer': None, 'acr_t1': None, 'acr_t2': None, 'dicom_count': 0}
    with app.test_request_context('/'):
        result = build_analysis('demo', data, 'uploaded_sets')
    assert result['report_url'].endswith('/api/report/demo.pdf')


def test_build_modules_fail_when_no_series():
    modules = build_modules({'localizer': None, 'acr_t1': None, 'acr_t2': None})
    assert len(modules) == 6
    assert all(module['status'] == 'FAIL' for module in modules)


def test_root_shows_app_when_logged_in(client):
    login(client)
    response = client.get('/')
    assert response.status_code == 200
    assert b'ACR Phantom Analysis Dashboard' in response.data


def test_login_page_no_admin_word(client):
    response = client.get('/')
    assert b'admin' not in response.data.lower()


def test_requirements_file_exists():
    assert (PROJECT_ROOT / 'requirements.txt').exists()


def test_run_file_exists():
    assert (PROJECT_ROOT / 'run.py').exists()


def test_readme_has_windows_activate_command():
    text = (PROJECT_ROOT / 'README.md').read_text()
    assert '.venv\\Scripts\\activate' in text


def test_readme_uses_run_py():
    text = (PROJECT_ROOT / 'README.md').read_text()
    assert 'python run.py' in text


def test_build_analysis_uses_uploaded_source():
    data = {'localizer': None, 'acr_t1': None, 'acr_t2': None, 'dicom_count': 0}
    with app.test_request_context('/'):
        result = build_analysis('demo', data, 'uploaded_sets')
    assert result['source'] == 'uploaded_sets'
