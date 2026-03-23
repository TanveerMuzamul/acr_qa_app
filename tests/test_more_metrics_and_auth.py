import json

import numpy as np
from pydicom.dataset import Dataset

from app.forms import LoginForm, RegisterForm
from app.models import Job, User
from app.services.qa_metrics import build_reasoned_results, compute_metrics_for_slice
from conftest import make_test_dicom_bytes


def test_compute_metrics_handles_missing_optional_tags():
    ds = Dataset()
    img = np.ones((32, 32), dtype=np.uint16)
    raw = compute_metrics_for_slice(ds, img)
    assert raw['image_shape'] == '32 x 32'
    assert raw['slice_thickness_mm'] is None
    assert raw['modality'] is None


def test_build_reasoned_results_passes_when_values_good():
    raw = {
        'modality': 'MR',
        'field_strength_t': 1.5,
        'pixel_spacing': (0.8, 0.8),
        'slice_thickness_mm': 5.0,
        'piu_percent': 90.0,
        'ghosting_ratio': 0.01,
        'lcd_spokes_total': 35,
        'lcd_spokes_per_slice': [8, 9, 9, 9],
        'series_kind': 'T1',
    }
    result = build_reasoned_results(raw)
    assert result['overall_status'] == 'PASS'


def test_build_reasoned_results_fails_missing_spacing():
    raw = {'modality': 'MR', 'slice_thickness_mm': 5.0, 'piu_percent': 90.0, 'ghosting_ratio': 0.01, 'lcd_spokes_total': 35, 'series_kind': 'T1'}
    result = build_reasoned_results(raw)
    assert result['overall_status'] == 'FAIL'
    assert result['metrics'][0]['status'] == 'FAIL'


def test_build_reasoned_results_uses_3t_piu_threshold():
    raw = {
        'modality': 'MR',
        'field_strength_t': 3.0,
        'pixel_spacing': (0.8, 0.8),
        'slice_thickness_mm': 5.0,
        'piu_percent': 80.0,
        'ghosting_ratio': 0.01,
        'lcd_spokes_total': 40,
        'series_kind': 'T1',
    }
    result = build_reasoned_results(raw)
    piu_metric = next(m for m in result['metrics'] if 'PIU' in m['name'])
    assert piu_metric['status'] == 'PASS'


def test_build_reasoned_results_uses_t2_lcd_limit():
    raw = {
        'modality': 'MR',
        'field_strength_t': 1.5,
        'pixel_spacing': (0.8, 0.8),
        'slice_thickness_mm': 5.0,
        'piu_percent': 90.0,
        'ghosting_ratio': 0.01,
        'lcd_spokes_total': 25,
        'series_kind': 'T2',
    }
    result = build_reasoned_results(raw)
    lcd_metric = next(m for m in result['metrics'] if 'Low-contrast' in m['name'])
    assert lcd_metric['status'] == 'PASS'


def test_user_password_hash_roundtrip():
    user = User(full_name='A', email='a@example.com')
    user.set_password('secret123')
    assert user.check_password('secret123') is True
    assert user.check_password('wrong') is False


def test_job_ensure_share_token_generates_value():
    job = Job(user_id=1, job_dir='x')
    token = job.ensure_share_token()
    assert token
    assert job.ensure_share_token() == token


def test_register_form_rejects_short_password(app):
    with app.test_request_context(method='POST', data={'full_name':'User','email':'u@example.com','password':'123','confirm_password':'123'}):
        form = RegisterForm()
        assert form.validate() is False


def test_login_form_requires_email(app):
    with app.test_request_context(method='POST', data={'email':'','password':'x'}):
        form = LoginForm()
        assert form.validate() is False


def test_register_normalizes_duplicate_email(client):
    first = client.post('/auth/register', data={'full_name':'User 1','email':'TEST@example.com','password':'secret123','confirm_password':'secret123'}, follow_redirects=False)
    second = client.post('/auth/register', data={'full_name':'User 2','email':'test@example.com','password':'secret123','confirm_password':'secret123'}, follow_redirects=False)
    assert first.status_code == 302
    assert second.status_code == 302
    assert '/auth/login' in second.headers['Location']


def test_login_rejects_bad_password(client):
    client.post('/auth/register', data={'full_name':'User','email':'user@example.com','password':'secret123','confirm_password':'secret123'}, follow_redirects=False)
    response = client.post('/auth/login', data={'email':'user@example.com','password':'wrong'}, follow_redirects=False)
    assert response.status_code == 401


def test_login_redirects_to_next(client):
    client.post('/auth/register', data={'full_name':'User','email':'user2@example.com','password':'secret123','confirm_password':'secret123'}, follow_redirects=False)
    response = client.post('/auth/login?next=/dashboard', data={'email':'user2@example.com','password':'secret123'}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard')


def test_dashboard_requires_login(client):
    response = client.get('/dashboard', follow_redirects=False)
    assert response.status_code == 302
    assert '/auth/login' in response.headers['Location']


def test_upload_requires_file_selection(logged_in_client):
    response = logged_in_client.post('/upload', data={}, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b'Please choose at least one' in response.data


def test_api_report_rejects_missing_series(logged_in_client):
    response = logged_in_client.post('/upload', data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')}, content_type='multipart/form-data', follow_redirects=False)
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])
    api = logged_in_client.get(f'/api/report/{job_id}')
    assert api.status_code == 400


def test_api_report_rejects_non_integer_slice(logged_in_client):
    response = logged_in_client.post('/upload', data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')}, content_type='multipart/form-data', follow_redirects=False)
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])
    with logged_in_client.application.app_context():
        from app import db
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series = next(s for s in index['series'] if s['series_uid'] != '__ALL__')
    api = logged_in_client.get(f"/api/report/{job_id}?series={series['series_key']}&slice=abc")
    assert api.status_code == 400
