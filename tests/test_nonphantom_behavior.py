import json
import io
import zipfile

from app.services.qa_metrics import build_reasoned_results
from conftest import make_test_dicom_bytes


def test_build_reasoned_results_passes_non_phantom_slice():
    raw = {
        'modality': 'MR',
        'field_strength_t': 1.5,
        'is_phantom_like': False,
        'pixel_spacing': (2.0, 2.0),
        'slice_thickness_mm': 2.0,
        'piu_percent': 10.0,
        'ghosting_ratio': 0.5,
        'lcd_spokes_total': 0,
        'series_kind': 'T1',
    }
    result = build_reasoned_results(raw)
    assert result['overall_status'] == 'PASS'
    assert all(m['status'] == 'PASS' for m in result['metrics'])
    assert result['overall_reason'] == 'Slice analysis complete. Results available below.'


def test_api_report_returns_non_phantom_pass_for_anatomical_like_slice(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(value=100), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])

    from app.models import Job
    from app import db
    with logged_in_client.application.app_context():
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series_id = index['series'][0]['series_key']

    api = logged_in_client.get(f'/api/report/{job_id}?series={series_id}&slice=0')
    payload = api.get_json()
    assert api.status_code == 200
    assert payload['overall_status'] == 'PASS'


def test_api_report_exposes_ignored_non_dicom_count(logged_in_client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('folder/a.txt', 'ignore')
        zf.writestr('folder/b.dcm', make_test_dicom_bytes().getvalue())
    buf.seek(0)

    response = logged_in_client.post('/upload', data={'input_files': (buf, 'mixed.zip')}, content_type='multipart/form-data', follow_redirects=False)
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])

    from app.models import Job
    from app import db
    with logged_in_client.application.app_context():
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series_id = index['series'][0]['series_key']

    api = logged_in_client.get(f'/api/report/{job_id}?series={series_id}&slice=0')
    payload = api.get_json()
    assert payload['ignored_non_dicom_files'] == 1


def test_report_page_series_list_has_no_all_uploaded_option(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    page = logged_in_client.get(response.headers['Location'])
    assert page.status_code == 200
    assert b'All uploaded files' not in page.data
