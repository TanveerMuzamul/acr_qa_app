import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from app.services.dicom_service import index_uploaded_content, load_series_slice
from app.services.qa_metrics import acr_lcd_spokes_total, phantom_likeness_from_image
from conftest import make_test_dicom_bytes


def _write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_image_bytes(fmt='PNG', size=(32, 32), value=128):
    img = Image.new('L', size, color=value)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_index_uploaded_content_assigns_series_keys(tmp_path):
    d1 = tmp_path / 'a.dcm'
    d2 = tmp_path / 'b.dcm'
    _write_bytes(d1, make_test_dicom_bytes().getvalue())
    _write_bytes(d2, make_test_dicom_bytes().getvalue())

    index = index_uploaded_content([str(d1), str(d2)], str(tmp_path))

    assert index['series']
    assert all('series_key' in s for s in index['series'])
    assert any(str(s['series_key']).startswith('series-') for s in index['series'] if s['series_uid'] not in {'__ALL__', '__NORMAL_IMAGES__'})


def test_index_uploaded_content_groups_normal_images(tmp_path):
    p1 = tmp_path / 'scan.png'
    p2 = tmp_path / 'scan2.jpg'
    _write_bytes(p1, _make_image_bytes('PNG'))
    _write_bytes(p2, _make_image_bytes('JPEG'))

    index = index_uploaded_content([str(p1), str(p2)], str(tmp_path))

    assert index['ignored_non_dicom_files'] == 0
    assert len(index['series']) == 1
    assert index['series'][0]['series_uid'] == '__NORMAL_IMAGES__'


def test_load_series_slice_supports_normal_image_index(tmp_path):
    p1 = tmp_path / 'scan.png'
    _write_bytes(p1, _make_image_bytes('PNG', size=(40, 24), value=90))
    index = index_uploaded_content([str(p1)], str(tmp_path))

    assert index['series'][0]['series_uid'] == '__NORMAL_IMAGES__'
    ds, img = load_series_slice(str(tmp_path), '__NORMAL_IMAGES__', 0)
    assert ds.Modality == 'OT'
    assert tuple(img.shape) == (24, 40)


def test_phantom_likeness_handles_small_images():
    import numpy as np
    result = phantom_likeness_from_image(np.zeros((8, 8), dtype='uint8'))
    assert result['is_phantom_like'] is False
    assert result['phantom_score'] == 0.0


def test_acr_lcd_spokes_total_returns_expected_keys():
    import numpy as np
    imgs = [np.zeros((64, 64), dtype='uint8') for _ in range(4)]
    result = acr_lcd_spokes_total(imgs)
    assert 'lcd_spokes_total' in result
    assert 'lcd_spokes_per_slice' in result
    assert len(result['lcd_spokes_per_slice']) == 4


def test_upload_zip_with_path_traversal_is_rejected(logged_in_client, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('../escape.txt', 'bad')
    buf.seek(0)

    response = logged_in_client.post(
        '/upload',
        data={'input_files': (buf, 'bad.zip')},
        content_type='multipart/form-data',
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b'Upload failed' in response.data


def test_report_page_hides_series_uid(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    page = logged_in_client.get(response.headers['Location'])
    assert page.status_code == 200
    assert b'Series UID' not in page.data


def test_api_report_uses_series_id_not_uid(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])

    report_page = logged_in_client.get(f'/report/{job_id}')
    assert report_page.status_code == 200

    from app.models import Job
    from app import db
    with logged_in_client.application.app_context():
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series = next(s for s in index['series'] if s['series_uid'] != '__ALL__')
        series_id = series['series_key']

    api = logged_in_client.get(f'/api/report/{job_id}?series={series_id}&slice=0')
    payload = api.get_json()
    assert api.status_code == 200
    assert payload['series_id'] == series_id
    assert 'series_uid' not in payload


def test_download_json_uses_series_id_selection(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])

    from app.models import Job
    from app import db
    with logged_in_client.application.app_context():
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series = next(s for s in index['series'] if s['series_uid'] != '__ALL__')
        series_id = series['series_key']

    res = logged_in_client.get(f'/report/{job_id}/download.json?series_id={series_id}&slice_index=0')
    payload = json.loads(res.data)
    assert res.status_code == 200
    assert payload['selection']['series_id'] == series_id
    assert 'series_uid' not in payload['selection']


def test_slice_png_route_accepts_public_series_id(logged_in_client):
    response = logged_in_client.post(
        '/upload',
        data={'input_files': (make_test_dicom_bytes(), 'sample.dcm')},
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    job_id = int(response.headers['Location'].rstrip('/').split('/')[-1])

    from app.models import Job
    from app import db
    with logged_in_client.application.app_context():
        job = db.session.get(Job, job_id)
        index = json.loads(job.summary_json)
        series = next(s for s in index['series'] if s['series_uid'] != '__ALL__')
        series_id = series['series_key']

    png = logged_in_client.get(f'/slice/{job_id}/{series_id}/0.png')
    assert png.status_code == 200
    assert png.mimetype == 'image/png'


def test_index_uploaded_content_drops_merged_all_series(tmp_path):
    d1 = tmp_path / 'a.dcm'
    d2 = tmp_path / 'b.dcm'
    _write_bytes(d1, make_test_dicom_bytes().getvalue())
    _write_bytes(d2, make_test_dicom_bytes().getvalue())
    index = index_uploaded_content([str(d1), str(d2)], str(tmp_path))
    assert all(s['series_uid'] != '__ALL__' for s in index['series'])


def test_low_contrast_metric_is_not_a_fail_when_missing():
    from app.services.qa_metrics import build_reasoned_results
    raw = {
        'modality': 'MR',
        'field_strength_t': 1.5,
        'pixel_spacing': (0.8, 0.8),
        'slice_thickness_mm': 5.0,
        'piu_percent': 90.0,
        'ghosting_ratio': 0.01,
        'series_kind': 'T1',
    }
    result = build_reasoned_results(raw)
    lcd_metric = next(m for m in result['metrics'] if 'Low-contrast' in m['name'])
    assert lcd_metric['status'] == 'PASS'


def test_index_uploaded_content_filters_out_localizer_series(tmp_path):
    d1 = tmp_path / 'localizer.dcm'
    d2 = tmp_path / 't1.dcm'
    _write_bytes(d1, make_test_dicom_bytes(series_uid='1.2.3.4.1', series_description='Localizer', instance_number=1).getvalue())
    _write_bytes(d2, make_test_dicom_bytes(series_uid='1.2.3.4.2', series_description='ACR T1 AXIAL', instance_number=1).getvalue())
    index = index_uploaded_content([str(d1), str(d2)], str(tmp_path))
    assert len(index['series']) == 1
    assert index['series'][0]['description'] == 'ACR T1 AXIAL'
