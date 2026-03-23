from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app import create_app, db
from app.models import User


@pytest.fixture()
def app(tmp_path: Path):
    upload_dir = tmp_path / "uploads"
    app = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
            "UPLOAD_FOLDER": str(upload_dir),
            "MAX_CONTENT_LENGTH": 25 * 1024 * 1024,
        }
    )

    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def user(app):
    with app.app_context():
        u = User(full_name="Tanveer", email="tanveer@example.com")
        u.set_password("password123")
        db.session.add(u)
        db.session.commit()
        return u


@pytest.fixture()
def logged_in_client(client, user):
    response = client.post(
        "/auth/login",
        data={"email": "tanveer@example.com", "password": "password123"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    return client


def make_test_dicom_bytes(shape=(64, 64), value=100, series_uid=None, series_description="Synthetic QA", instance_number=1) -> BytesIO:
    arr = np.full(shape, value, dtype=np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("test.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Patient"
    ds.Modality = "MR"
    ds.SeriesDescription = series_description
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.InstanceNumber = instance_number
    ds.Rows = arr.shape[0]
    ds.Columns = arr.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 5.0
    ds.MagneticFieldStrength = 1.5
    ds.PixelData = arr.tobytes()

    bio = BytesIO()
    ds.save_as(bio)
    bio.seek(0)
    return bio
