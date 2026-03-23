from conftest import make_test_dicom_bytes


def test_register_login_logout_flow(client):
    response = client.post(
        "/auth/register",
        data={
            "full_name": "Student User",
            "email": "student@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Log in" in response.data or b"Login" in response.data

    response = client.post(
        "/auth/login",
        data={"email": "student@example.com", "password": "secret123"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Dashboard" in response.data

    response = client.get("/auth/logout", follow_redirects=True)
    assert response.status_code == 200
    assert b"ACR" in response.data or b"MRI" in response.data



def test_upload_creates_report(logged_in_client):
    data = {
        "input_files": (make_test_dicom_bytes(), "sample.dcm"),
    }
    response = logged_in_client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/report/" in response.headers["Location"]


def test_delete_and_clear_jobs(logged_in_client):
    data = {"input_files": (make_test_dicom_bytes(), "sample.dcm")}
    response = logged_in_client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=False)
    assert response.status_code == 302
    job_url = response.headers["Location"]
    job_id = int(job_url.rstrip('/').split('/')[-1])

    response = logged_in_client.post(f"/jobs/{job_id}/delete", follow_redirects=True)
    assert response.status_code == 200
    assert b"deleted" in response.data.lower()

    # Create two jobs and clear all
    for idx in range(2):
        data = {"input_files": (make_test_dicom_bytes(value=100+idx), f"sample{idx}.dcm")}
        response = logged_in_client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=False)
        assert response.status_code == 302

    response = logged_in_client.post("/jobs/clear", follow_redirects=True)
    assert response.status_code == 200
    assert b"cleared" in response.data.lower()
